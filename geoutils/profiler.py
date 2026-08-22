# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains profiling functions."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import wraps
from multiprocessing import Pipe, connection
from threading import Event, Thread
from typing import Any, Callable

import pandas as pd

from geoutils._misc import import_optional

MB = 1_000_000


@dataclass
class ProfileMetrics:
    """
    Normalized profiling metrics for one measured call.

    Memory values are reported in decimal megabytes. Dask worker metrics are the cluster aggregate reported by
    :class:`distributed.diagnostics.MemorySampler` when a distributed client is active.
    """

    runtime_s: float
    peak_client_rss_mb: float
    peak_dask_worker_process_rss_mb: float | None = None
    peak_dask_spilled_mb: float | None = None
    client_rss_mb: list[tuple[float, float]] = field(default_factory=list)
    dask_worker_process_rss_mb: list[tuple[float, float]] = field(default_factory=list)
    dask_spilled_mb: list[tuple[float, float]] = field(default_factory=list)
    dask_client_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a plain dictionary."""

        return asdict(self)


class _ProcessRSSSampler(Thread):
    """Sample RSS memory for one process in the background."""

    def __init__(self, pid: int, interval: float) -> None:
        psutil = import_optional("psutil")

        super().__init__(daemon=True)
        self.interval = interval
        self.process = psutil.Process(pid)
        self.samples_mb: list[tuple[float, float]] = []
        self._stop_event = Event()

    def stop(self) -> None:
        """Stop sampling."""

        self._stop_event.set()

    def run(self) -> None:
        """Run the sampler."""

        while not self._stop_event.is_set():
            self.samples_mb.append((time.time(), self.process.memory_info().rss / MB))
            self._stop_event.wait(self.interval)
        self.samples_mb.append((time.time(), self.process.memory_info().rss / MB))


def _get_active_dask_client() -> Any | None:
    """Return the active distributed client, if distributed is installed and one is active."""

    try:
        from distributed import get_client
    except ImportError:
        return None

    try:
        return get_client()
    except ValueError:
        return None


def _memory_value_to_mb(value: Any) -> float:
    """Normalize Dask memory sampler values to decimal megabytes."""

    if value is None:
        return 0.0
    if hasattr(value, "memory"):
        value = value.memory
    elif isinstance(value, dict):
        value = sum(_memory_value_to_mb(v) * MB for v in value.values())
    return float(value) / MB


def _memory_sampler_trace_mb(sampler: Any, label: str) -> list[tuple[float, float]]:
    """Extract a normalized memory trace from a Dask MemorySampler."""

    df = sampler.to_pandas()
    if df.empty or label not in df:
        return []

    series = df[label].dropna()
    return [(timestamp.timestamp(), _memory_value_to_mb(value)) for timestamp, value in series.items()]


def profile_call(
    func: Callable[..., Any],
    *args: Any,
    interval: float = 0.05,
    dask: bool | None = None,
    client: Any | None = None,
    **kwargs: Any,
) -> tuple[Any, ProfileMetrics]:
    """
    Profile one function call with the appropriate GeoUtils profiling backends.

    The current/client Python process RSS is always sampled with psutil. If a distributed Dask client is active, or if
    ``client`` is passed explicitly, Dask's :class:`distributed.diagnostics.MemorySampler` is also used to measure
    aggregate worker process RSS and spilled memory.

    :param func: Callable to execute.
    :param args: Positional arguments passed to ``func``.
    :param interval: Sampling interval in seconds.
    :param dask: Whether to force Dask worker-memory sampling. By default, an active distributed client is detected.
    :param client: Explicit distributed client to sample. If omitted, the active client is used when present.
    :param kwargs: Keyword arguments passed to ``func``.
    :returns: Tuple of ``(result, metrics)``.
    """

    if interval <= 0:
        raise ValueError("Argument 'interval' must be strictly positive.")

    if client is None:
        client = _get_active_dask_client()
    use_dask = client is not None if dask is None else dask

    if use_dask and client is None:
        distributed = import_optional("distributed", extra_name="benchmark")
        try:
            client = distributed.get_client()
        except ValueError as exc:
            raise ValueError("Dask profiling was requested, but no active distributed client was found.") from exc

    process_sampler = _ProcessRSSSampler(os.getpid(), interval=interval)
    process_sampler.start()
    start_time = time.time()

    try:
        if use_dask:
            from distributed.diagnostics import MemorySampler

            worker_process = MemorySampler()
            spilled = MemorySampler()
            with worker_process.sample("worker_process", client=client, measure="process", interval=interval):
                with spilled.sample("spilled", client=client, measure="spilled", interval=interval):
                    result = func(*args, **kwargs)
        else:
            worker_process = None
            spilled = None
            result = func(*args, **kwargs)
    finally:
        runtime_s = time.time() - start_time
        process_sampler.stop()
        process_sampler.join()

    client_trace = process_sampler.samples_mb
    worker_process_trace = _memory_sampler_trace_mb(worker_process, "worker_process") if worker_process else []
    spilled_trace = _memory_sampler_trace_mb(spilled, "spilled") if spilled else []
    metrics = ProfileMetrics(
        runtime_s=runtime_s,
        peak_client_rss_mb=max((value for _, value in client_trace), default=0.0),
        peak_dask_worker_process_rss_mb=max((value for _, value in worker_process_trace), default=None),
        peak_dask_spilled_mb=max((value for _, value in spilled_trace), default=None),
        client_rss_mb=client_trace,
        dask_worker_process_rss_mb=worker_process_trace,
        dask_spilled_mb=spilled_trace,
        dask_client_detected=bool(use_dask),
    )
    return result, metrics


class Profiler:
    """
    Main profiler class for Geoutils
    """

    enabled = False
    save_graphs = False
    save_raw_data = False
    columns = ["level", "uuid_function", "name", "uuid_parent", "time", "call_time", "memory"]
    _profiling_info = pd.DataFrame(columns=columns)
    selection_activated = False
    functions_selected = []
    running_processes = []

    @staticmethod
    def enable(save_graphs: bool = False, save_raw_data: bool = False) -> None:
        """
        Enables the profiler if save_graphs or save_raw_data is activated.

        :param save_graphs: Save the default graphs generated.
        :param save_raw_data: Save the raw data on calls as a .pickle file.
        """

        # To immediately raise errors if they are not installed
        import_optional("plotly")
        import_optional("psutil")

        Profiler.save_graphs = save_graphs
        Profiler.save_raw_data = save_raw_data
        Profiler.enabled = Profiler.save_graphs or Profiler.save_raw_data

        # Reset profiling information as a new Profiler is enabled
        Profiler.reset()

    @staticmethod
    def selection_functions(functions: list[str]) -> None:
        """
        List the functions to profile by their name

        :param functions: list of the functions name to profile
        """
        Profiler.selection_activated = True
        Profiler.functions_selected = functions

    @staticmethod
    def reset_selection_functions() -> None:
        """
        Cancel the possible selection of functions to profile
        """
        Profiler.selection_activated = False
        Profiler.functions_selected = []

    @staticmethod
    def add_profiling_info(info: dict[str, float | int | Any | str | list[Any] | None]) -> None:
        """
        Add profiling info to the profiling DataFrame.

        :param info: dictionary with profiling data keys
        """
        Profiler._profiling_info.loc[len(Profiler._profiling_info)] = {
            "level": info["level"],
            "uuid_function": info["uuid_function"],
            "name": info["name"],
            "uuid_parent": info["uuid_parent"],
            "time": info["time"],
            "call_time": info["call_time"],
            "memory": info["memory"],
        }

    @staticmethod
    def generate_summary(output: str = None) -> None:
        """
        Generate Profiling summary

        :param output: Output directory path, if None output is "output_profiling" in the current directory
        """

        import_optional("plotly")
        import plotly.express as px

        if output is None:
            output = "output_profiling"

        if not Profiler.enabled or len(Profiler._profiling_info) == 0:
            return

        if Profiler.save_raw_data or Profiler.save_graphs:
            os.makedirs(output, exist_ok=True)

        if Profiler.save_raw_data:
            Profiler._profiling_info.to_pickle(os.path.join(output, "raw_data.pickle"))
        Profiler._profiling_info["text_display"] = (
            Profiler._profiling_info["name"] + " (" + Profiler._profiling_info["time"].round(2).astype(str) + " s)"
        )

        if Profiler.save_graphs:
            # time profiling flame graph
            fig = px.icicle(
                Profiler._profiling_info,
                names="text_display",
                ids="uuid_function",
                parents="uuid_parent",
                values="time",
                title="Time profiling icicle graph (functions tagged only)",
                color="time",
                color_continuous_scale="thermal",
                branchvalues="total",
            )

            fig.update_traces(tiling_orientation="v")

            fig.write_html(os.path.join(output, "time_graph.html"))

            # memory profiling graph
            for _, call_row in Profiler._profiling_info[Profiler._profiling_info["memory"].notnull()].iterrows():
                path_fig = os.path.join(output, "memory_{}.html".format(call_row["name"]))
                Profiler.plot_trace_for_call(call_row["uuid_function"], "memory", path_fig)

    @staticmethod
    def get_profiling_info(function_name: str = None) -> pd.DataFrame:
        """
        Get profiling dataframe.
        If function_name is filled, it returns only matching rows (empty if no "name" matches).

        :param function_name: function name to show the profiled information
        :return dataframe information restrains function_name if filled
        """

        if Profiler._profiling_info.empty or not function_name:
            return Profiler._profiling_info

        if function_name:
            function_list = Profiler._profiling_info.loc[Profiler._profiling_info["name"] == function_name]
            return function_list

    @staticmethod
    def reset() -> None:
        """
        Reset profiling dataframe.
        """
        Profiler._profiling_info = pd.DataFrame(columns=Profiler.columns)

    @staticmethod
    def plot_trace_for_call(uuid_function: str, data_name: str, path_fig: str) -> None:
        """
        Plot memory (or any resource tracked) usage over time for a function call, with markers for its subcalls.
        :param uuid_function: UUID of the parent function call
        :param data_name: The name of the data to plot (if cpu consumption were to be added for example)
        :param path_fig: The path to save the output plot
        """

        import_optional("plotly")
        import plotly.graph_objects as go

        # Get the parent call entry
        parent_row = Profiler._profiling_info[Profiler._profiling_info["uuid_function"] == uuid_function]
        if parent_row.empty:
            return None
        parent_row = parent_row.iloc[0]

        call_start_time = parent_row["call_time"]
        times = [data[0] - call_start_time for data in parent_row[data_name]]
        values = [data[1] for data in parent_row[data_name]]

        # Collect subcalls (direct children)
        subcalls = Profiler._profiling_info[Profiler._profiling_info["uuid_parent"] == uuid_function]

        # Plot memory usage line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=values, mode="lines+markers", name=f"{data_name} usage"))

        # Base Y position for markers
        base_y = max(values)
        offset_step = (max(values) - min(values)) / 50  # how much higher each subsequent label goes
        current_offset = -offset_step * 2

        for _, row in list(subcalls.iterrows())[::-1]:
            sub_t = row["call_time"] - call_start_time
            sub_name = row["name"]

            # start function
            y_position = base_y + current_offset
            fig.add_trace(
                go.Scatter(
                    x=[sub_t],
                    y=[y_position],
                    mode="markers+text",
                    marker={
                        "color": "black",
                        "size": 8,
                    },
                    text=[sub_name],
                    textposition="middle right",  # text right next to marker at same height
                    showlegend=False,
                )
            )

            fig.add_shape(
                type="line",
                x0=sub_t,
                x1=sub_t,
                y0=min(values),
                y1=y_position,
                line={
                    "color": "black",
                    "width": 1,
                    "dash": "dot",
                },
            )

            # Increment offset for end function
            current_offset += offset_step
            y_position = base_y + current_offset

            fig.add_trace(
                go.Scatter(
                    x=[sub_t + row["time"]],
                    y=[y_position],
                    mode="markers+text",
                    marker={
                        "color": "black",
                        "size": 8,
                    },
                    text=["end " + sub_name],
                    textposition="middle right",  # text right next to marker at same height
                    showlegend=False,
                )
            )

            fig.add_shape(
                type="line",
                x0=sub_t + row["time"],
                x1=sub_t + row["time"],
                y0=min(values),
                y1=y_position,
                line={
                    "color": "black",
                    "width": 1,
                    "dash": "dot",
                },
            )

            # Increment offset for next marker
            current_offset += offset_step

        fig.update_layout(
            title="{} usage during {} call".format(data_name, parent_row["name"]),
            xaxis_title="Time (s)",
            yaxis_title="Memory (MB)",
            showlegend=True,
        )

        fig.write_html(path_fig)


def profile(name: str, interval: int | float = 0.005, memprof: bool = False):  # type: ignore
    """
    Geoutils profiling decorator

    To profile other functions and add them to the summary graphs and data, simply add the @profile decorator before
    them, providing a descriptive name. If you also want to track memory usage over time for a specific function call,
    set memprof=True in the decorator and if the function is too fast (or slow) for the default memory sampling
    interval, you can modify it with the interval parameter (in seconds).

    :param name: name of the function in the report
    :param interval: memory sampling interval (seconds)
    :param memprof: whether to profile the memory consumption

    :example:
        from geoutils import profiler

        @profiler.profile("my profiled function", memprof=True, interval=0.05)
        def my_function():

    """

    def decorator_generator(func):  # type: ignore
        """
        Inner function
        """

        @wraps(func)
        def wrapper_profile(*args, **kwargs):  # type: ignore
            """
            Profiling wrapper

            Generate profiling logs of function, run

            :return: func(*args, **kwargs)
            """
            # if profiling is disabled, remove overhead

            if not Profiler.enabled:
                return func(*args, **kwargs)

            func_name = name
            if Profiler.selection_activated and name not in Profiler.functions_selected:
                return func(*args, **kwargs)

            uuid_function = str(uuid.uuid4())
            uuid_parent = Profiler.running_processes[-1] if Profiler.running_processes else "__main__"
            level = len(Profiler.running_processes)

            if name is None:
                func_name = func.__name__.capitalize()

            Profiler.running_processes.append(uuid_function)

            if memprof:
                # Launch memory profiling thread
                child_pipe, parent_pipe = Pipe()
                thread_monitoring = MemProf(os.getpid(), child_pipe, interval=interval)
                thread_monitoring.start()
                if parent_pipe.poll(1):  # wait for thread to start
                    parent_pipe.recv()

            start_time = time.time()
            res = func(*args, **kwargs)
            total_time = time.time() - start_time

            if memprof:
                # end memprofiling monitoring
                parent_pipe.send(0)

            Profiler.running_processes.pop(-1)  # remove function from call list

            func_data = {
                "level": level,
                "uuid_function": uuid_function,
                "name": func_name,
                "uuid_parent": uuid_parent,
                "time": total_time,
                "call_time": start_time,
                "memory": thread_monitoring.mem_data if memprof else None,
            }
            Profiler.add_profiling_info(func_data)
            return res

        return wrapper_profile

    return decorator_generator


class MemProf(Thread):
    """
    MemProf

    Profiling thread
    """

    def __init__(self, pid: int, pipe: connection.Connection, interval: float) -> None:
        """
        Init function of MemProf

        :param pid: The process ID of the monitored process
        :param pipe: The pipe used to send the end monitoring signal
        :param interval: Time interval (seconds) between memory measurements
        """
        psutil = import_optional("psutil")

        super().__init__()
        self.pipe = pipe
        self.interval = interval
        self.process = psutil.Process(pid)
        self.mem_data = []

    def run(self) -> None:
        """
        Run the memory profiling thread
        """

        try:
            # tell parent profiling is ready
            self.pipe.send(0)

            while True:

                timestamp = time.time()

                # Get memory in megabytes
                current_mem = self.process.memory_info().rss / 1000000
                self.mem_data.append((timestamp, current_mem))

                if self.pipe.poll(self.interval):
                    break

        except BrokenPipeError:
            logging.debug("Broken pipe error in log wrapper.")
