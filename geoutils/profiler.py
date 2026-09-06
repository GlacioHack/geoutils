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

"""Measure function runtime and memory for local inspection and controlled benchmarks.

``profile_call`` is the common measurement engine, while ``Profiler`` organizes decorated calls into a larger
execution summary.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import wraps
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
    peak_client_mem_mb: float
    peak_process_tree_mem_mb: float | None = None
    peak_child_process_mem_mb: float | None = None
    peak_dask_worker_process_mem_mb: float | None = None
    peak_dask_spilled_mb: float | None = None
    client_mem_mb: list[tuple[float, float]] = field(default_factory=list)
    process_tree_mem_mb: list[tuple[float, float]] = field(default_factory=list)
    child_process_mem_mb: list[tuple[float, float]] = field(default_factory=list)
    dask_worker_process_mem_mb: list[tuple[float, float]] = field(default_factory=list)
    dask_spilled_mb: list[tuple[float, float]] = field(default_factory=list)
    dask_client_detected: bool = False

    def _memory_traces(self) -> tuple[tuple[str, list[tuple[float, float]]], ...]:
        """Return every supported memory trace with its display name."""

        return (
            ("Python process", self.client_mem_mb),
            ("Complete process tree", self.process_tree_mem_mb),
            ("Largest child process", self.child_process_mem_mb),
            ("Dask worker processes", self.dask_worker_process_mem_mb),
            ("Dask spilled memory", self.dask_spilled_mb),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a plain dictionary."""

        return asdict(self)

    def plot(self) -> Any:
        """Return an interactive Plotly figure of every available memory trace."""

        # Import Plotly only when a graph is requested so profiling keeps the dependency optional
        import_optional("plotly")
        import plotly.graph_objects as go

        traces = self._memory_traces()

        # Align psutil and Dask timestamps on one elapsed-time axis
        timestamps = [timestamp for _, trace in traces for timestamp, _ in trace]
        start_time = min(timestamps, default=0.0)
        figure = go.Figure()
        for name, trace in traces:
            if not trace:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[timestamp - start_time for timestamp, _ in trace],
                    y=[memory_mb for _, memory_mb in trace],
                    mode="lines",
                    name=name,
                    hovertemplate="%{x:.2f} s<br>%{y:.1f} MB<extra>%{fullData.name}</extra>",
                )
            )

        # Runtime in the title connects the memory trace with the scalar timing result
        figure.update_layout(
            title=f"Memory used during a {self.runtime_s:.2f} s function call",
            xaxis_title="Elapsed time (s)",
            yaxis_title="Memory (MB)",
            hovermode="x unified",
        )
        return figure


class _ProcessMemorySampler(Thread):
    """Sample memory for one process and, optionally, all its children."""

    def __init__(self, pid: int, interval: float, include_children: bool = False) -> None:
        """Prepare background sampling for one process ID and interval."""

        # Keep psutil optional until process metrics are explicitly requested
        psutil = import_optional("psutil")

        super().__init__(daemon=True)
        self.interval = interval
        self.process = psutil.Process(pid)
        self.include_children = include_children
        self.samples_mb: list[tuple[float, float]] = []
        self.process_tree_samples_mb: list[tuple[float, float]] = []
        self.child_process_samples_mb: list[tuple[float, float]] = []
        self._stop_event = Event()

    def stop(self) -> None:
        """Stop sampling."""

        self._stop_event.set()

    def run(self) -> None:
        """Run the sampler."""

        # Record timestamped memory until the profiled function signals completion
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self.interval)
        # Capture one final value so very short calls still have an end sample
        self._sample()

    def _sample(self) -> None:
        """Record memory for the parent, complete process tree and largest child."""

        # A worker can finish between discovery and memory inspection
        psutil = import_optional("psutil")
        timestamp = time.time()
        try:
            parent_mem_mb = self.process.memory_info().rss / MB
        except psutil.Error:
            return
        self.samples_mb.append((timestamp, parent_mem_mb))

        if not self.include_children:
            return

        # Sum concurrent memory so multiprocessing and subprocess runs use one boundary
        child_mem_mb = []
        for child in self.process.children(recursive=True):
            try:
                child_mem_mb.append(child.memory_info().rss / MB)
            except psutil.Error:
                continue
        self.process_tree_samples_mb.append((timestamp, parent_mem_mb + sum(child_mem_mb)))
        self.child_process_samples_mb.append((timestamp, max(child_mem_mb, default=0.0)))


def _get_active_dask_client() -> Any | None:
    """Return the active distributed client, if distributed is installed and one is active."""

    # Import at runtime so local profiling does not require Distributed
    try:
        from distributed import get_client
    except ImportError:
        return None

    # ``get_client`` raises when Distributed is installed but no cluster is active
    try:
        return get_client()
    except ValueError:
        return None


def _memory_value_to_mb(value: Any) -> float:
    """Normalize Dask memory sampler values to decimal megabytes."""

    # MemorySampler versions may return scalars, records or per-worker mappings
    if value is None:
        return 0.0
    if hasattr(value, "memory"):
        value = value.memory
    elif isinstance(value, dict):
        value = sum(_memory_value_to_mb(v) * MB for v in value.values())
    return float(value) / MB


def _memory_sampler_trace_mb(sampler: Any, label: str) -> list[tuple[float, float]]:
    """Extract a normalized memory trace from a Dask MemorySampler."""

    # Convert the sampler's internal mapping to one timestamped numeric series
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
    include_children: bool = False,
    profile_memory: bool = True,
    **kwargs: Any,
) -> tuple[Any, ProfileMetrics]:
    """
    Profile one function call with the appropriate GeoUtils profiling backends.

    When memory profiling is enabled, the current/client Python process memory is sampled with psutil. If a distributed
    Dask client is active, or if ``client`` is passed explicitly, Dask's
    :class:`distributed.diagnostics.MemorySampler` also measures aggregate worker process memory and spilled memory.
    Child-process sampling can be enabled for multiprocessing and subprocess workflows.

    :param func: Callable to execute.
    :param args: Positional arguments passed to ``func``.
    :param interval: Sampling interval in seconds.
    :param dask: Whether to force Dask worker-memory sampling. By default, an active distributed client is detected.
    :param client: Explicit distributed client to sample. If omitted, the active client is used when present.
    :param include_children: Whether to sample the complete process tree and the largest child process.
    :param profile_memory: Whether to collect memory samples in addition to the runtime.
    :param kwargs: Keyword arguments passed to ``func``.
    :returns: Tuple of ``(result, metrics)``.
    """

    if interval <= 0:
        raise ValueError("Argument 'interval' must be strictly positive.")

    # Timing-only calls use the same execution boundary without importing optional memory profilers
    if not profile_memory:
        start_time = time.time()
        result = func(*args, **kwargs)
        return result, ProfileMetrics(runtime_s=time.time() - start_time, peak_client_mem_mb=0.0)

    # Prefer an explicit client, otherwise discover the active distributed context
    if client is None:
        client = _get_active_dask_client()
    use_dask = client is not None if dask is None else dask

    # A forced Dask profile must fail clearly when there is no worker cluster
    if use_dask and client is None:
        distributed = import_optional("distributed", extra_name="benchmark")
        try:
            client = distributed.get_client()
        except ValueError as exc:
            raise ValueError("Dask profiling was requested, but no active distributed client was found.") from exc

    # Client memory is sampled in a background thread around the complete call
    process_sampler = _ProcessMemorySampler(os.getpid(), interval=interval, include_children=include_children)
    process_sampler.start()
    start_time = time.time()

    try:
        if use_dask:
            # Distributed samples aggregate process and spilled bytes across workers
            from distributed.diagnostics import MemorySampler

            worker_process = MemorySampler()
            spilled = MemorySampler()
            with worker_process.sample("worker_process", client=client, measure="process", interval=interval):
                with spilled.sample("spilled", client=client, measure="spilled", interval=interval):
                    result = func(*args, **kwargs)
        else:
            # Local calls retain the same result and client metrics without worker traces
            worker_process = None
            spilled = None
            result = func(*args, **kwargs)
    finally:
        runtime_s = time.time() - start_time
        process_sampler.stop()
        process_sampler.join()

    # Normalize all traces before deriving comparable peak measurements
    client_trace = process_sampler.samples_mb
    process_tree_trace = process_sampler.process_tree_samples_mb
    child_process_trace = process_sampler.child_process_samples_mb
    worker_process_trace = _memory_sampler_trace_mb(worker_process, "worker_process") if worker_process else []
    spilled_trace = _memory_sampler_trace_mb(spilled, "spilled") if spilled else []
    metrics = ProfileMetrics(
        runtime_s=runtime_s,
        peak_client_mem_mb=max((value for _, value in client_trace), default=0.0),
        peak_process_tree_mem_mb=max((value for _, value in process_tree_trace), default=None),
        peak_child_process_mem_mb=max((value for _, value in child_process_trace), default=None),
        peak_dask_worker_process_mem_mb=max((value for _, value in worker_process_trace), default=None),
        peak_dask_spilled_mb=max((value for _, value in spilled_trace), default=None),
        client_mem_mb=client_trace,
        process_tree_mem_mb=process_tree_trace,
        child_process_mem_mb=child_process_trace,
        dask_worker_process_mem_mb=worker_process_trace,
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
    columns = ["level", "uuid_function", "name", "uuid_parent", "time", "call_time", "memory", "metrics"]
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
    def add_profiling_info(info: dict[str, Any]) -> None:
        """
        Add profiling info to the profiling DataFrame.

        :param info: dictionary with profiling data keys
        """
        memory = info["memory"]
        metrics = info.get("metrics")
        if metrics is None:
            memory_trace = memory if isinstance(memory, list) else []
            metrics = ProfileMetrics(
                runtime_s=float(info["time"]),
                peak_client_mem_mb=max((value for _, value in memory_trace), default=0.0),
                client_mem_mb=memory_trace,
            )

        Profiler._profiling_info.loc[len(Profiler._profiling_info)] = {
            "level": info["level"],
            "uuid_function": info["uuid_function"],
            "name": info["name"],
            "uuid_parent": info["uuid_parent"],
            "time": info["time"],
            "call_time": info["call_time"],
            "memory": memory,
            "metrics": metrics,
        }

    @staticmethod
    def generate_summary(output: str = None) -> None:
        """
        Generate Profiling summary

        :param output: Output directory path, if None output is "output_profiling" in the current directory
        """

        if output is None:
            output = "output_profiling"

        if not Profiler.enabled or len(Profiler._profiling_info) == 0:
            return

        if Profiler.save_raw_data or Profiler.save_graphs:
            os.makedirs(output, exist_ok=True)

        if Profiler.save_raw_data:
            Profiler._profiling_info.to_pickle(os.path.join(output, "raw_data.pickle"))

        if Profiler.save_graphs:
            Profiler.plot_time_summary(os.path.join(output, "time_graph.html"))

            # memory profiling graph
            for _, call_row in Profiler._profiling_info[Profiler._profiling_info["memory"].notnull()].iterrows():
                path_fig = os.path.join(output, "memory_{}.html".format(call_row["name"]))
                Profiler.plot_trace_for_call(call_row["uuid_function"], "memory", path_fig)

    @staticmethod
    def plot_time_summary(path_fig: str | None = None) -> Any | None:
        """
        Plot the runtime hierarchy for all collected calls.

        :param path_fig: Optional path where the Plotly figure is saved as HTML.
        :returns: Plotly figure, or None when no calls have been collected.
        """

        if Profiler._profiling_info.empty:
            return None

        import_optional("plotly")
        import plotly.express as px

        profiling_info = Profiler._profiling_info.copy()
        profiling_info["text_display"] = (
            profiling_info["name"] + " (" + profiling_info["time"].round(2).astype(str) + " s)"
        )
        fig = px.icicle(
            profiling_info,
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

        if path_fig is not None:
            fig.write_html(path_fig)
        return fig

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
    def plot_trace_for_call(uuid_function: str, data_name: str, path_fig: str | None = None) -> Any | None:
        """
        Plot memory (or any resource tracked) usage over time for a function call, with markers for its subcalls.
        :param uuid_function: UUID of the parent function call
        :param data_name: The name of the data to plot (if cpu consumption were to be added for example)
        :param path_fig: The optional path to save the output plot
        :returns: Plotly figure, or None when the function call is not present
        """

        import_optional("plotly")
        import plotly.graph_objects as go

        # Get the parent call entry
        parent_row = Profiler._profiling_info[Profiler._profiling_info["uuid_function"] == uuid_function]
        if parent_row.empty:
            return None
        parent_row = parent_row.iloc[0]

        call_start_time = parent_row["call_time"]
        metrics = parent_row.get("metrics")
        if data_name == "memory" and isinstance(metrics, ProfileMetrics):
            traces = tuple((name, trace) for name, trace in metrics._memory_traces() if trace)
        else:
            traces = ((f"{data_name} usage", parent_row[data_name]),)

        values = [value for _, trace in traces for _, value in trace]
        if not values:
            return None

        # Collect subcalls (direct children)
        subcalls = Profiler._profiling_info[Profiler._profiling_info["uuid_parent"] == uuid_function]

        # Plot all measurements collected for this resource
        fig = go.Figure()
        for trace_name, trace in traces:
            fig.add_trace(
                go.Scatter(
                    x=[timestamp - call_start_time for timestamp, _ in trace],
                    y=[value for _, value in trace],
                    mode="lines+markers",
                    name=trace_name,
                )
            )

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
            title="{} usage during {}".format(data_name.capitalize(), parent_row["name"]),
            xaxis_title="Time (s)",
            yaxis_title="Memory (MB)",
            showlegend=True,
        )

        # Saving remains available for reports while notebooks can display the returned figure directly
        if path_fig is not None:
            fig.write_html(path_fig)
        return fig


def profile(
    name: str,
    interval: int | float = 0.005,
    memprof: bool = False,
    *,
    collect: bool = True,
) -> Any:
    """
    Geoutils profiling decorator

    To profile other functions and add them to the summary graphs and data, simply add the @profile decorator before
    them, providing a descriptive name. If you also want to track memory usage over time for a specific function call,
    set memprof=True in the decorator and if the function is too fast (or slow) for the default memory sampling
    interval, you can modify it with the interval parameter (in seconds).

    :param name: name of the function in the report
    :param interval: memory sampling interval (seconds)
    :param memprof: whether to profile the memory consumption
    :param collect: whether calls to the decorated function are collected

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

            if not collect or not Profiler.enabled:
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
            call_time = time.time()
            try:
                res, metrics = profile_call(
                    lambda: func(*args, **kwargs),
                    interval=float(interval),
                    profile_memory=memprof,
                )
            finally:
                Profiler.running_processes.pop(-1)  # remove function from call list

            func_data = {
                "level": level,
                "uuid_function": uuid_function,
                "name": func_name,
                "uuid_parent": uuid_parent,
                "time": metrics.runtime_s,
                "call_time": call_time,
                "memory": metrics.client_mem_mb if memprof else None,
                "metrics": metrics,
            }
            Profiler.add_profiling_info(func_data)
            return res

        return wrapper_profile

    return decorator_generator
