"""Test profiling tools."""

from __future__ import annotations

import glob
import os
import os.path as op
from importlib.util import find_spec
from typing import Any, Callable

import pandas as pd
import pytest

import geoutils as gu
import geoutils.profiler as profiler_module
from geoutils import examples
from geoutils.profiler import ProfileMetrics, Profiler, profile, profile_call


class TestProfiling:
    """Check lightweight call metrics and the configurable GeoUtils profiling interface."""

    def test_profile_call__psutil_only(self) -> None:
        """This test checks that a local call records process memory without Dask metrics."""

        pytest.importorskip("psutil")

        result, metrics = profile_call(lambda: sum(range(10)), interval=0.001)

        assert result == 45
        assert metrics.runtime_s >= 0
        assert metrics.peak_client_mem_mb > 0
        assert metrics.peak_process_tree_mem_mb is None
        assert metrics.peak_child_process_mem_mb is None
        assert metrics.peak_dask_worker_process_mem_mb is None
        assert metrics.peak_dask_spilled_mb is None
        assert not metrics.dask_client_detected

    def test_profile_call__timing_only(self) -> None:
        """This test checks that calls can be timed without starting any memory profiler."""

        result, metrics = profile_call(lambda: sum(range(10)), profile_memory=False)

        assert result == 45
        assert metrics.runtime_s >= 0
        assert metrics.peak_client_mem_mb == 0
        assert metrics.client_mem_mb == []
        assert not metrics.dask_client_detected

    @pytest.mark.parametrize("memprof", [False, True])
    def test_profile_decorator__uses_profile_call(self, monkeypatch: pytest.MonkeyPatch, memprof: bool) -> None:
        """This test checks that the decorator stores measurements returned by the common profiling engine."""

        pytest.importorskip("plotly")
        pytest.importorskip("psutil")
        expected_metrics = ProfileMetrics(
            runtime_s=1.5,
            peak_client_mem_mb=120.0,
            client_mem_mb=[(10.0, 100.0), (11.0, 120.0)],
        )
        profile_call_arguments: list[dict[str, Any]] = []

        def fake_profile_call(func: Callable[[], Any], *args: Any, **kwargs: Any) -> tuple[Any, ProfileMetrics]:
            profile_call_arguments.append(kwargs)
            return func(*args), expected_metrics

        monkeypatch.setattr(profiler_module, "profile_call", fake_profile_call)
        Profiler.enable(save_raw_data=True)

        @profile("decorated function", memprof=memprof, interval=0.1)
        def decorated_function(value: int) -> int:
            return value + 1

        assert decorated_function(1) == 2
        call = Profiler.get_profiling_info("decorated function").iloc[0]
        assert profile_call_arguments == [{"interval": 0.1, "profile_memory": memprof}]
        assert call["time"] == expected_metrics.runtime_s
        assert call["memory"] == (expected_metrics.client_mem_mb if memprof else None)
        assert call["metrics"] is expected_metrics

        time_figure = Profiler.plot_time_summary()
        assert time_figure is not None
        assert list(time_figure.data[0].labels) == ["decorated function (1.5 s)"]

    def test_profile_metrics__plot(self) -> None:
        """This test checks that process and Dask memory share one elapsed-time plot."""

        pytest.importorskip("plotly")

        # Fixed timestamps verify that independent profiler traces share one time origin
        metrics = ProfileMetrics(
            runtime_s=1.5,
            peak_client_mem_mb=120.0,
            client_mem_mb=[(10.0, 100.0), (11.0, 120.0)],
            dask_worker_process_mem_mb=[(10.5, 40.0), (11.0, 60.0)],
        )
        figure = metrics.plot()

        assert [trace.name for trace in figure.data] == ["Python process", "Dask worker processes"]
        assert list(figure.data[0].x) == [0.0, 1.0]
        assert list(figure.data[1].x) == [0.5, 1.0]
        assert figure.layout.yaxis.title.text == "Memory (MB)"

    def test_profile_call__process_tree(self) -> None:
        """This test checks that child-process memory is included when explicitly requested."""

        pytest.importorskip("psutil")

        from geoutils.multiproc.cluster import MpCluster

        # Keep one worker alive long enough for the sampler to observe its memory
        with MpCluster(conf={"nb_workers": 1, "max_tasks_per_child": None}) as cluster:
            result, metrics = profile_call(
                lambda: cluster.compute(cluster.submit(sum, range(10))),
                interval=0.001,
                include_children=True,
            )

        # The tree contains the client and at least one worker process
        assert result == 45
        assert metrics.peak_process_tree_mem_mb is not None
        assert metrics.peak_process_tree_mem_mb >= metrics.peak_client_mem_mb
        assert metrics.peak_child_process_mem_mb is not None
        assert metrics.peak_child_process_mem_mb > 0

    def test_profile_call__active_dask_client(self, tmp_path: str) -> None:
        """This test checks that direct and decorated calls capture active Dask worker memory."""

        pytest.importorskip("dask")
        pytest.importorskip("distributed")
        pytest.importorskip("plotly")

        import dask.array as da
        from distributed import Client, LocalCluster

        with LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=":0",
            scheduler_kwargs={"dashboard": False},
            local_directory=tmp_path,
        ) as cluster:
            with Client(cluster) as client:
                arr = da.ones((64, 64), chunks=(16, 16), dtype="float32")
                result, metrics = profile_call(lambda: float(arr.mean().compute()), interval=0.001, client=client)

                Profiler.enable(save_raw_data=True)

                @profile("decorated Dask computation", memprof=True, interval=0.001)
                def decorated_dask_computation() -> float:
                    return float(arr.mean().compute())

                @profile("decorated Dask call", memprof=True, interval=0.001)
                def decorated_dask_call() -> float:
                    return decorated_dask_computation()

                decorated_result = decorated_dask_call()
                decorated_call = Profiler.get_profiling_info("decorated Dask call").iloc[0]
                decorated_metrics = decorated_call["metrics"]
                decorated_computation = Profiler.get_profiling_info("decorated Dask computation").iloc[0]
                computation_metrics = decorated_computation["metrics"]

        assert result == 1.0
        assert metrics.peak_client_mem_mb > 0
        assert metrics.peak_dask_worker_process_mem_mb is not None
        assert metrics.peak_dask_worker_process_mem_mb > 0
        assert metrics.peak_dask_spilled_mb is not None
        assert metrics.dask_client_detected
        assert decorated_result == 1.0
        assert decorated_metrics.peak_dask_worker_process_mem_mb is not None
        assert decorated_metrics.peak_dask_worker_process_mem_mb > 0
        assert decorated_metrics.dask_client_detected
        assert computation_metrics.peak_dask_worker_process_mem_mb is not None
        assert computation_metrics.peak_dask_worker_process_mem_mb > 0
        assert decorated_computation["uuid_parent"] == decorated_call["uuid_function"]

        figure = Profiler.plot_trace_for_call(decorated_computation["uuid_function"], "memory")
        assert figure is not None
        assert [trace.name for trace in figure.data] == [
            "Python process",
            "Dask worker processes",
            "Dask spilled memory",
        ]
        assert figure.layout.title.text == "Memory usage during decorated Dask computation"

    @pytest.mark.skipif(
        find_spec("psutil") is not None and find_spec("plotly") is not None,
        reason="Only runs if psutil or plotly is missing.",
    )
    def test_profiling__missing_dep(self) -> None:
        """This test checks that enabling profiling reports missing optional dependencies."""

        from geoutils.profiler import Profiler

        with pytest.raises(ImportError, match="Optional dependency 'plotly' required.*"):
            Profiler.enable()

    @pytest.mark.parametrize(
        "profiling_configuration",
        [(False, False, True), (True, False, True), (False, True, True), (True, True, True), (True, True, False)],
    )
    @pytest.mark.parametrize("profiling_function", ["get_stats", "subsample"])
    def test_profiling_configuration(
        self, profiling_configuration: tuple[bool, bool, bool], profiling_function: str, tmp_path: str
    ) -> None:
        """This test checks that every output configuration saves only its requested profiling artifacts."""
        pytest.importorskip("plotly")
        pytest.importorskip("psutil")

        s_gr = profiling_configuration[0]
        s_rd = profiling_configuration[1]
        output_given = profiling_configuration[2]

        Profiler.enable(save_graphs=s_gr, save_raw_data=s_rd)

        dem = gu.Raster(examples.get_path_test("everest_landsat_b4"))
        if profiling_function == "get_stats":
            dem.get_stats()
        if profiling_function == "subsample":
            gu.Raster.subsample(dem, 2)

        if output_given:
            Profiler.generate_summary(tmp_path)
            output_path = tmp_path
        else:
            os.chdir(tmp_path)
            Profiler.generate_summary()
            output_path = "output_profiling"

        # if profiling is activate
        if s_rd or s_gr:

            # in each case, output dir exist
            assert op.isdir(output_path)

            # if save_raw_data:
            if s_rd:
                # check pickle
                assert op.isfile(op.join(output_path, "raw_data.pickle"))

                # check data in pickle
                df = pd.read_pickle(op.join(output_path, "raw_data.pickle"))
                assert len(df) == 1

            else:
                assert not op.isfile(op.join(output_path, "raw_data.pickle"))

            # if save_graphs:
            if s_gr:
                # check if all output graphs (time_graph + mem graph/profiled function called)
                # are generated
                assert op.isfile(op.join(output_path, "time_graph.html"))
                if profiling_function == "get_stats":
                    assert op.isfile(op.join(output_path, "memory_geoutils.raster.base.get_stats.html"))
                elif profiling_function == "subsample":
                    assert op.isfile(op.join(output_path, "memory_geoutils.raster.base.subsample.html"))
            else:
                assert not len(glob.glob(op.join(output_path, "*.html")))

        else:
            # if profiling is deactivated : nothing generated in output dir
            assert not len(glob.glob(op.join(output_path, "*")))

    def test_profiling_functions_management(self) -> None:
        """This test checks that only shared numerical implementations are collected and can be reset."""

        pytest.importorskip("plotly")
        pytest.importorskip("psutil")

        Profiler.enable(save_graphs=False, save_raw_data=True)

        assert len(Profiler.get_profiling_info()) == 0
        dem = gu.Raster(examples.get_path_test("everest_landsat_b4"))

        assert len(Profiler.get_profiling_info()) == 0
        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 1
        assert len(Profiler.get_profiling_info(function_name="geoutils.raster.base.get_stats")) == 1
        assert len(Profiler.get_profiling_info(function_name="geoutils.stats.stats._statistics")) == 0
        assert len(Profiler.get_profiling_info(function_name="no_name")) == 0

        Profiler.reset()
        assert len(Profiler.get_profiling_info()) == 0

    def test_selections_functions(self) -> None:
        """This test checks that selection limits collection to named numerical implementations."""
        pytest.importorskip("plotly")
        pytest.importorskip("psutil")

        Profiler.enable(save_graphs=False, save_raw_data=True)
        Profiler.selection_functions(["geoutils.raster.base.get_stats"])
        dem = gu.Raster(examples.get_path_test("everest_landsat_b4"))

        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 1

        Profiler.selection_functions(["geoutils.raster.base.subsample"])
        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 1

        Profiler.reset_selection_functions()
        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 2
