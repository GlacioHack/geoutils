"""Test the xdem.profiling functions."""

from __future__ import annotations

import glob
import os
import os.path as op

import pandas as pd
import pytest

import geoutils as gu
from geoutils import examples
from geoutils.profiler import Profiler

pytest.importorskip("plotly")  # import for CI


class TestProfiling:

    # Test that there's no crash when giving profiling configuration
    @pytest.mark.parametrize(
        "profiling_configuration",
        [(False, False, True), (True, False, True), (False, True, True), (True, True, True), (True, True, False)],
    )  # type: ignore
    @pytest.mark.parametrize("profiling_function", ["load", "get_stats", "subsample", "output_given"])  # type: ignore
    def test_profiling_configuration(self, profiling_configuration, profiling_function, tmp_path) -> None:
        """
        Test the all combinaisons of profiling with three examples of profiled functions.
        """
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
                if profiling_function == "get_stats":
                    assert len(df) == 3
                elif profiling_function == "subsample":
                    assert len(df) == 2
                else:
                    assert len(df) == 1

            else:
                assert not op.isfile(op.join(output_path, "raw_data.pickle"))

            # if save_graphs:
            if s_gr:
                # check if all output graphs (time_graph + mem graph/profiled function called)
                # are generated
                assert op.isfile(op.join(output_path, "time_graph.html"))
                assert op.isfile(op.join(output_path, "memory_geoutils.raster.raster.__init__.html"))
                if profiling_function == "get_stats":
                    assert op.isfile(op.join(output_path, "memory_geoutils.stats.stats._statistics.html"))
                    assert op.isfile(op.join(output_path, "memory_geoutils.raster.raster.get_stats.html"))
                elif profiling_function == "sampling":
                    assert op.isfile(op.join(output_path, "memory_geoutils.raster.raster.subsample.html"))
            else:
                assert not len(glob.glob(op.join(output_path, "*.html")))

        else:
            # if profiling is deactivated : nothing generated in output dir
            assert not len(glob.glob(op.join(output_path, "*")))

    def test_profiling_functions_management(self) -> None:
        """
        Test the management of profiling functions information.
        """
        Profiler.enable(save_graphs=False, save_raw_data=True)

        assert len(Profiler.get_profiling_info()) == 0
        gu.Raster(examples.get_path_test("everest_landsat_b4"))

        assert len(Profiler.get_profiling_info()) == 1
        assert len(Profiler.get_profiling_info(function_name="geoutils.raster.raster.__init__")) == 1
        assert len(Profiler.get_profiling_info(function_name="geoutils.stats.stats.get_stats")) == 0
        assert len(Profiler.get_profiling_info(function_name="no_name")) == 0

        Profiler.reset()
        assert len(Profiler.get_profiling_info()) == 0

    def test_selections_functions(self) -> None:
        """
        Test the selection of functions to profile (all or by theirs names).
        """
        Profiler.enable(save_graphs=False, save_raw_data=True)
        Profiler.selection_functions(["geoutils.stats.stats._statistics"])
        dem = gu.Raster(examples.get_path_test("everest_landsat_b4"))

        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 1

        Profiler.selection_functions(["geoutils.raster.raster.__init__"])
        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 1

        Profiler.reset_selection_functions()
        dem.get_stats()
        assert len(Profiler.get_profiling_info()) == 3
