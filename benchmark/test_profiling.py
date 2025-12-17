"""
Test file to quantify the impact of profiling and decorators.

- Profiling enabled and all decorators enabled
- Profiling enabled and all decorators disabled (empty selection functions)
- Profiling not called and decorators present in the code
- Profiling not called and decorators commented out in the code

The basic test run NTEST_GET_STATS get_stats with NTEST runs on exploradores_aster_dem. In terms of decorators involved,
there are geoutils.raster.raster.get_stats and geoutils.raster.raster.get_stats (so 2 per get_stats).

Package time and tracemalloc are used to compare duration and memory used for each subtests.
Each test saves its results in a CSV to reduce memory comsumption from dataframe operation.

WARNING: Take care of the last one that comments and decomments :
- @profiler.profile("geoutils.raster.raster.get_stats" ... decorator
- @profiler.profile("geoutils.stats.stats._statistics" ... decorator
WARNING: The last test only work with an Geoutils editable installation
WARNING: This benchmark work from benchmark directory
"""

import time
import tracemalloc

import pandas as pd  # type: ignore

import geoutils as gu
from geoutils.profiler import Profiler

NTEST = 20
NTEST_GET_STATS = 100


def replace(file: str, before: str, after: str):
    """
    Replace a string in a file by a specific string

    :param file path
    :param before string to replace
    :param before string to replace with
    """

    with open(file) as f:
        s = f.read()
        s = s.replace(before, after)
        with open(file, "w") as f:
            f.write(s)


def my_test(dem: gu.Raster):
    """
    Run the general test and extract its duration (s) and the max of memory used

    :param dem
    :return duration (s), peok_memory (KiB)
    """

    tracemalloc.reset_peak()
    tracemalloc.start()
    start_time = time.time()
    for t in range(NTEST_GET_STATS):
        dem.get_stats()
    duration = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    return peak_memory / 1024, duration


def test_profiler_and_decorators_activated(dem: gu.Raster):
    """
    Run my_test NTEST time and compute duration (s) and memory consumption max
    with enabled Profiler and all decorated functions activated
    Save each test metrics in test_profiler_and_decorators_activated.csv
    """
    columns = ["profiling", "decorators", "memory", "time"]
    profiling_info = pd.DataFrame(columns=columns)

    for test in range(NTEST):
        Profiler.enable(save_graphs=True, save_raw_data=True)
        peak_memory, duration = my_test(dem)
        profiling_info.loc[len(profiling_info)] = {
            "profiling": "activated",
            "decorators": "presents and activated",
            "memory": peak_memory,
            "time": duration,
        }
    profiling_info.to_csv("test_profiler_and_decorators_activated.csv")


def test_profiler_and_decorators_not_selected(dem: gu.Raster):
    """
    Run my_test NTEST time and compute duration (s) and memory consumption max
    with enabled Profiler and all decorated functions not selected
    Save each test metrics in test_profiler_and_decorators_not_selected.csv
    """
    columns = ["profiling", "decorators", "memory", "time"]
    profiling_info = pd.DataFrame(columns=columns)

    for test in range(NTEST):
        Profiler.enable(save_graphs=True, save_raw_data=True)
        Profiler.selection_functions("my_test")
        peak_memory, duration = my_test(dem)
        profiling_info.loc[len(profiling_info)] = {
            "profiling": "activated",
            "decorators": "presents and desactivated",
            "memory": peak_memory,
            "time": duration,
        }
    profiling_info.to_csv("test_profiler_and_decorators_not_selected.csv")


def test_no_profiling_decorators_presents(dem: gu.Raster):
    """
    Run my_test NTEST time and compute duration (s) and memory consumption max
    with no Profiler and all decorated functions
    Save each test metrics in test_no_profiling_decorators_presents.csv
    """
    columns = ["profiling", "decorators", "memory", "time"]
    profiling_info = pd.DataFrame(columns=columns)

    for test in range(NTEST):
        peak_memory, duration = my_test(dem)
        profiling_info.loc[len(profiling_info)] = {
            "profiling": "desactivate",
            "decorators": "presents",
            "memory": peak_memory,
            "time": duration,
        }

    profiling_info.to_csv("test_no_profiling_decorators_presents.csv")


def test_no_profiling_no_decorators(dem: gu.Raster):
    """
    Run my_test NTEST time and compute duration (s) and memory consumption max
    with no Profiler and commented decorated functions
    Save each test metrics in test_no_profiling_no_decorators.csv
    """
    # remove get_stats decorator
    file_get_stats = "../geoutils/raster/raster.py"
    str_dec_get_stats = '    @profiler.profile("geoutils.raster.raster.get_stats"'
    str_no_dec_get_stats = '    #@profiler.profile("geoutils.raster.raster.get_stats"'
    replace(file_get_stats, str_dec_get_stats, str_no_dec_get_stats)

    # remove get_stats decorator
    file_stats = "../geoutils/stats/stats.py"
    str_dec_stats = '@profiler.profile("geoutils.stats.stats._statistics"'
    str_no_dec_stats = '#@profiler.profile("geoutils.stats.stats._statistics"'

    replace(file_stats, str_dec_stats, str_no_dec_stats)
    columns = ["profiling", "decorators", "memory", "time"]
    profiling_info = pd.DataFrame(columns=columns)

    for test in range(NTEST):
        peak_memory, duration = my_test(dem)
        profiling_info.loc[len(profiling_info)] = {
            "profiling": "desactivate",
            "decorators": "commented",
            "memory": peak_memory,
            "time": duration,
        }

    replace(file_get_stats, str_no_dec_get_stats, str_dec_get_stats)
    replace(file_stats, str_no_dec_stats, str_dec_stats)
    profiling_info.to_csv("test_no_profiling_no_decorators.csv")


dem = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))

test_profiler_and_decorators_activated(dem)
test_profiler_and_decorators_not_selected(dem)
test_no_profiling_decorators_presents(dem)
test_no_profiling_no_decorators(dem)

df_profiler_and_decorators_activated = pd.read_csv("test_profiler_and_decorators_activated.csv", index_col=0)
df_profiler_and_decorators_not_selected = pd.read_csv("test_profiler_and_decorators_not_selected.csv", index_col=0)
df_no_profiling_decorators_presents = pd.read_csv("test_no_profiling_decorators_presents.csv", index_col=0)
df_no_profiling_no_decorators = pd.read_csv("test_no_profiling_no_decorators.csv", index_col=0)

df = pd.concat(
    [
        df_profiler_and_decorators_activated,
        df_profiler_and_decorators_not_selected,
        df_no_profiling_decorators_presents,
        df_no_profiling_no_decorators,
    ]
)

print("MEAN")
print(df.groupby(["profiling", "decorators"]).mean())
print("----------------")
print("STD")
print(df.groupby(["profiling", "decorators"]).std())
