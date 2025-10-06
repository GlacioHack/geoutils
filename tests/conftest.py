# """Configuration file for Pytest."""

# import pytest
# from dask.distributed import Client, LocalCluster
#
# # Configure a session-scoped fixture for a Dask cluster
# @pytest.fixture(scope="session")
# def dask_client():
#     """
#     Sets up a Dask LocalCluster and Client once for the entire test session.
#     """
#     # Initialize the cluster and client
#     cluster = LocalCluster(n_workers=2, processes=True, threads_per_worker=1)
#     client = Client(cluster)
#
#     # Yield the client to the tests
#     yield client
#
#     # Teardown: close the client and cluster after all tests are run
#     client.close()
#     cluster.close()

# import pytest
# import os
# from distributed import Client, LocalCluster
# from filelock import FileLock
#
# # Path to a lock file to prevent multiple xdist workers from creating the cluster
# LOCK_PATH = "dask_cluster.lock"
# # Path to a file that will store the scheduler address
# CLUSTER_INFO_PATH = "dask_cluster_info"
#
# @pytest.fixture(scope="session")
# def dask_client():
#     """
#     A pytest fixture that creates and shares a Dask cluster for the entire test session.
#     It uses a file lock to ensure only one worker creates the cluster and writes its info.
#     """
#     if os.environ.get("PYTEST_XDIST_WORKER"):
#         # This is a worker process; wait for the cluster to be ready
#         with FileLock(LOCK_PATH):
#             with open(CLUSTER_INFO_PATH, "r") as f:
#                 scheduler_address = f.read()
#             client = Client(scheduler_address)
#         yield client
#         client.close()
#     else:
#         # This is the main process; create the cluster
#         with FileLock(LOCK_PATH):
#             cluster = LocalCluster(n_workers=4, processes=True, threads_per_worker=1)
#             client = Client(cluster)
#             with open(CLUSTER_INFO_PATH, "w") as f:
#                 f.write(client.scheduler.address)
#
#         yield client
#         client.close()
#         cluster.close()
#         # Clean up the files
#         os.remove(LOCK_PATH)
#         os.remove(CLUSTER_INFO_PATH)