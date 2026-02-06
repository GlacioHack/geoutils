# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
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

"""This module defines the cluster configurations."""

import multiprocessing
from multiprocessing.pool import Pool
from typing import Any, Callable, Dict, List, Optional


class ClusterGenerator:
    def __new__(cls, name: str, nb_workers: int = 2) -> "AbstractCluster":  # type: ignore
        """
        Factory method to create different types of clusters based on the name argument.
        - If 'basic' is provided, a BasicCluster is instantiated.
        - Otherwise, an MpCluster (multiprocessing) is created with the given number of workers.
        """
        if name == "basic":
            cluster: AbstractCluster = BasicCluster()
        else:
            cluster = MpCluster(conf={"nb_workers": nb_workers})
        return cluster


class AbstractCluster:
    def __init__(self) -> None:
        """
        Base class for clusters. Initializes the pool attribute.
        Meant to be subclassed and extended.
        """
        self.pool: Optional[Pool] = None

    def __enter__(self) -> "AbstractCluster":
        """Context manager entry point, returning the cluster instance."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point, ensures the cluster is properly closed."""
        self.close()

    def close(self) -> None:
        """Method to clean up resources. To be implemented by subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def launch_task(
        self, fun: Callable[..., Any], args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Method to launch a task. This should be implemented by subclasses.
        :param fun: The function to run.
        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_res(self, future: Any) -> Any:
        """
        Retrieve the result from a launched task. Meant to be subclassed.
        :param future: The future object representing the result of an asynchronous task.
        """
        return future

    def return_wrapper(self) -> None:
        """Wrapper for returned values, should be customized in subclasses if needed."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def tile_retriever(self, res: Any) -> None:
        """Method to retrieve and process tiles, specific to subclasses."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class BasicCluster(AbstractCluster):
    def launch_task(
        self, fun: Callable[..., Any], args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Launches a task synchronously in a basic cluster (no multiprocessing).
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return fun(*args, **kwargs)


class MpCluster(AbstractCluster):
    def __init__(self, conf: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes a multiprocessing cluster.
        :param conf: Configuration dictionary, which may contain the number of workers.
        """
        super().__init__()
        nb_workers = 1
        if conf is not None:
            nb_workers = conf.get("nb_workers", 1)
        # Using the 'forkserver' context for more controlled process handling
        ctx_in_main = multiprocessing.get_context("fork")
        # Create a pool of workers with max 10 tasks per child process
        self.pool = ctx_in_main.Pool(processes=nb_workers, maxtasksperchild=10)

    def close(self) -> None:
        """Closes the multiprocessing pool by terminating and joining workers."""
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()

    def launch_task(
        self, fun: Callable[..., Any], args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Launches a task asynchronously in the multiprocessing pool.
        :param fun: The function to execute in parallel.
        :param args: The positional arguments for the function.
        :param kwargs: The keyword arguments for the function.

        :return: an asynchronous result (future object).
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if self.pool is not None:
            return self.pool.apply_async(fun, args=args, kwds=kwargs)

    def get_res(self, future: Any) -> Any:
        """
        Retrieves the result of a completed asynchronous task.
        """
        return future.get(timeout=5000)
