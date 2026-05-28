"""Process status and thread management helpers for pipeline execution."""

import logging
import pathlib
from collections import OrderedDict
from time import sleep
from typing import Optional, Union

import flask
from flask import Response

from main_utils import (
    HTTPResponses,
    ProcessStatus,
    StoppableThread,
    add_status_to_running_process,
    steps_relation_dict,
    StepsName,
    BaseUtil,
)


def populate_running_processes(
    app: flask.Flask, path: Union[str, pathlib.Path], running_processes: dict
):
    for process in get_all_processes(path):
        _finished = [
            _finished_step.get("name") for _finished_step in process.get("status", [])
        ]
        _process_name = process.get("name", None)
        if _process_name is None:
            app.logger.warning(
                f"Skipping process entry with no name and '{_finished}' steps."
            )
            continue

        for _step, _rank in steps_relation_dict.items():
            if _step not in _finished:
                add_status_to_running_process(
                    _process_name, _step, ProcessStatus.NOT_PRESENT, running_processes
                )
            else:
                add_status_to_running_process(
                    _process_name, _step, ProcessStatus.FINISHED, running_processes
                )
    return running_processes


def get_all_processes(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    _process_detailed = list()
    for _proc in path.glob("*"):
        if _proc.is_dir() and not _proc.stem.startswith("."):
            _proc_name = _proc.stem.lower()
            _steps_list = list()
            for _pickle in pathlib.Path(path / _proc_name).glob("*.pickle"):
                _pickle_stem = _pickle.stem.lower()
                _step = _pickle_stem.removeprefix(f"{_proc_name}_")
                if steps_relation_dict.get(_step, False):
                    _steps_list.append(
                        {
                            "rank": steps_relation_dict.get(_step),
                            "name": _step,
                            "status": ProcessStatus.FINISHED,
                        }
                    )
            _ord_dict = OrderedDict()
            _ord_dict["name"] = _proc_name
            _ord_dict["status"] = sorted(_steps_list, key=lambda x: x.get("rank", -1))
            _process_detailed.append(_ord_dict)
    return _process_detailed


def start_processes(
    app: flask.Flask,
    processes: tuple,
    process_name: str,
    process_tracker: dict[str, dict],
    thread_store: dict[str, StoppableThread],
    active_process_objs: dict[str, dict],
    last_step: str,
):
    _name_marker = {
        StepsName.DATA: "**data**, embedding, clustering, graph, integration",
        StepsName.EMBEDDING: "data, **embedding**, clustering, graph, integration",
        StepsName.CLUSTERING: "data, embedding, **clustering**, graph, integration",
        StepsName.GRAPH: "data, embedding, clustering, **graph**, integration",
        StepsName.INTEGRATION: "data, embedding, clustering, graph, **integration**",
    }
    for process_obj, _fact, _name in processes:
        process_obj: BaseUtil
        this_thread: Optional[StoppableThread] = thread_store.get(process_name, None)
        if this_thread is not None and this_thread.set_to_stop:
            add_status_to_running_process(
                process_name=process_name,
                step_name=_name,
                step_status=ProcessStatus.NOT_PRESENT,
                running_processes=process_tracker,
            )
            continue
        log_warning = f"Something went wrong with one of the previous steps: {_name_marker[_name]}."
        if any(
            [
                True
                for d in process_tracker.get(process_name, {}).get("status", [])
                if (d.get("name") == _name and d.get("status") == ProcessStatus.ABORTED)
            ]
        ):
            app.logger.warning(log_warning + f"\n So this one was aborted: '{_name}'.")
            continue
        try:
            if this_thread.set_to_stop:
                app.logger.info(
                    f"Thread for '{process_name}' was stopped before this step '{_name}'. So subsequent steps will not be started."
                )
                return
            process_obj.start_process(
                cache_name=process_name,
                process_factory=_fact,
                process_tracker=process_tracker,
                active_process_objs=active_process_objs,
                thread=this_thread,
            )
            if _name == last_step:
                app.logger.info(f"Pipeline finished with last step: '{_name}'.")
        except FileNotFoundError as e:
            app.logger.warning(log_warning + f"\nThere is a pickle file missing: {e}")


def start_thread(
    app: flask.Flask,
    process_name: str,
    pipeline_thread: StoppableThread,
    threading_store: Optional[dict[str, StoppableThread]],
):
    app.logger.info(f"Starting thread for '{process_name}'.")
    pipeline_thread.start()
    # if threading_store is not None: threading_store[process_name] = pipeline_thread
    sleep(1.5)
    return True


def stop_thread(
    app: flask.Flask,
    process_name: str,
    threading_store: dict[str, StoppableThread],
    process_tracker: dict[str, dict],
    hard_stop: bool = False,
):
    # ToDo: delete config for aborted steps
    # ToDo: maybe allow for hard stop? -> terminate the thread even if a step is not yet finished
    app.logger.info(f"Trying to stop thread for '{process_name}'.")
    if hard_stop:
        app.logger.warning(f"Hard stopping a thread is not implemented.")
        hard_stop = False
    _thread = threading_store.get(process_name, None)
    _process = process_tracker.get(process_name, None)

    if _thread is None or _process is None:
        _msg = f"No thread/process for '{process_name}' was found."
        logging.error(_msg)
        return Response(_msg, status=int(HTTPResponses.INTERNAL_SERVER_ERROR))

    _current_step = next(
        (
            step
            for step in sorted(
                _process.get("status", {}), key=lambda p: p.get("rank", 99)
            )
            if step.get("status", None) == ProcessStatus.RUNNING
        ),
        None,
    )
    if _current_step is None:
        _msg = f"Couldn't find a running step in the pipeline '{process_name}'."
        logging.error(_msg)
        return Response(_msg, status=int(HTTPResponses.INTERNAL_SERVER_ERROR))
    _thread.stop(hard_stop=hard_stop)

    try:
        for _step in sorted(
            _process.get("status", {}), key=lambda p: p.get("rank", 99)
        ):
            if _step.get("rank") <= _current_step.get("rank"):
                if _step.get("name") == _current_step.get("name"):
                    process_tracker[process_name]["status"][_step.get("rank") - 1][
                        "status"
                    ] = ProcessStatus.STOPPED
                continue
            process_tracker[process_name]["status"][_step.get("rank") - 1]["status"] = (
                ProcessStatus.ABORTED
            )
    except Exception:
        pass

    app.logger.info(
        f"Thread for '{process_name}' will be stopped after the present step ('{_current_step.get('name', None)}') is completed."
    )
    return Response("Process will be stopped.", status=int(HTTPResponses.ACCEPTED))
