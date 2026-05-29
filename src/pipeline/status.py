"""Pipeline step names, process status values, and status tracking helpers."""

from dataclasses import dataclass, field
from enum import Enum


class ProcessStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"
    NOT_PRESENT = "not present"
    STOPPED = "stopped"


class StepsName:
    DATA = "data"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"
    GRAPH = "graph"
    INTEGRATION = "integration"
    ALL = [DATA, EMBEDDING, CLUSTERING, GRAPH, INTEGRATION]


@dataclass
class PipelineQueryParams:
    process_name: str
    language: str
    skip_present: bool
    omitted_pipeline_steps: list[str] = field(default_factory=list)
    return_statistics: bool = False


# Backwards-compatible name for existing imports. New code should use
# PipelineQueryParams.
pipeline_query_params = PipelineQueryParams

steps_relation_dict = {
    StepsName.DATA: 1,
    StepsName.EMBEDDING: 2,
    StepsName.CLUSTERING: 3,
    StepsName.GRAPH: 4,
    StepsName.INTEGRATION: 5,
}


class PipelineLanguage:
    language_map = {
        "en": "en",
        "english": "en",
        "englisch": "en",
        "de": "de",
        "german": "de",
        "deutsch": "de",
    }

    @staticmethod
    def language_from_string(lang):
        return PipelineLanguage.language_map.get(lang.lower(), "en")


def add_status_to_running_process(
    process_name: str,
    step_name: str,
    step_status: ProcessStatus,
    running_processes: dict,
):
    _step = {
        "name": step_name,
        "rank": steps_relation_dict[step_name],
        "status": step_status,
    }
    _remove = -1
    if not running_processes.get(process_name, False):
        running_processes[process_name] = {
            "name": process_name,
            "status": [],
        }
    else:
        for _i, _status in enumerate(running_processes[process_name]["status"]):
            if _status.get("name", False) == step_name:
                _remove = _i
                break
        if _remove >= 0:
            running_processes[process_name]["status"].pop(_remove)

    if _remove >= 0:
        running_processes[process_name]["status"].insert(_remove, _step)
    else:
        running_processes[process_name]["status"].append(_step)
    return running_processes
