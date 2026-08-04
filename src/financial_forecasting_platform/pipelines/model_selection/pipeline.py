from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import get_runs, find_best_run, get_registered_model_from_run, assign_champion_alias


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
                func=get_runs,
                inputs=["params:experiment_name"],
                outputs="runs",
                name="get_runs_node"
        ),
        Node(
                func=find_best_run,
                inputs=["runs", "params:metric"],
                outputs="best_run_id",
                name="find_best_run_node"
        ),
        Node(
                func=get_registered_model_from_run,
                inputs=["best_run_id"],
                outputs=["model_name", "model_version"],
                name="get_registered_model_node"
        ),
        Node(
                func=assign_champion_alias,
                inputs=["model_version", "model_name"],
                outputs=None,
                name="assign_champion_node"
        ),
    ])
