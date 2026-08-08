from mlflow import MlflowClient


def get_runs(experiment_name: str = "financial_forecasting_platform"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' does not exist. "
            "Run the model_training pipeline before model_selection."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId IS NOT NULL",
        order_by=["attributes.start_time DESC"],
    )

    return runs

def find_best_run(runs, metric: str = 'roc_auc'):
    if not runs:
        raise ValueError(
            "No candidate runs found for model selection. "
            "Run the model_training pipeline before model_selection."
        )

    eligible_runs = [
        run for run in runs
        if metric in run.data.metrics
        and run.data.metrics[metric] is not None
    ]

    if not eligible_runs:
        raise ValueError(
            f"None of the candidate runs have the metric '{metric}' logged. "
            "Ensure the metric is logged by the model_training pipeline."
        )

    best_run = max(
        eligible_runs,
        key=lambda r: (
            r.data.metrics[metric],
            r.info.start_time or 0,
        ),
    )

    return best_run.info.run_id

def get_registered_model_from_run(run_id: str):

    client = MlflowClient()
    versions = client.search_model_versions(
        f"run_id='{run_id}'"
    )

    if not versions:
        raise ValueError(
            f"No registered model found for run {run_id}"
        )

    return versions[0].name, versions[0].version

def assign_champion_alias(
    model_version: str,
    model_name: str
):
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version,
    )
