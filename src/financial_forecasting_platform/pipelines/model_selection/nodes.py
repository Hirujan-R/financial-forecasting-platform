from mlflow import MlflowClient


def get_runs(experiment_name: str = "financial_forecasting_platform"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.parentRunId IS NOT NULL"
    )

    return runs

def find_best_run(runs, metric: str = 'roc_auc'):
    best_run = max(runs, key= lambda r: r.data.metrics[metric])
    return best_run.info.run_id

def get_registered_model_from_run(run_id: str):

    client = MlflowClient()
    versions = client.search_model_versions(
        f"run_id='{run_id}'"
    )

    if not versions:
        raise Exception(
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
