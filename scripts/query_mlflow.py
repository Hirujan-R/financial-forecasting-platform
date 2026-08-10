# ruff: noqa: T201
#!/usr/bin/env python3
"""Retrieve MLflow experiment results from the project's MLflow server.

Examples
--------
List experiments::

    python scripts/query_mlflow.py --uri http://52.202.10.130:5000

Show runs + metrics for the default experiment::

    python scripts/query_mlflow.py --uri http://52.202.10.130:5000 --runs

Export the runs to CSV::

    python scripts/query_mlflow.py --uri http://52.202.10.130:5000 --runs --csv out.csv

Show registered models and the champion::

    python scripts/query_mlflow.py --uri http://52.202.10.130:5000 --models
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
import requests
from mlflow import MlflowClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uri",
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment name to inspect (default: the default experiment)",
    )
    parser.add_argument(
        "--runs",
        action="store_true",
        help="Print runs and their metrics for the experiment",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Print registered models and aliases",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of runs to show (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to write the runs table as CSV",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path to write the runs data as JSON",
    )
    return parser.parse_args()


def _experiments(client: MlflowClient) -> list[dict]:
    out = []
    for exp in client.search_experiments():
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        out.append(
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "latest_run_count_checked": len(runs),
            }
        )
    return out


def _runs_table(client: MlflowClient, experiment_name: str, limit: int) -> list[dict]:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise SystemExit(f"Experiment '{experiment_name}' not found.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=limit,
    )
    rows = []
    for run in runs:
        rows.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "params": dict(run.data.params),
                "metrics": dict(run.data.metrics),
                "tags": dict(run.data.tags),
            }
        )
    return rows


def _models(uri: str) -> list[dict]:
    """Fetch registered models via the REST API.

    Used instead of the SDK client because the model-registry REST contract
    differs between mlflow versions (e.g. 2.x serves ``/registered-models/search``
    while newer clients call ``/registered-models/list``), so the client can
    silently return an empty list against a mismatched server.
    """
    response = requests.get(
        f"{uri.rstrip('/')}/api/2.0/mlflow/registered-models/search",
        timeout=15,
    )
    response.raise_for_status()

    out = []
    for model in response.json().get("registered_models", []):
        versions = []
        for v in model.get("latest_versions", []):
            versions.append(
                {
                    "version": v.get("version"),
                    "stage": v.get("current_stage"),
                    "status": v.get("status"),
                    "run_id": v.get("run_id"),
                    "tags": v.get("tags", []),
                }
            )
        out.append(
            {
                "name": model.get("name"),
                "aliases": model.get("aliases"),
                "versions": versions,
            }
        )
    return out


def main() -> None:
    args = _parse_args()
    client = MlflowClient(tracking_uri=args.uri, registry_uri=args.uri)

    if args.models:
        models = _models(args.uri)
        print("=== Registered models ===")
        for m in models:
            print(f"\nModel: {m['name']}")
            print(f"  Aliases: {m['aliases']}")
            for v in m["versions"]:
                print(
                    f"  v{v['version']} [{v['status']}] stage={v['stage']} "
                    f"run={v['run_id']} tags={v['tags']}"
                )
        if args.json:
            _write_json(args.json, models)
        return

    experiments = _experiments(client)
    print("=== Experiments ===")
    for exp in experiments:
        print(f"  {exp['experiment_id']:>10}  {exp['name']}")

    if args.runs:
        experiment_name = args.experiment or "Default"
        runs = _runs_table(client, experiment_name, args.limit)
        print(f"\n=== Runs for experiment '{experiment_name}' ===")
        if not runs:
            print("  (no runs)")
        for r in runs:
            metrics = ", ".join(
                f"{k}={v:.4f}" for k, v in r["metrics"].items()
            )
            print(
                f"  {r['run_id'][:8]} {r['status']:10} "
                f"name={r['run_name'] or 'n/a'}"
            )
            if metrics:
                print(f"    metrics: {metrics}")
        if args.csv:
            pd.DataFrame(runs).to_csv(args.csv, index=False)
            print(f"\nWrote CSV to {args.csv}")
        if args.json:
            _write_json(args.json, runs)


def _write_json(path: str, data) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"Wrote JSON to {path}")


if __name__ == "__main__":
    sys.exit(main())
