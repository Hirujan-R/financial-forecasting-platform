"""
This module contains example tests for a Kedro project.
Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""
import pytest
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality

from unittest.mock import patch

class TestKedroRun:
    @patch("mlflow.evaluate")
    @patch("mlflow.data.from_pandas")
    @patch("mlflow.sklearn.log_model")
    @patch("mlflow.log_params")
    @patch("mlflow.set_tags")
    @patch("mlflow.start_run")
    def test_kedro_run_no_pipeline(
        self,
        mock_start_run,
        mock_set_tags,
        mock_log_params,
        mock_log_model,
        mock_from_pandas,
        mock_evaluate,
    ):
        bootstrap_project(Path.cwd())
        with KedroSession.create(project_path=Path.cwd()) as session:
            session.run()
