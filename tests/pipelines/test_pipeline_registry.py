from kedro.pipeline import Pipeline

from financial_forecasting_platform.pipeline_registry import register_pipelines
from financial_forecasting_platform.pipelines.data_ingestion.pipeline import (
    create_pipeline as create_data_ingestion_pipeline,
)
from financial_forecasting_platform.pipelines.feature_engineering.pipeline import (
    create_pipeline as create_feature_engineering_pipeline,
)


class TestRegisterPipelines:
    def test_returns_dict(self):
        result = register_pipelines()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        result = register_pipelines()
        assert "data_ingestion" in result
        assert "feature_engineering" in result
        assert "__default__" in result

    def test_all_values_are_pipelines(self):
        result = register_pipelines()
        for name, pipeline in result.items():
            assert isinstance(pipeline, Pipeline), f"{name} is not a Pipeline"

    def test_data_ingestion_pipeline(self):
        pipelines = register_pipelines()
        expected = create_data_ingestion_pipeline()
        assert [n.name for n in pipelines["data_ingestion"].nodes] == [
            n.name for n in expected.nodes
        ]

    def test_feature_engineering_pipeline(self):
        pipelines = register_pipelines()
        expected = create_feature_engineering_pipeline()
        assert [n.name for n in pipelines["feature_engineering"].nodes] == [
            n.name for n in expected.nodes
        ]

    def test_default_pipeline_combines_both(self):
        pipelines = register_pipelines()
        default = pipelines["__default__"]
        data_ingestion = pipelines["data_ingestion"]
        feature_engineering = pipelines["feature_engineering"]

        default_node_names = {n.name for n in default.nodes}
        assert default_node_names == {n.name for n in data_ingestion.nodes} | {
            n.name for n in feature_engineering.nodes
        }

    def test_default_pipeline_has_two_nodes(self):
        pipelines = register_pipelines()
        assert len(pipelines["__default__"].nodes) == 2

    def test_total_pipeline_count(self):
        pipelines = register_pipelines()
        assert len(pipelines) == 3
