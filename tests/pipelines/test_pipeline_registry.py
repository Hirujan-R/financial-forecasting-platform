from kedro.pipeline import Pipeline

from financial_forecasting_platform.pipeline_registry import register_pipelines
from financial_forecasting_platform.pipelines.data_ingestion.pipeline import (
    create_pipeline as create_data_ingestion_pipeline,
)
from financial_forecasting_platform.pipelines.data_validation.pipeline import (
    create_pipeline as create_data_validation_pipeline,
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
        assert "data_validation" in result
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

    def test_data_validation_pipeline(self):
        pipelines = register_pipelines()
        expected = create_data_validation_pipeline()
        assert [n.name for n in pipelines["data_validation"].nodes] == [
            n.name for n in expected.nodes
        ]

    def test_feature_engineering_pipeline(self):
        pipelines = register_pipelines()
        expected = create_feature_engineering_pipeline()
        assert [n.name for n in pipelines["feature_engineering"].nodes] == [
            n.name for n in expected.nodes
        ]

    def test_default_pipeline_combines_all(self):
        pipelines = register_pipelines()
        default = pipelines["__default__"]

        default_node_names = {n.name for n in default.nodes}
        expected_node_names = set()
        for name, pipe in pipelines.items():
            if name != "__default__":
                expected_node_names |= {n.name for n in pipe.nodes}

        assert default_node_names == expected_node_names

    def test_default_pipeline_node_count(self):
        pipelines = register_pipelines()
        expected_count = sum(
            len(pipe.nodes) for name, pipe in pipelines.items() if name != "__default__"
        )
        assert len(pipelines["__default__"].nodes) == expected_count

    def test_total_pipeline_count(self):
        pipelines = register_pipelines()
        assert len(pipelines) == 7
