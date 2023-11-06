from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_model_input_table, preprocess_companies, preprocess_shuttles

from spacelfights_dynamic import settings


def create_pipeline(**kwargs) -> Pipeline:
    data_processing = pipeline(
        [
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="preprocess_companies_node",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="preprocess_shuttles_node",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_shuttles", "preprocessed_companies", "reviews"],
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )

    pipes = []
    for namespace in settings.DYNAMIC_PIPELINES_MAPPING.keys():
        pipes.append(
            pipeline(
                data_processing,
                inputs={
                    "companies": "companies",
                    "shuttles": "shuttles",
                    "reviews": "reviews",
                },
                namespace=namespace,
                tags=settings.DYNAMIC_PIPELINES_MAPPING[namespace],
            )
        )
    return sum(pipes)
