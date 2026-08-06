from kedro.config import OmegaConfigLoader
from pathlib import Path


def load_parameters():

    conf_path = Path("conf")

    config_loader = OmegaConfigLoader(
        conf_source=str(conf_path)
    )

    params = config_loader["parameters"]

    return params
