import os

import hydra
from omegaconf import OmegaConf

from lib import trainer
from lib.hydra import get_config_name


OmegaConf.register_resolver("config_name", get_config_name)


@hydra.main(config_path="conf", config_name=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    print(os.getcwd())
    root_code_directory = os.path.dirname(os.path.realpath(__file__))
    trainer.main(cfg, root_code_directory)


if __name__ == "__main__":
    main()
