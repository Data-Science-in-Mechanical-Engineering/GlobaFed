import hydra
from omegaconf import DictConfig, OmegaConf

from fedbo.utils.hydra import MainConfig


@hydra.main(
    config_path="../../fedbo/config", config_name="federated_mlp", version_base=None
)
def test_hydra(config: DictConfig):
    yaml = OmegaConf.to_yaml(config)
    print(yaml)
    main_cfg: MainConfig = hydra.utils.instantiate(config)
    optimizer = main_cfg.create_optimizer()
    optimizer.logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(main_cfg, "\n")


if __name__ == "__main__":
    test_hydra()
