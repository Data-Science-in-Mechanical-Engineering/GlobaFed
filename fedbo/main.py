from __future__ import annotations
import jax
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fedbo.utils.hydra import MainConfig

jax.config.update("jax_enable_x64", True)


@hydra.main(config_path="config", config_name="federated_mlp", version_base=None)
def main(config: DictConfig):
    main_cfg: MainConfig = hydra.utils.instantiate(config)
    optimizer = main_cfg.create_optimizer()
    optimizer.logger.log_config(OmegaConf.to_container(config, resolve=True))
    optimizer.run()


if __name__ == "__main__":
    main()
