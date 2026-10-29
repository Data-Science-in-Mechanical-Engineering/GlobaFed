from abc import ABC
from functools import partial


class HydraInit(ABC):
    @classmethod
    def hydra_init(cls, **kwargs):
        return partial(cls, **kwargs)
