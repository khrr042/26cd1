# encoding: utf-8
from .veri import VeRi
from .aicity20 import AICity20


__factory = {
    'veri': VeRi,
    'aicity20': AICity20,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)