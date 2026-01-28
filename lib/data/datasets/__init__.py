from .cityflow import CityFlow
from .veri776 import VeRi776
from .vehiclex import VehicleX, VehicleXTranslated

__factory = {
    'cityflow': CityFlow,
    'veri776': VeRi776,
    'vehiclex': VehicleX,
    'vehiclex_translated': VehicleXTranslated,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
