from .meta import extractor, from_json
from .batch_iter_factory import BatchIterFactoryFin, BatchIterFactoryInf

name2batch_iter_factory = {
    'fin': BatchIterFactoryFin,
    'inf': BatchIterFactoryInf
}


name2meta = {
    'extractor': extractor,
    'from_json': from_json
}