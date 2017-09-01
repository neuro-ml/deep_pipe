from .meta import extractor, from_json
from .batch_iter_factory import BatchIterFactoryFin, BatchIterFactoryInf
from .transformers import segm_prob2msegm

name2batch_iter_factory = {
    'fin': BatchIterFactoryFin,
    'inf': BatchIterFactoryInf
}


name2meta = {
    'extractor': extractor,
    'from_json': from_json
}


name2transform = {
    'segm_prob2msegm': segm_prob2msegm
}
