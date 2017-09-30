import functools

from dpipe.config import register_inline
from .from_csv import FromCSVMultiple

Isles2017 = register_inline(functools.partial(
    FromCSVMultiple,
    modalities=['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV'],
    targets=['OT'],
    metadata_rpath='meta2017.csv'
), 'isles2017')
