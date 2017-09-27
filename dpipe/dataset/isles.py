import functools

from dpipe.config import register_inline
from .from_csv import FromCSVMultiple

Isles2017 = register_inline(functools.partial(
    FromCSVMultiple,
    modalities=['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV'],
    target='OT',
    metadata_rpath='data.csv'
))
