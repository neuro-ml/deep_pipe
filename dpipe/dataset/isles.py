import functools

from dpipe.config.register import register
from .from_metadata import FromMetadata

# 2017
Isles2017 = register('isles2017', 'dataset')(functools.partial(
    FromMetadata,
    modalities=['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV'],
    target='OT',
    metadata_rpath='data.csv'
))
