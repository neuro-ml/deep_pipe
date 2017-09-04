import functools

from .from_metadata import FromMetadata

# 2017
Isles2017 = functools.partial(
    FromMetadata,
    modalities=['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV'],
    target='OT',
    metadata_rpath='data.csv'
)
