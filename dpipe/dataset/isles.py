import functools

from .dataframe import FromMetadata

# 2017
Isles2017 = functools.partial(
    FromMetadata,
    modalities=['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV'],
    target='OT'
)
