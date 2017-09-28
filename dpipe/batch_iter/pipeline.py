import functools

from dpipe.externals.resource_manager.resource_manager import register_inline, register
from dpipe.externals.pdp.pdp import Pipeline, LambdaTransformer, Chunker, Source

register_inline = functools.partial(register_inline, module_type='pdp')
register = functools.partial(register, module_type='pdp')


@register()
def pipeline(source, transformers):
    if type(source) is not Source:
        source = Source(source)
    return Pipeline(source, *transformers)


register_inline(LambdaTransformer)
register_inline(Chunker)
register_inline(Source)
