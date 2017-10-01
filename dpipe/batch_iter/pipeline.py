import functools

from dpipe.config import register_inline, register
from dpipe.externals.pdp.pdp import Pipeline, LambdaTransformer, Chunker, Source, \
    pack_args

register_inline = functools.partial(register_inline, module_type='pdp')
register = functools.partial(register, module_type='pdp')


@register()
def pipeline(source, transformers):
    if type(source) is not Source:
        source = Source(source)
    return Pipeline(source, *transformers)


@register()
def transformer(f, pack=False, **kwargs):
    if pack:
        f = pack_args(f)
    return LambdaTransformer(f, **kwargs)


register_inline(Chunker)
register_inline(Source)
