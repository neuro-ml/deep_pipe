import functools

from pdp import Pipeline, One2One, Many2One, Source, pack_args

from dpipe.config import register_inline, register

register_inline = functools.partial(register_inline, module_type='pdp')
register = functools.partial(register, module_type='pdp')


@register()
def pipeline(source, transformers):
    if type(source) is not Source:
        source = Source(source, buffer_size=1)
    return Pipeline(source, *transformers)


@register()
def transformer(f, pack=False, **kwargs):
    if pack:
        f = pack_args(f)
    return One2One(f, **kwargs)


register_inline(Many2One)
register_inline(Source)
