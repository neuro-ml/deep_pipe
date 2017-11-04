# import functools
#
# from pdp import Pipeline, One2One, Many2One, Source, pack_args, One2Many
#
# from dpipe.config import register_inline, register
#
# register = register(module_type='pdp')
#
#
# @register
# def pipeline(source, transformers, batch_size=None):
#     if type(source) is not Source:
#         source = Source(source, buffer_size=1)
#     if batch_size is not None:
#         transformers.extend([
#             Many2One(chunk_size=batch_size, buffer_size=2),
#             One2One(pack_args, buffer_size=2),
#         ])
#     return Pipeline(source, *transformers)
#
#
# @register
# def source(iterable, buffer_size=1):
#     return Source(iterable, buffer_size=buffer_size)
#
#
# @register
# def one2one(f, pack=False, n_workers=1, buffer_size=1):
#     if pack:
#         f = pack_args(f)
#     return One2One(f, n_workers=n_workers, buffer_size=buffer_size)
#
#
# @register
# def one2many(f, pack=False, n_workers=1, buffer_size=1):
#     if pack:
#         f = pack_args(f)
#     return One2Many(f, n_workers=n_workers, buffer_size=buffer_size)
#
#
# @register
# def many2one(chunk_size, buffer_size=1):
#     return Many2One(chunk_size, buffer_size=buffer_size)
