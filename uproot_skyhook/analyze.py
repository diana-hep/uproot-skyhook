#!/usr/bin/env python

# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy
import uproot

import uproot_skyhook.layout

# def file(name, filepath, treepath, location_prefix=None, localsource=uproot.MemmapSource.defaults, xrootdsource=uproot.XRootDSource.defaults, httpsource=uproot.HTTPSource.defaults, **options):
#     fullfilepath = filepath if location_prefix is None else location_prefix + filepath
#     uprootfile = uproot.open(fullfilepath, localsource=localsource, xrootdsource=xrootdsource, httpsource=httpsource, **options)

#     numentries = 0
#     colnames = []
#     columns = []
#     branches = []
#     for branchname, uprootbranch in uprootfile[treepath].iteritems(recursive=True):
#         if uprootbranch.numbaskets != uprootbranch._numgoodbaskets:
#             raise NotImplementedError("branch recovery not handled by uproot-skyhook")

#         baskets = []
#         for i in range(uprootbranch.numbaskets):
#             source = uprootbranch._source.threadlocal()
#             key = uprootbranch._basketkey(source, i, True)
#             cursor = uproot.source.cursor.Cursor(key._fSeekKey + key._fKeylen)

#             compression = uproot_skyhook.layout.none
#             compressedbytes = key._fNbytes - key._fKeylen
#             uncompressedbytes = key._fObjlen

#             if compressedbytes == uncompressedbytes:
#                 pages = [uproot_skyhook.layout.Page(cursor.index, compressedbytes, uncompressedbytes)]
#             else:
#                 pages = []
#                 start = cursor.index
#                 while cursor.index - start < compressedbytes:
#                     algo, method, c1, c2, c3, u1, u2, u3 = cursor.fields(source.parent(), uproot.source.compressed.CompressedSource._header)
#                     cbytes = c1 + (c2 << 8) + (c3 << 16)
#                     uncbytes = u1 + (u2 << 8) + (u3 << 16)
#                     if algo == b"ZL":
#                         compression = uproot_skyhook.layout.zlib
#                     elif algo == b"XZ":
#                         compression = uproot_skyhook.layout.lzma
#                     elif algo == b"L4":
#                         compression = uproot_skyhook.layout.lz4
#                         cursor.skip(8)
#                         cbytes -= 8
#                     elif algo == b"CS":
#                         raise ValueError("unsupported compression algorithm: 'old' (according to ROOT comments, hasn't been used in 20+ years!)")

#                     pages.append(uproot_skyhook.layout.Page(cursor.index, cbytes, uncbytes))
#                     cursor.skip(cbytes)

#             data_border = 0 if key._fObjlen == key.border else key.border
#             baskets.append(uproot_skyhook.layout.Basket(compression, pages, data_border))

#         colnames.append(branchname.decode("utf-8"))
#         columns.append(uproot_skyhook.layout.Column(uprootbranch.interpretation, None if uprootbranch.title is None else uprootbranch.title.decode("utf-8")))
#         branches.append(uproot_skyhook.layout.Branch(uprootbranch._fBasketEntry[: uprootbranch.numbaskets + 1], baskets))
#         numentries = max(numentries, branches[-1].local_offsets[-1])
        
#     file = uproot_skyhook.layout.File(filepath, uprootfile._context.tfile["_fUUID"], branches)
#     return uproot_skyhook.layout.Dataset(name, treepath, colnames, columns, [file], [0, numentries], location_prefix=location_prefix)

def file(name, filepath, treepath, location_prefix=None, localsource=uproot.MemmapSource.defaults, xrootdsource=uproot.XRootDSource.defaults, httpsource=uproot.HTTPSource.defaults, **options):
    fullfilepath = filepath if location_prefix is None else location_prefix + filepath
    uprootfile = uproot.open(fullfilepath, localsource=localsource, xrootdsource=xrootdsource, httpsource=httpsource, **options)

    numentries = 0
    colnames = []
    columns = []
    branches = []
    for branchname, uprootbranch in uprootfile[treepath].iteritems(recursive=True):
        if uprootbranch.numbaskets != uprootbranch._numgoodbaskets:
            raise NotImplementedError("branch recovery not handled by uproot-skyhook")

        local_offsets = uprootbranch._fBasketEntry[: uprootbranch.numbaskets + 1]
        page_seeks = numpy.empty(uprootbranch.numbaskets, dtype="<u8")   # extremely rare, though possible, for numpages > numbaskets
        compression = None
        compressedbytes = numpy.empty(uprootbranch.numbaskets, dtype="<u4")
        uncompressedbytes = numpy.empty(uprootbranch.numbaskets, dtype="<u4")
        basket_page_offsets = numpy.empty(uprootbranch.numbaskets + 1, dtype="<u4")
        basket_page_offsets[0] = 0
        basket_data_borders = numpy.zeros(uprootbranch.numbaskets, dtype="<u4")

        for i in range(uprootbranch.numbaskets):
            source = uprootbranch._source.threadlocal()
            key = uprootbranch._basketkey(source, i, True)
            cursor = uproot.source.cursor.Cursor(key._fSeekKey + key._fKeylen)

            basket_compressedbytes = key._fNbytes - key._fKeylen
            basket_uncompressedbytes = key._fObjlen
            
            if basket_compressedbytes == basket_uncompressedbytes:
                if compression is not None and compression != uproot_skyhook.layout.none:
                    raise ValueError("different compression used by different branches")
                pagei = basket_page_offsets[i]
                page_seeks[pagei] = cursor.index
                compressedbytes[pagei] = basket_compressedbytes
                uncompressedbytes[pagei] = basket_uncompressedbytes
                pagei += 1
                basket_page_offsets[i + 1] = pagei

            else:
                pagei = basket_page_offsets[i]
                start = cursor.index
                while cursor.index - start < basket_compressedbytes:
                    algo, method, c1, c2, c3, u1, u2, u3 = cursor.fields(source.parent(), uproot.source.compressed.CompressedSource._header)
                    page_compressedbytes = c1 + (c2 << 8) + (c3 << 16)
                    page_uncompressedbytes = u1 + (u2 << 8) + (u3 << 16)
                    if algo == b"ZL":
                        if compression is not None and compression != uproot_skyhook.layout.zlib:
                            raise ValueError("different compression used by different branches")
                        compression = uproot_skyhook.layout.zlib
                    elif algo == b"XZ":
                        if compression is not None and compression != uproot_skyhook.layout.lzma:
                            raise ValueError("different compression used by different branches")
                        compression = uproot_skyhook.layout.lzma
                    elif algo == b"L4":
                        if compression is not None and compression != uproot_skyhook.layout.lz4:
                            raise ValueError("different compression used by different branches")
                        compression = uproot_skyhook.layout.lz4
                        cursor.skip(8)
                        page_compressedbytes -= 8
                    elif algo == b"CS":
                        raise ValueError("unsupported compression algorithm: 'old' (according to ROOT comments, hasn't been used in 20+ years!)")

                    if pagei >= len(page_seeks):
                        page_seeks = numpy.resize(page_seeks, int(len(page_seeks)*1.2))
                        compressedbytes = numpy.resize(compressedbytes, int(len(compressedbytes)*1.2))
                        uncompressedbytes = numpy.resize(uncompressedbytes, int(len(uncompressedbytes)*1.2))

                    page_seeks[pagei] = cursor.index
                    compressedbytes[pagei] = page_compressedbytes
                    uncompressedbytes[pagei] = page_uncompressedbytes
                    pagei += 1

                    cursor.skip(page_compressedbytes)

                basket_page_offsets[i + 1] = pagei

            basket_data_borders[i] = 0 if key._fObjlen == key.border else key.border

        if len(page_seeks) > basket_page_offsets[-1]:
            page_seeks = page_seeks[:basket_page_offsets[-1]].copy()
            compressedbytes = compressedbytes[:basket_page_offsets[-1]].copy()
            uncompressedbytes = uncompressedbytes[:basket_page_offsets[-1]].copy()

        colnames.append(branchname.decode("utf-8"))
        columns.append(uproot_skyhook.layout.Column(uprootbranch.interpretation, None if uprootbranch.title == b"" or uprootbranch.title is None else uprootbranch.title.decode("utf-8")))
        branches.append(uproot_skyhook.layout.Branch(local_offsets, page_seeks, compression, compressedbytes, uncompressedbytes, basket_page_offsets, basket_data_borders))
        numentries = max(numentries, branches[-1].local_offsets[-1])
        
    file = uproot_skyhook.layout.File(filepath, uprootfile._context.tfile["_fUUID"], branches)
    return uproot_skyhook.layout.Dataset(name, treepath, colnames, columns, [file], [0, numentries], location_prefix=location_prefix)
