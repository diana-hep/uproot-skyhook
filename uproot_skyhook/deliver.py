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

import os
import zlib
try:
    import lzma
except ImportError:
    import backports.lzma as lzma
try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

import numpy
import lz4.block

import uproot_skyhook.layout

decompress = {
    uproot_skyhook.layout.none: lambda x, uncompressed_size: x,
    uproot_skyhook.layout.zlib: lambda x, uncompressed_size: zlib.decompress(x),
    uproot_skyhook.layout.lzma: lambda x, uncompressed_size: lzma.decompress(x),
    uproot_skyhook.layout.lz4: lz4.block.decompress
    }

def _normalize_entrystartstop(dataset, entrystart, entrystop):
    if entrystart is None:
        entrystart = 0
    if entrystart < 0:
        entrystart += dataset.numentries
    if entrystop is None:
        entrystop = dataset.numentries
    if entrystop < 0:
        entrystop += dataset.numentries
    if not 0 <= entrystart < dataset.numentries:
        raise ValueError("entrystart out of bounds")
    if not 0 <= entrystop <= dataset.numentries:
        raise ValueError("entrystop out of bounds")
    if entrystop < entrystart:
        raise ValueError("entrystop must be greater than or equal to entrystart")
    return entrystart, entrystop

def _numitems_numentries(dataset, colindex, interpretation, entrystart, entrystop):
    filestart, filestop = numpy.searchsorted(dataset.global_offsets, (entrystart, entrystop), side="left")
    if dataset.global_offsets[filestart] > entrystart:
        filestart -= 1

    out = []
    for filei in range(filestart, filestop):
        file = dataset.files[filei]
        branch = file.branches[colindex]

        globalbot, globaltop = dataset.global_offsets[filei], dataset.global_offsets[filei + 1]
        localstart = min(globaltop, max(0, int(entrystart - globalbot)))
        localstop = min(globaltop, max(0, int(entrystop - globalbot)))

        basketstart, basketstop = numpy.searchsorted(branch.local_offsets, (localstart, localstop), side="left")
        if branch.local_offsets[basketstart] > localstart:
            basketstart -= 1

        for basketi in range(basketstart, basketstop):
            if branch.basket_data_borders is None:
                numbytes = sum(branch.uncompressedbytes[pagei] for pagei in range(branch.basket_page_offsets[basketi], branch.basket_page_offsets[basketi + 1]))
            else:
                numbytes = branch.basket_data_borders[basketi]

            numentries = branch.local_offsets[basketi + 1] - branch.local_offsets[basketi]

            out.append((interpretation.numitems(numbytes, numentries), numentries))

    return out

def baskets(dataset, colname, entrystart=None, entrystop=None):
    if colname not in dataset.colnames:
        raise ValueError("colname not recognized")
    colindex = dataset.colnames.index(colname)

    entrystart, entrystop = _normalize_entrystartstop(dataset, entrystart, entrystop)

    return _baskets(dataset, colindex, entrystart, entrystop)

def _baskets(dataset, colindex, entrystart, entrystop):
    filestart, filestop = numpy.searchsorted(dataset.global_offsets, (entrystart, entrystop), side="left")
    if dataset.global_offsets[filestart] > entrystart:
        filestart -= 1

    for filei in range(filestart, filestop):
        file = dataset.files[filei]
        branch = file.branches[colindex]

        location = file.location if dataset.location_prefix is None else dataset.location_prefix + file.location
        with FileArray.open(location) as filearray:
            globalbot, globaltop = int(dataset.global_offsets[filei]), int(dataset.global_offsets[filei + 1])
            localstart = min(globaltop, max(0, int(entrystart) - globalbot))
            localstop = min(globaltop, max(0, int(entrystop) - globalbot))

            basketstart, basketstop = numpy.searchsorted(branch.local_offsets, (localstart, localstop), side="left")
            if branch.local_offsets[basketstart] > localstart:
                basketstart -= 1

            for basketi in range(basketstart, basketstop):
                basketdata = []
                basket_uncompressedbytes = 0
                for pagei in range(branch.basket_page_offsets[basketi], branch.basket_page_offsets[basketi + 1]):
                    page_seek = branch.page_seeks[pagei]
                    compressedbytes = branch.compressedbytes[pagei]
                    uncompressedbytes = branch.uncompressedbytes[pagei]
                    compresseddata = filearray[page_seek : page_seek + compressedbytes]
                    basketdata.append(decompress[branch.compression](compresseddata, uncompressedbytes))
                    basket_uncompressedbytes += uncompressedbytes

                if len(basketdata) == 1:
                    basketdata = basketdata[0]
                else:
                    basketdata = numpy.concatenate(basketdata)

                if branch.basket_data_borders is None:
                    data, byteoffsets = basketdata, None
                else:
                    keylen = branch.basket_keylens[basketi]
                    border = branch.basket_data_borders[basketi]
                    objlen = basket_uncompressedbytes
                    last = border + keylen

                    data = basketdata[:border]
                    byteoffsets = numpy.empty((objlen - border - 4) // 4, dtype=numpy.int32)      # native endian
                    byteoffsets[:-1] = basketdata[border + 4 : -4].view(">i4")   # read as big-endian and convert
                    byteoffsets[-1] = last
                    numpy.subtract(byteoffsets, keylen, byteoffsets)

                globalstart, globalstop = int(branch.local_offsets[basketi]) + globalbot, int(branch.local_offsets[basketi + 1]) + globalbot
                localbot, localtop = int(branch.local_offsets[basketi]), int(branch.local_offsets[basketi + 1])
                basketstart = min(localtop - localbot, max(0, localstart - localbot))
                basketstop = min(localtop - localbot, max(0, localstop - localbot))
                yield globalstart, globalstop, localstart, localstop, basketstart, basketstop, data, byteoffsets

class TBranch(object):
    _fLeaves = ()

def array(dataset, colname, entrystart=None, entrystop=None):
    if colname not in dataset.colnames:
        raise ValueError("colname not recognized")
    colindex = dataset.colnames.index(colname)
    column = dataset.columns[colindex]
    interpretation = column.interp

    entrystart, entrystop = _normalize_entrystartstop(dataset, entrystart, entrystop)

    numitems_numentries = _numitems_numentries(dataset, colindex, interpretation, entrystart, entrystop)
    basket_itemoffset = numpy.empty(len(numitems_numentries) + 1, dtype=int)
    basket_entryoffset = numpy.empty(len(numitems_numentries) + 1, dtype=int)
    basket_itemoffset[0] = 0
    basket_entryoffset[0] = 0
    basket_itemoffset[1:] = numpy.cumsum([x for x, y in numitems_numentries])
    basket_entryoffset[1:] = numpy.cumsum([y for x, y in numitems_numentries])

    destination = interpretation.destination(basket_itemoffset[-1], int(entrystop - entrystart))
    
    baskets = _baskets(dataset, colindex, entrystart, entrystop)
    for j, (globalstart, globalstop, localstart, localstop, basketstart, basketstop, data, byteoffsets) in enumerate(baskets):
        source = interpretation.fromroot(data, byteoffsets, basketstart, basketstop)

        expecteditems = basket_itemoffset[j + 1] - basket_itemoffset[j]
        source_numitems = interpretation.source_numitems(source)

        expectedentries = basket_entryoffset[j + 1] - basket_entryoffset[j]
        source_numentries = basketstop - basketstart

        if j + 1 == len(numitems_numentries):
            if expecteditems > source_numitems:
                basket_itemoffset[j + 1] -= expecteditems - source_numitems
            if expectedentries > source_numentries:
                basket_entryoffset[j + 1] -= expectedentries - source_numentries

        elif j == 0:
            if expecteditems > source_numitems:
                basket_itemoffset[j] += expecteditems - source_numitems
            if expectedentries > source_numentries:
                basket_entryoffset[j] += expectedentries - source_numentries

        interpretation.fill(source,
                            destination,
                            basket_itemoffset[j],
                            basket_itemoffset[j + 1],
                            basket_entryoffset[j],
                            basket_entryoffset[j + 1])

    clipped = interpretation.clip(destination,
                                  basket_itemoffset[0],
                                  basket_itemoffset[-1],
                                  basket_entryoffset[0],
                                  basket_entryoffset[-1])

    return interpretation.finalize(clipped, TBranch())

class FileArray(object):
    @classmethod
    def open(cls, location):
        parsed = urlparse(location)
        if parsed.scheme == "file" or len(parsed.scheme) == 0:
            return MemmapFileArray(os.path.expanduser(parsed.netloc + parsed.path))
        else:
            raise NotImplementedError(parsed.scheme)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MemmapFileArray(FileArray):
    def __init__(self, location):
        self._data = numpy.memmap(location, dtype=numpy.uint8, mode="r")

    def __getitem__(self, slice):
        return self._data[slice]

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._data
