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
import flatbuffers

import uproot
import uproot_skyhook.interpretation
import uproot_skyhook.lazyobject
import uproot_skyhook.layout_generated.Compression
import uproot_skyhook.layout_generated.Page
import uproot_skyhook.layout_generated.Basket
import uproot_skyhook.layout_generated.Branch
import uproot_skyhook.layout_generated.Column
import uproot_skyhook.layout_generated.File
import uproot_skyhook.layout_generated.Dataset

class Compression(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return type(self).__module__ + "." + self.name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self is other or (isinstance(other, type(self)) and self.value == other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

none = Compression("none", uproot_skyhook.layout_generated.Compression.Compression.none)
zlib = Compression("zlib", uproot_skyhook.layout_generated.Compression.Compression.zlib)
lzma = Compression("lzma", uproot_skyhook.layout_generated.Compression.Compression.lzma)
old  = Compression("old",  uproot_skyhook.layout_generated.Compression.Compression.old)
lz4  = Compression("lz4",  uproot_skyhook.layout_generated.Compression.Compression.lz4)

compressions = {
    uproot_skyhook.layout_generated.Compression.Compression.none: none,
    uproot_skyhook.layout_generated.Compression.Compression.zlib: zlib,
    uproot_skyhook.layout_generated.Compression.Compression.lzma: lzma,
    uproot_skyhook.layout_generated.Compression.Compression.old: old,
    uproot_skyhook.layout_generated.Compression.Compression.lz4: lz4,
    }

class Layout(object):
    @classmethod
    def fromflatbuffers(cls, flatbuffers):
        self = cls.__new__(cls)
        self._flatbuffers = flatbuffers
        return self

def finalize_string(x):
    if x is None:
        return None
    else:
        return x.decode("utf-8")

class Page(Layout):
    file_seek = uproot_skyhook.lazyobject.lazyproperty("file_seek", None)
    compressedbytes = uproot_skyhook.lazyobject.lazyproperty("compressedbytes", None)
    uncompressedbytes = uproot_skyhook.lazyobject.lazyproperty("uncompressedbytes", None)

    def __init__(self, file_seek, compressedbytes, uncompressedbytes):
        self.file_seek = file_seek
        self.compressedbytes = compressedbytes
        self.uncompressedbytes = uncompressedbytes

    def _toflatbuffers(self, builder):
        return uproot_skyhook.layout_generated.Page.CreatePage(builder, self.file_seek, self.compressedbytes, self.uncompressedbytes)

    @property
    def compressed(self):
        return self.compressedbytes != self.uncompressedbytes

class Basket(Layout):
    compression = uproot_skyhook.lazyobject.lazyproperty("compression", lambda x: compressions[x])
    pages = uproot_skyhook.lazyobject.lazyproperty("pages", Page.fromflatbuffers)
    data_border = uproot_skyhook.lazyobject.lazyproperty("data_border", None)

    def __init__(self, compression, pages, data_border):
        self.compression = compression
        self.pages = pages
        self.data_border = data_border

    def _toflatbuffers(self, builder):
        pages = [x._toflatbuffers(builder) for x in self.pages]

        uproot_skyhook.layout_generated.Basket.BasketStartPagesVector(builder, len(pages))
        for x in pages[::-1]:
            builder.PrependUOffsetTRelative(x)
        pages = builder.EndVector(len(pages))

        uproot_skyhook.layout_generated.Basket.BasketStart(builder)
        uproot_skyhook.layout_generated.Basket.BasketAddCompression(builder, self.compression.value)
        uproot_skyhook.layout_generated.Basket.BasketAddPages(builder, pages)
        uproot_skyhook.layout_generated.Basket.BasketAddDataBorder(builder, self.data_border)
        return uproot_skyhook.layout_generated.Basket.BasketEnd(builder)

    @property
    def compressed(self):
        return any(x.compressed for x in self.pages)

class Branch(Layout):
    local_offsets = uproot_skyhook.lazyobject.lazyproperty("local_offsets", None)
    baskets = uproot_skyhook.lazyobject.lazyproperty("baskets", Basket.fromflatbuffers)

    def __init__(self, local_offsets, baskets):
        local_offsets = numpy.array(local_offsets, dtype="<u8", copy=False)

        if len(local_offsets) == 0 or local_offsets[0] != 0:
            raise ValueError("local_offsets must start with 0")
        if not (local_offsets[1:] >= local_offsets[:-1]).all():
            raise ValueError("local_offsets must be monatonically increasing")

        self.local_offsets = local_offsets
        self.baskets = baskets

    def _toflatbuffers(self, builder):
        baskets = [x._toflatbuffers(builder) for x in self.baskets]
        uproot_skyhook.layout_generated.Branch.BranchStartBasketsVector(builder, len(baskets))
        for x in baskets[::-1]:
            builder.PrependUOffsetTRelative(x)
        baskets = builder.EndVector(len(baskets))

        uproot_skyhook.layout_generated.Branch.BranchStartLocalOffsetsVector(builder, len(self.local_offsets))
        builder.head = builder.head - self.local_offsets.nbytes
        builder.Bytes[builder.head : builder.head + self.local_offsets.nbytes] = self.local_offsets.tostring()
        local_offsets = builder.EndVector(len(self.local_offsets))

        uproot_skyhook.layout_generated.Branch.BranchStart(builder)
        uproot_skyhook.layout_generated.Branch.BranchAddLocalOffsets(builder, local_offsets)
        uproot_skyhook.layout_generated.Branch.BranchAddBaskets(builder, baskets)
        return uproot_skyhook.layout_generated.Branch.BranchEnd(builder)

    @property
    def numentries(self):
        return self.local_offsets[-1]

class Column(Layout):
    interp = uproot_skyhook.lazyobject.lazyproperty("interp", uproot_skyhook.interpretation.fromflatbuffers)
    title = uproot_skyhook.lazyobject.lazyproperty("title", finalize_string)

    def __init__(self, interp, title=None):
        self.interp = interp
        self.title = title

    def _toflatbuffers(self, builder):
        interp = uproot_skyhook.interpretation.toflatbuffers(builder, self.interp)
        if self.title is not None:
            title = builder.CreateString(self.title.encode("utf-8"))

        uproot_skyhook.layout_generated.Column.ColumnStart(builder)
        uproot_skyhook.layout_generated.Column.ColumnAddInterp(builder, interp)
        if self.title is not None:
            uproot_skyhook.layout_generated.Column.ColumnAddTitle(builder, title)
        return uproot_skyhook.layout_generated.Column.ColumnEnd(builder)

class File(Layout):
    location = uproot_skyhook.lazyobject.lazyproperty("location", finalize_string)
    uuid = uproot_skyhook.lazyobject.lazyproperty("uuid", finalize_string)
    branches = uproot_skyhook.lazyobject.lazyproperty("branches", Branch.fromflatbuffers)

    def __init__(self, location, uuid, branches):
        self.location = location
        self.uuid = uuid
        self.branches = branches

    def _toflatbuffers(self, builder):
        branches = [x._toflatbuffers(builder) for x in self.branches]
        uproot_skyhook.layout_generated.File.FileStartBranchesVector(builder, len(branches))
        for x in branches[::-1]:
            builder.PrependUOffsetTRelative(x)
        branches = builder.EndVector(len(branches))

        location = builder.CreateString(self.location.encode("utf-8"))
        uuid = builder.CreateString(self.uuid.encode("utf-8"))

        uproot_skyhook.layout_generated.File.FileStart(builder)
        uproot_skyhook.layout_generated.File.FileAddLocation(builder, location)
        uproot_skyhook.layout_generated.File.FileAddUuid(builder, uuid)
        uproot_skyhook.layout_generated.File.FileAddBranches(builder, branches)
        return uproot_skyhook.layout_generated.File.FileEnd(builder)

    @property
    def numentries(self):
        if len(self.branches) == 0:
            return 0
        else:
            return max(x.numentries for x in self.branches)

class Dataset(Layout):
    name = uproot_skyhook.lazyobject.lazyproperty("name", finalize_string)
    treepath = uproot_skyhook.lazyobject.lazyproperty("treepath", finalize_string)
    colnames = uproot_skyhook.lazyobject.lazyproperty("columns", finalize_string)
    columns = uproot_skyhook.lazyobject.lazyproperty("columns", Column.fromflatbuffers)
    files = uproot_skyhook.lazyobject.lazyproperty("files", File.fromflatbuffers)
    global_offsets = uproot_skyhook.lazyobject.lazyproperty("global_offsets", None)
    location_prefix = uproot_skyhook.lazyobject.lazyproperty("location_prefix", finalize_string)

    def __init__(self, name, treepath, colnames, columns, files, global_offsets=None, location_prefix=None):
        if len(colnames) != len(columns):
            raise ValueError("colnames and columns must have the same length")

        if global_offsets is None:
            if len(files) == 0:
                global_offsets = numpy.array([0], dtype="<u8")
            else:
                global_offsets = numpy.empty(len(files) + 1, dtype="<u8")
                global_offsets[0] = 0
                global_offsets[1:] = numpy.cumsum(x.numentries for x in files)
        else:
            global_offsets = numpy.array(global_offsets, copy=False)

        if len(global_offsets) == 0 or global_offsets[0] != 0:
            raise ValueError("global_offsets must start with 0")
        if not (global_offsets[1:] >= global_offsets[:-1]).all():
            raise ValueError("global_offsets must be monatonically increasing")

        self.name = name
        self.treepath = treepath
        self.colnames = colnames
        self.columns = columns
        self.files = files
        self.global_offsets = global_offsets
        self.location_prefix = location_prefix
