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
    def fromflatbuffers(cls, fb):
        self = cls.__new__(cls)
        self._flatbuffers = fb
        return self

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def __eq__(self, other):
        return self is other or (isinstance(other, Page) and self.file_seek == other.file_seek and self.compressedbytes == other.compressedbytes and self.uncompressedbytes == other.uncompressedbytes)

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

    def __eq__(self, other):
        return self is other or (isinstance(other, Basket) and self.compression == other.compression and self.pages == other.pages and self.data_border == other.data_border)

    def _toflatbuffers(self, builder):
        uproot_skyhook.layout_generated.Basket.BasketStartPagesVector(builder, len(self.pages))
        for x in self.pages[::-1]:
            x._toflatbuffers(builder)
        pages = builder.EndVector(len(self.pages))

        uproot_skyhook.layout_generated.Basket.BasketStart(builder)
        uproot_skyhook.layout_generated.Basket.BasketAddCompression(builder, self.compression.value)
        uproot_skyhook.layout_generated.Basket.BasketAddPages(builder, pages)
        uproot_skyhook.layout_generated.Basket.BasketAddDataBorder(builder, self.data_border)
        return uproot_skyhook.layout_generated.Basket.BasketEnd(builder)

    @property
    def compressed(self):
        return any(x.compressed for x in self.pages)

class Branch(Layout):
    baskets = uproot_skyhook.lazyobject.lazyproperty("baskets", Basket.fromflatbuffers)

    @property
    def local_offsets(self):
        out = getattr(self, "_local_offsets", None)
        if out is not None:
            return out
        self._local_offsets = self._flatbuffers.LocalOffsetsAsNumpy()
        return self._local_offsets

    @local_offsets.setter
    def local_offsets(self, value):
        self._local_offsets = value

    @classmethod
    def empty(cls):
        return cls([0], [])

    def __init__(self, local_offsets, baskets):
        local_offsets = numpy.array(local_offsets, dtype="<u8", copy=False)

        if len(local_offsets) == 0 or local_offsets[0] != 0:
            raise ValueError("local_offsets must start with 0")
        if not (local_offsets[1:] >= local_offsets[:-1]).all():
            raise ValueError("local_offsets must be monatonically increasing")
        if len(local_offsets) != len(baskets) + 1:
            raise ValueError("len(local_offsets) must be len(baskets) + 1")

        self.local_offsets = local_offsets
        self.baskets = baskets

    def __eq__(self, other):
        return self is other or (isinstance(other, Branch) and numpy.array_equal(self.local_offsets, other.local_offsets) and self.baskets == other.baskets)

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

    def __eq__(self, other):
        return self is other or (isinstance(other, Column) and self.interp.identifier == other.interp.identifier and self.title == other.title)

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
    uuid = uproot_skyhook.lazyobject.lazyproperty("uuid", None)
    branches = uproot_skyhook.lazyobject.lazyproperty("branches", Branch.fromflatbuffers)

    def __init__(self, location, uuid, branches):
        self.location = location
        self.uuid = uuid
        self.branches = branches

    def __eq__(self, other):
        return self is other or (isinstance(other, File) and self.location == other.location and self.uuid == other.uuid and self.branches == other.branches)

    def _toflatbuffers(self, builder):
        branches = [x._toflatbuffers(builder) for x in self.branches]
        uproot_skyhook.layout_generated.File.FileStartBranchesVector(builder, len(branches))
        for x in branches[::-1]:
            builder.PrependUOffsetTRelative(x)
        branches = builder.EndVector(len(branches))

        location = builder.CreateString(self.location.encode("utf-8"))
        uuid = builder.CreateString(self.uuid)

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
    colnames = uproot_skyhook.lazyobject.lazyproperty("colnames", finalize_string)
    columns = uproot_skyhook.lazyobject.lazyproperty("columns", Column.fromflatbuffers)
    files = uproot_skyhook.lazyobject.lazyproperty("files", File.fromflatbuffers)
    location_prefix = uproot_skyhook.lazyobject.lazyproperty("location_prefix", finalize_string)

    @property
    def global_offsets(self):
        out = getattr(self, "_global_offsets", None)
        if out is not None:
            return out
        self._global_offsets = self._flatbuffers.GlobalOffsetsAsNumpy()
        return self._global_offsets

    @global_offsets.setter
    def global_offsets(self, value):
        self._global_offsets = value

    def __init__(self, name, treepath, colnames, columns, files, global_offsets, location_prefix=None):
        if len(colnames) != len(columns):
            raise ValueError("colnames and columns must have the same length")

        if len(colnames) != len(set(colnames)):
            raise ValueError("colnames must be unique")

        global_offsets = numpy.array(global_offsets, dtype="<u8", copy=False)

        if len(global_offsets) == 0 or global_offsets[0] != 0:
            raise ValueError("global_offsets must start with 0")
        if not (global_offsets[1:] >= global_offsets[:-1]).all():
            raise ValueError("global_offsets must be monatonically increasing")
        if len(global_offsets) != len(files) + 1:
            raise ValueError("len(global_offsets) must be len(files) + 1")

        self.name = name
        self.treepath = treepath
        self.colnames = colnames
        self.columns = columns
        self.files = files
        self.global_offsets = global_offsets
        self.location_prefix = location_prefix

    @property
    def numentries(self):
        return self.global_offsets[-1]

    def __eq__(self, other):
        return self is other or (isinstance(other, Dataset) and self.name == other.name and self.treepath == other.treepath and self.colnames == other.colnames and self.columns == other.columns and self.files == other.files and numpy.array_equal(self.global_offsets, other.global_offsets) and self.location_prefix == other.location_prefix)

    def __add__(self, other):
        if not isinstance(other, Dataset):
            raise ValueError("cannot add {0} and {1}".format(type(self), type(other)))
        if self.name != other.name:
            raise ValueError("dataset names differ: {0} and {1}".format(repr(self.name), repr(other.name)))
        if self.treepath != other.treepath:
            raise ValueError("dataset treepaths differ: {0} and {1}".format(repr(self.treepath), repr(other.treepath)))
        if self.location_prefix != other.location_prefix:
            raise ValueError("dataset location_prefixes differ: {0} and {1}".format(repr(self.location_prefix), repr(other.location_prefix)))

        otherlookup = {n: i for i, n in enumerate(other.colnames)}

        colnames = list(self.colnames)
        columns = list(self.columns)
        selfmap = list(range(len(colnames)))
        othermap = [otherlookup.get(n, None) for n in colnames]
        seen = set(colnames)

        for n, x in zip(other.colnames, other.columns):
            if n not in seen:
                selfmap.append(None)
                othermap.append(len(colnames))
                colnames.append(n)
                columns.append(x)
                
        assert len(selfmap) == len(colnames)
        assert len(othermap) == len(colnames)

        files = []

        for file in self.files:
            branches = []
            for i in selfmap:
                if i is None:
                    branches.append(Branch.empty())
                else:
                    branches.append(file.branches[i])
            files.append(File(file.location, file.uuid, branches))

        for file in other.files:
            branches = []
            for i in othermap:
                if i is None:
                    branches.append(Branch.empty())
                else:
                    branches.append(file.branches[i])
            files.append(File(file.location, file.uuid, branches))

        global_offsets = numpy.array(len(self.global_offsets) + len(other.global_offsets), dtype="<u8")
        global_offsets[:len(self.global_offsets)] = self.global_offsets
        global_offsets[len(self.global_offsets):] = other.global_offsets + self.global_offsets[-1]

        return Dataset(self.name, self.treepath, colnames, columns, files, global_offsets, location_prefix=self.location_prefix)

    def _toflatbuffers(self, builder):
        files = [x._toflatbuffers(builder) for x in self.files]
        uproot_skyhook.layout_generated.Dataset.DatasetStartFilesVector(builder, len(files))
        for x in files[::-1]:
            builder.PrependUOffsetTRelative(x)
        files = builder.EndVector(len(files))

        columns = [x._toflatbuffers(builder) for x in self.columns]
        uproot_skyhook.layout_generated.Dataset.DatasetStartColumnsVector(builder, len(columns))
        for x in columns[::-1]:
            builder.PrependUOffsetTRelative(x)
        columns = builder.EndVector(len(columns))

        colnames = [builder.CreateString(x.encode("utf-8")) for x in self.colnames]
        uproot_skyhook.layout_generated.Dataset.DatasetStartColnamesVector(builder, len(colnames))
        for x in colnames[::-1]:
            builder.PrependUOffsetTRelative(x)
        colnames = builder.EndVector(len(colnames))

        uproot_skyhook.layout_generated.Dataset.DatasetStartGlobalOffsetsVector(builder, len(self.global_offsets))
        builder.head = builder.head - self.global_offsets.nbytes
        builder.Bytes[builder.head : builder.head + self.global_offsets.nbytes] = self.global_offsets.tostring()
        global_offsets = builder.EndVector(len(self.global_offsets))

        name = builder.CreateString(self.name.encode("utf-8"))
        treepath = builder.CreateString(self.treepath.encode("utf-8"))
        if self.location_prefix is not None:
            location_prefix = builder.CreateString(self.location_prefix.encode("utf-8"))

        uproot_skyhook.layout_generated.Dataset.DatasetStart(builder)
        uproot_skyhook.layout_generated.Dataset.DatasetAddName(builder, name)
        uproot_skyhook.layout_generated.Dataset.DatasetAddTreepath(builder, treepath)
        uproot_skyhook.layout_generated.Dataset.DatasetAddColnames(builder, colnames)
        uproot_skyhook.layout_generated.Dataset.DatasetAddColumns(builder, columns)
        uproot_skyhook.layout_generated.Dataset.DatasetAddFiles(builder, files)
        uproot_skyhook.layout_generated.Dataset.DatasetAddGlobalOffsets(builder, global_offsets)
        if self.location_prefix is not None:
            uproot_skyhook.layout_generated.Dataset.DatasetAddLocationPrefix(builder, location_prefix)
        return uproot_skyhook.layout_generated.Dataset.DatasetEnd(builder)

    def tobuffer(self):
        builder = flatbuffers.Builder(1024)
        builder.Finish(toflatbuffers(builder, self))
        return builder.Output()

    def tonumpy(self):
        return numpy.frombuffer(self.tobuffer(), dtype=numpy.uint8)

    def tofile(self, filename):
        with open(filename, "wb") as file:
            file.write(b"roly")
            file.write(self.tobuffer())

def frombuffer(buffer, offset=0):
    return fromflatbuffers(uproot_skyhook.layout_generated.Dataset.Dataset.GetRootAsDataset(buffer, offset))

def fromnumpy(array):
    return frombuffer(array)

def fromfile(filename):
    file = numpy.memmap(filename, dtype=numpy.uint8, mode="r")
    if file[:4].tostring() != b"roly":
        raise OSError("file does not begin with magic 'roly'")
    return fromnumpy(file[4:])
    
def fromflatbuffers(fb):
    return Dataset.fromflatbuffers(fb)

def tobuffer(dataset):
    return dataset.tobuffer()

def toflatbuffers(builder, dataset):
    return dataset._toflatbuffers(builder)
