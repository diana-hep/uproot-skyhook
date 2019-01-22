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

class Page(Layout):
    file_seek = uproot_skyhook.lazyobject.lazyproperty("file_seek", lambda x: x)
    compressedbytes = uproot_skyhook.lazyobject.lazyproperty("compressedbytes", lambda x: x)
    uncompressedbytes = uproot_skyhook.lazyobject.lazyproperty("uncompressedbytes", lambda x: x)

    def __init__(self, file_seek, compressedbytes, uncompressedbytes):
        self.file_seek = file_seek
        self.compressedbytes = compressedbytes
        self.uncompressedbytes = uncompressedbytes

class Basket(Layout):
    compression = uproot_skyhook.lazyobject.lazyproperty("compression", lambda x: compressions[x])
    pages = uproot_skyhook.lazyobject.lazyproperty("pages", Page.fromflatbuffers)
    data_border = uproot_skyhook.lazyobject.lazyproperty("data_border", lambda x: x)

    def __init__(self, compression, pages, data_border):
        self.compression = compression
        self.pages = pages
        self.data_border = data_border

class Branch(Layout):
    local_offsets = uproot_skyhook.lazyobject.lazyproperty("local_offsets", lambda x: x)
    baskets = uproot_skyhook.lazyobject.lazyproperty("baskets", Basket.fromflatbuffers)

    def __init__(self, local_offsets, baskets):
        self.local_offsets = local_offsets
        self.baskets = baskets

class Column(Layout):
    name = uproot_skyhook.lazyobject.lazyproperty("name", lambda x: x.decode("utf-8"))
    interp = uproot_skyhook.lazyobject.lazyproperty("interp", uproot_skyhook.interpretation.interp_fromflatbuffers)
    title = uproot_skyhook.lazyobject.lazyproperty("title", lambda x: x.decode("utf-8"))
    aliases = uproot_skyhook.lazyobject.lazyproperty("aliases", lambda x: x.decode("utf-8"))

    def __init__(self, name, interp, title, aliases):
        self.name = name
        self.interp = interp
        self.title = title
        self.aliases = aliases

class File(Layout):
    location = uproot_skyhook.lazyobject.lazyproperty("location", lambda x: x.decode("utf-8"))
    uuid = uproot_skyhook.lazyobject.lazyproperty("uuid", lambda x: x.decode("utf-8"))
    branches = uproot_skyhook.lazyobject.lazyproperty("branches", Branch.fromflatbuffers)

    def __init__(self, location, uuid, branches):
        self.location = location
        self.uuid = uuid4
        self.branches = branches
