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

class Layout(object): pass

class Page(Layout):
    def __init__(self, file_seek, compressedbytes, uncompressedbytes):
        self.file_seek = file_seek
        self.compressedbytes = compressedbytes
        self.uncompressedbytes = uncompressedbytes

