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

import unittest

import numpy
import flatbuffers

import uproot
import uproot_skyhook.serialization

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def roundtrip_interp(self, interp):
        builder = flatbuffers.Builder(1024)
        builder.Finish(uproot_skyhook.serialization.interp_toflatbuffers(builder, interp))
        serialized = builder.Output()
        assert uproot_skyhook.serialization.interp_frombuffer(serialized).identifier == interp.identifier

    def test_serialization_asdtype_flat(self):
        self.roundtrip_interp(uproot.asdtype(numpy.bool_))
        self.roundtrip_interp(uproot.asdtype("i1"))
        self.roundtrip_interp(uproot.asdtype("<i2"))
        self.roundtrip_interp(uproot.asdtype(">i2"))
        self.roundtrip_interp(uproot.asdtype("<i4"))
        self.roundtrip_interp(uproot.asdtype(">i4"))
        self.roundtrip_interp(uproot.asdtype("<i8"))
        self.roundtrip_interp(uproot.asdtype(">i8"))
        self.roundtrip_interp(uproot.asdtype("u1"))
        self.roundtrip_interp(uproot.asdtype("<u2"))
        self.roundtrip_interp(uproot.asdtype(">u2"))
        self.roundtrip_interp(uproot.asdtype("<u4"))
        self.roundtrip_interp(uproot.asdtype(">u4"))
        self.roundtrip_interp(uproot.asdtype("<u8"))
        self.roundtrip_interp(uproot.asdtype(">u8"))
        self.roundtrip_interp(uproot.asdtype("<f4"))
        self.roundtrip_interp(uproot.asdtype(">f4"))
        self.roundtrip_interp(uproot.asdtype("<f8"))
        self.roundtrip_interp(uproot.asdtype(">f8"))

        self.roundtrip_interp(uproot.asdtype(numpy.bool_, "i1"))
        self.roundtrip_interp(uproot.asdtype("i1", numpy.bool_))
        self.roundtrip_interp(uproot.asdtype("<i2", "<u2"))
        self.roundtrip_interp(uproot.asdtype(">i2", ">u2"))
        self.roundtrip_interp(uproot.asdtype("<i4", "<u4"))
        self.roundtrip_interp(uproot.asdtype(">i4", ">u4"))
        self.roundtrip_interp(uproot.asdtype("<i8", "<u8"))
        self.roundtrip_interp(uproot.asdtype(">i8", ">u8"))
        self.roundtrip_interp(uproot.asdtype("u1", "i1"))
        self.roundtrip_interp(uproot.asdtype("<u2", "<i2"))
        self.roundtrip_interp(uproot.asdtype(">u2", ">i2"))
        self.roundtrip_interp(uproot.asdtype("<u4", "<i4"))
        self.roundtrip_interp(uproot.asdtype(">u4", ">i4"))
        self.roundtrip_interp(uproot.asdtype("<u8", "<i8"))
        self.roundtrip_interp(uproot.asdtype(">u8", ">i8"))
        self.roundtrip_interp(uproot.asdtype("<f4", "<i4"))
        self.roundtrip_interp(uproot.asdtype(">f4", ">i4"))
        self.roundtrip_interp(uproot.asdtype("<f8", "<u8"))
        self.roundtrip_interp(uproot.asdtype(">f8", ">u8"))

        self.roundtrip_interp(uproot.asdtype((numpy.bool_, 10)))
        self.roundtrip_interp(uproot.asdtype(("i1", 10)))
        self.roundtrip_interp(uproot.asdtype(("<i2", 10)))
        self.roundtrip_interp(uproot.asdtype((">i2", 10)))
        self.roundtrip_interp(uproot.asdtype(("<i4", 10)))
        self.roundtrip_interp(uproot.asdtype((">i4", 10)))
        self.roundtrip_interp(uproot.asdtype(("<i8", 10)))
        self.roundtrip_interp(uproot.asdtype((">i8", 10)))
        self.roundtrip_interp(uproot.asdtype(("u1", 10)))
        self.roundtrip_interp(uproot.asdtype(("<u2", 10)))
        self.roundtrip_interp(uproot.asdtype((">u2", 10)))
        self.roundtrip_interp(uproot.asdtype(("<u4", 10)))
        self.roundtrip_interp(uproot.asdtype((">u4", 10)))
        self.roundtrip_interp(uproot.asdtype(("<u8", 10)))
        self.roundtrip_interp(uproot.asdtype((">u8", 10)))
        self.roundtrip_interp(uproot.asdtype(("<f4", 10)))
        self.roundtrip_interp(uproot.asdtype((">f4", 10)))
        self.roundtrip_interp(uproot.asdtype(("<f8", 10)))
        self.roundtrip_interp(uproot.asdtype((">f8", 10)))

    def test_serialization_asdtype_record(self):
        self.roundtrip_interp(uproot.asdtype([("one", int), ("two", float), ("three", bool)]))

    def test_serialization_asdouble32(self):
        self.roundtrip_interp(uproot.asdouble32(3.14, 99.9, 10))
