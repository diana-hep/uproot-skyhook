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

try:
    from collections.abc import Iterable
    from collections.abc import Sequence
except ImportError:
    from collections import Iterable
    from collections import Sequence

class LazyList(Sequence):
    def __init__(self, get, length):
        self._get = get
        self._got = [None] * length

    def __len__(self):
        return len(self._got)

    def __getitem__(self, where):
        out = self._got[where]
        if out is None:
            normalized = where
            if normalized < 0:
                normalized += len(self)
            out = self._got[normalized] = self._get(normalized)
        return out

    def __repr__(self):
        tmp = [repr(x) for x in self]
        if sum(len(x) for x in tmp) < 100:
            return "[" + ", ".join(tmp) + "]"
        elif len(tmp) < 100:
            return "[" + ",\n ".join(tmp) + "]"
        else:
            return "[" + ",\n ".join(tmp[:50]) + ",\n ..." + ",\n ".join(tmp[:50]) + "]"

    def __eq__(self, other):
        if not isinstance(other, (LazyList, Iterable)):
            return False
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        else:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

def lazyproperty(name, finalize):
    def name2fb(name):
        return "".join(x.capitalize() for x in name.split("_"))

    @property
    def prop(self):
        uname = "_" + name
        if hasattr(self, uname):
            return getattr(self, uname)

        fbname = name2fb(name)
        fblenname = fbname + "Length"
        lenmethod = getattr(self._flatbuffers, fblenname, None)
        if lenmethod is None:
            out = getattr(self._flatbuffers, fbname)()
            if finalize is not None:
                out = finalize(out)
        elif finalize is None:
            out = LazyList(getattr(self._flatbuffers, fbname), lenmethod())
        else:
            out = LazyList(lambda i: finalize(getattr(self._flatbuffers, fbname)(i)), lenmethod())
        setattr(self, uname, out)
        return out

    @prop.setter
    def prop(self, value):
        setattr(self, "_" + name, value)

    return prop
