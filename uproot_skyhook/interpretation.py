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

import sys

import uproot
import flatbuffers

def toflatbuffers(builder, interp):
    def dtype2fb(dtype):
        if dtype.kind == "b":
            return uproot_skyhook.interpretation.DType.DType.dtype_bool

        elif dtype.kind == "i":
            if dtype.itemsize == 1:
                return uproot_skyhook.interpretation.DType.DType.dtype_int8
            elif dtype.itemsize == 2:
                return uproot_skyhook.interpretation.DType.DType.dtype_int16
            elif dtype.itemsize == 4:
                return uproot_skyhook.interpretation.DType.DType.dtype_int32
            elif dtype.itemsize == 8:
                return uproot_skyhook.interpretation.DType.DType.dtype_int64
            else:
                raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))

        elif dtype.kind == "u":
            if dtype.itemsize == 1:
                return uproot_skyhook.interpretation.DType.DType.dtype_uint8
            elif dtype.itemsize == 2:
                return uproot_skyhook.interpretation.DType.DType.dtype_uint16
            elif dtype.itemsize == 4:
                return uproot_skyhook.interpretation.DType.DType.dtype_uint32
            elif dtype.itemsize == 8:
                return uproot_skyhook.interpretation.DType.DType.dtype_uint64
            else:
                raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))

        elif dtype.kind == "t":
            if dtype.itemsize == 4:
                return uproot_skyhook.interpretation.DType.DType.dtype_float32
            elif dtype.itemsize == 8:
                return uproot_skyhook.interpretation.DType.DType.dtype_float64
            else:
                raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))

        else:
            raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))

    if isinstance(interp, uproot.asdtype):
        if interp.fromdtype.names is None and interp.todtype.names is None:
            fromdtype = dtype2fb(interp.fromdtype)

            if interp.fromdtype.kind == interp.todtype.kind and interp.fromdtype.itemsize == interp.todtype.itemsize:
                todtype = uproot_skyhook.interpretation.DType.DType.dtype_unspecified
            else:
                todtype = dtype2fb(interp.todtype)

            frombigendian = (interp.fromdtype.byteorder == ">") or (interp.fromdtype.byteorder == "=" and sys.byteorder == "big")
            tobigendian = (interp.todtype.byteorder == ">") or (interp.todtype.byteorder == "=" and sys.byteorder == "big")
            fromdims = None if interp.fromdtype.subdtype is None else interp.fromdtype.subdtype[1]
            todims = None if interp.todtype.subdtype is None else interp.todtype.subdtype[1]

            if fromdims is not None:
                uproot_skyhook.interpretation.Flat.FlatStartFromdimsVector(builder, len(fromdims))
                for x in fromdims[::-1]:
                    builder.PrependUInt(x)
                fromdims = builder.EndVector(len(fromdims))

            if todims is not None:
                uproot_skyhook.interpretation.Flat.FlatStartTodimsVector(builder, len(todims))
                for x in todims[::-1]:
                    builder.PrependUInt(x)
                todims = builder.EndVector(len(todims))

            uproot_skyhook.interpretation.Flat.FlatStart(builder)
            uproot_skyhook.interpretation.Flat.FlatAddFromdtype(fromdtype)
            if todtype != uproot_skyhook.interpretation.DType.DType.dtype_unspecified:
                uproot_skyhook.interpretation.Flat.FlatAddTodtype(todtype)
            if frombigendian is not True:
                uproot_skyhook.interpretation.Flat.FlatAddFrombigendian(frombigendian)
            if tobigendian is not True:
                uproot_skyhook.interpretation.Flat.FlatAddTobigendian(tobigendian)
            if fromdims is not None:
                uproot_skyhook.interpretation.Flat.FlatAddFromdims(fromdims)
            if todims is not None:
                uproot_skyhook.interpretation.Flat.FlatAddTodims(todims)
            return uproot_skyhook.interpretation.Flat.FlatEnd(builder)

        elif interp.fromdtype.names is not None and interp.todtype.names is not None:
            raise NotImplementedError

        else:
            raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))

    elif isinstance(interp, uproot.asdouble32):
        raise NotImplementedError

    elif isinstance(interp, uproot.asstlbitset):
        raise NotImplementedError

    elif isinstance(interp, uproot.asjagged):
        raise NotImplementedError

    elif isinstance(interp, uproot.asstring):
        raise NotImplementedError

    elif isinstance(interp, uproot.asobj):
        raise NotImplementedError

    else:
        raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))
