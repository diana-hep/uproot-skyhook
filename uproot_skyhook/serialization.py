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

import flatbuffers

import uproot
import uproot_skyhook.interpretation.DType
import uproot_skyhook.interpretation.Primitive
import uproot_skyhook.interpretation.Flat
import uproot_skyhook.interpretation.Double32
import uproot_skyhook.interpretation.STLBitSet
import uproot_skyhook.interpretation.Jagged
import uproot_skyhook.interpretation.String
import uproot_skyhook.interpretation.TableObj

def interp_toflatbuffers(builder, interp):
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

        elif dtype.kind == "f":
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
                uproot_skyhook.interpretation.Primitive.PrimitiveStartDimsVector(builder, len(fromdims))
                for x in fromdims[::-1]:
                    builder.PrependUInt(x)
                fromdims = builder.EndVector(len(fromdims))

            uproot_skyhook.interpretation.Primitive.PrimitiveStart(builder)
            uproot_skyhook.interpretation.Primitive.PrimitiveAddDtype(builder, fromdtype)
            uproot_skyhook.interpretation.Primitive.PrimitiveAddBigendian(builder, frombigendian)
            if fromdims is not None:
                uproot_skyhook.interpretation.Primitive.PrimitiveAddDims(builder, fromdims)
            fromtype = uproot_skyhook.interpretation.Primitive.PrimitiveEnd(builder)

            if todims is not None:
                uproot_skyhook.interpretation.Primitive.PrimitiveStartDimsVector(builder, len(todims))
                for x in todims[::-1]:
                    builder.PrependUInt(x)
                todims = builder.EndVector(len(todims))

            uproot_skyhook.interpretation.Primitive.PrimitiveStart(builder)
            uproot_skyhook.interpretation.Primitive.PrimitiveAddDtype(builder, todtype)
            uproot_skyhook.interpretation.Primitive.PrimitiveAddBigendian(builder, tobigendian)
            if todims is not None:
                uproot_skyhook.interpretation.Primitive.PrimitiveAddDims(builder, todims)
            totype = uproot_skyhook.interpretation.Primitive.PrimitiveEnd(builder)

            uproot_skyhook.interpretation.Flat.FlatStart(builder)
            uproot_skyhook.interpretation.Flat.FlatAddFromtype(builder, fromtype)
            uproot_skyhook.interpretation.Flat.FlatAddTotype(builder, totype)
            return uproot_skyhook.interpretation.Flat.FlatEnd(builder)

        elif interp.fromdtype.names is not None and interp.todtype.names is not None:
            fromtypes = []
            for name in interp.fromdtype.names:
                dt = dtype2fb(interp.fromdtype[name])
                big = (interp.fromdtype[name].byteorder == ">") or (interp.fromdtype[name].byteorder == "=" and sys.byteorder == "big")
                if interp.fromdtype[name].subdtype is not None:
                    raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))
                uproot_skyhook.interpretation.Primitive.PrimitiveStart(builder)
                uproot_skyhook.interpretation.Primitive.PrimitiveAddDtype(builder, dt)
                uproot_skyhook.interpretation.Primitive.PrimitiveAddBigendian(builder, big)
                fromtypes.append(uproot_skyhook.interpretation.Primitive.PrimitiveEnd(builder))

            totypes = []
            for name in interp.todtype.names:
                dt = dtype2fb(interp.todtype[name])
                big = (interp.todtype[name].byteorder == ">") or (interp.todtype[name].byteorder == "=" and sys.byteorder == "big")
                if interp.todtype[name].subdtype is not None:
                    raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))
                uproot_skyhook.interpretation.Primitive.PrimitiveStart(builder)
                uproot_skyhook.interpretation.Primitive.PrimitiveAddDtype(builder, dt)
                uproot_skyhook.interpretation.Primitive.PrimitiveAddBigendian(builder, big)
                totypes.append(uproot_skyhook.interpretation.Primitive.PrimitiveEnd(builder))

            fromnames = [builder.CreateString(x.encode("utf-8")) for x in interp.fromdtype.names]
            tonames = [builder.CreateString(x.encode("utf-8")) for x in interp.todtype.names]

            uproot_skyhook.interpretation.Record.RecordStartFromtypesVector(builder, len(fromtypes))
            for x in fromtypes[::-1]:
                builder.PrependUOffsetTRelative(x)
            fromtypes = builder.EndVector(len(fromtypes))

            uproot_skyhook.interpretation.Record.RecordStartTotypesVector(builder, len(totypes))
            for x in totypes[::-1]:
                builder.PrependUOffsetTRelative(x)
            totypes = builder.EndVector(len(totypes))

            uproot_skyhook.interpretation.Record.RecordStartFromnamesVector(builder, len(fromnames))
            for x in fromnames[::-1]:
                builder.PrependUOffsetTRelative(x)
            fromnames = builder.EndVector(len(fromnames))

            uproot_skyhook.interpretation.Record.RecordStartTonamesVector(builder, len(tonames))
            for x in tonames[::-1]:
                builder.PrependUOffsetTRelative(x)
            tonames = builder.EndVector(len(tonames))

            uproot_skyhook.interpretation.Record.RecordStart(builder)
            uproot_skyhook.interpretation.Record.RecordAddFromtypes(builder, fromtypes)
            uproot_skyhook.interpretation.Record.RecordAddTotypes(builder, totypes)
            uproot_skyhook.interpretation.Record.RecordAddFromnames(builder, fromnames)
            uproot_skyhook.interpretation.Record.RecordAddTonames(builder, tonames)
            return uproot_skyhook.interpretation.Record.RecordEnd(builder)

        else:
            raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))

    elif isinstance(interp, uproot.asdouble32):
        fromdims = None if len(interp.fromdims) is None else interp.fromdims
        todims = None if len(interp.todims) is None else interp.todims

        if fromdims is not None:
            uproot_skyhook.interpretation.Double32.Double32StartFromdimsVector(builder, len(fromdims))
            for x in fromdims[::-1]:
                builder.PrependUInt(x)
            fromdims = builder.EndVector(len(fromdims))

        if todims is not None:
            uproot_skyhook.interpretation.Double32.Double32StartTodimsVector(builder, len(todims))
            for x in todims[::-1]:
                builder.PrependUInt(x)
            todims = builder.EndVector(len(todims))

        uproot_skyhook.interpretation.Double32.Double32Start(builder)
        uproot_skyhook.interpretation.Double32.Double32AddLow(builder, interp.low)
        uproot_skyhook.interpretation.Double32.Double32AddHigh(builder, interp.high)
        uproot_skyhook.interpretation.Double32.Double32AddNumbits(builder, interp.numbits)
        if fromdims is not None:
            uproot_skyhook.interpretation.Double32.Double32AddFromdims(builder, fromdims)
        if todims is not None:
            uproot_skyhook.interpretation.Double32.Double32AddTodims(builder, todims)
        return uproot_skyhook.interpretation.Double32.Double32End(builder)

    elif isinstance(interp, uproot.asstlbitset):
        uproot_skyhook.interpretation.STLBitSet.STLBitSetStart(builder)
        uproot_skyhook.interpretation.STLBitSet.STLBitSetAddNumbytes(builder, interp.numbytes)
        return uproot_skyhook.interpretation.STLBitSet.STLBitSetEnd(builder)

    elif isinstance(interp, uproot.asjagged):
        content = toflatbuffers(builder, interp.content)
        uproot_skyhook.interpretation.Jagged.JaggedStart(builder)
        uproot_skyhook.interpretation.Jagged.JaggedAddContent(builder, content)
        uproot_skyhook.interpretation.Jagged.JaggedAddSkipbytes(builder, interp.skipbytes)
        return uproot_skyhook.interpretation.Jagged.JaggedEnd(builder)

    elif isinstance(interp, uproot.asstring):
        uproot_skyhook.interpretation.String.StringStart(builder)
        uproot_skyhook.interpretation.String.StringAddSkipbytes(builder, interp.skipbytes)
        return uproot_skyhook.interpretation.String.StringEnd(builder)

    elif isinstance(interp, uproot.asobj) and isinstance(interp.content, uproot.astable):
        content = toflatbuffers(builder, interp.content.content)
        qualname = [builder.CreateString(x.encode("utf-8")) for x in (interp.cls.__module__, interp.cls.__name__)]
        uproot_skyhook.interpretation.TableObj.TableObjStartQualnameVector(builder, len(qualname))
        for x in qualname[::-1]:
            builder.PrependUOffsetTRelative(x)
        qualname = builder.EndVector(len(qualname))

        uproot_skyhook.interpretation.TableObj.TableObjStart(builder)
        uproot_skyhook.interpretation.TableObj.TableObjAddContent(builder, content)
        uproot_skyhook.interpretation.TableObj.TableObjAddQualname(builder, qualname)
        return uproot_skyhook.interpretation.TableObj.TableObjEnd(builder)

    else:
        raise NotImplementedError("SkyHook layout of Interpretation {0} not implemented".format(repr(interp)))
