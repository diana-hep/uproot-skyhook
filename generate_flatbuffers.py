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
import shutil

if __name__ == "__main__":
    os.rename(os.path.join("uproot_skyhook", "__init__.py"), os.path.join("uproot_skyhook.__init__.py"))

    if os.path.exists(os.path.join("uproot_skyhook", "interpretation_generated")):
        shutil.rmtree(os.path.join("uproot_skyhook", "interpretation_generated"))

    if os.path.exists(os.path.join("uproot_skyhook", "layout_generated")):
        shutil.rmtree(os.path.join("uproot_skyhook", "layout_generated"))

    os.system("flatc --python interpretation.fbs")
    os.system("flatc --python layout.fbs")

    with open(os.path.join("uproot_skyhook", "layout_generated", "Column.py")) as f:
        tmp = f.read().replace("from .Interpretation import Interpretation", "from uproot_skyhook.interpretation_generated.Interpretation import Interpretation")
    with open(os.path.join("uproot_skyhook", "layout_generated", "Column.py"), "w") as f:
        f.write(tmp)

    os.rename(os.path.join("uproot_skyhook.__init__.py"), os.path.join("uproot_skyhook", "__init__.py"))
