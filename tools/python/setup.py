from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

import sys, os

class vftrace_build_ext(build_ext):
  def build_extensions(self):
     build_ext.build_extensions(self)

python_version = sys.version
sources = []
if "3.8" in python_version:
   sources.append("pyhooks_308.c")
elif "3.12" in python_version:
   sources.append("pyhooks_312.c")
else:
   sources.append("pyhooks_308.c")


vftr_ext = Extension("vftrace",
                     sources = sources,
                     extra_compile_args=['-I/home/cweiss/Vftrace/src',
                                         '-I/home/cweiss/Vftrace/src/hwprof',
                                         '-I/home/cweiss/Vftrace/external/tinyexpr',
                     ],
                     extra_objects=['-L/home/cweiss/Vftrace/build/src/.libs',
                                    '-lvftrace']
)

setup(name = "vftrace",
      cmdclass={"build_ext": vftrace_build_ext,},
      #extra_link_args=[""],
      version="0.0.1",
      description="Vftrace interface for python",
      ext_modules=[vftr_ext])
