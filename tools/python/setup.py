from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext

import os

class vftrace_build_ext(build_ext):
  def build_extensions(self):
     build_ext.build_extensions(self)

vftr_ext = Extension("vftrace",
                     sources = ["pyhooks.c"],
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
