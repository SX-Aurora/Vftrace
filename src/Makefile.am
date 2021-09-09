lib_LTLIBRARIES = libvftrace.la libvftr_dlopen.la
noinst_LTLIBRARIES = libvftr_pause.la

include_HEADERS = vftrace.h

SUBDIRS = mpi

external_includes = -I$(top_srcdir)/external/tinyexpr/ \
		    -I$(top_srcdir)/external/jsmn/

libvftrace_la_LIBADD = libvftr_pause.la \
		       mpi/vftr_mpi/libvftr_mpi.la
# external libraries
libvftrace_la_LIBADD += ${top_builddir}/external/libtinyexpr.la

if WITH_MPI
libvftrace_la_LIBADD += mpi/c-wrapper/libvftr_cmpiwrap.la
if ENABLE_FORTRAN
libvftrace_la_LIBADD += mpi/f-wrapper/libvftr_fmpiwrap.la \
			mpi/f2vftr/libvftr_mpi_f2vftr.la
if ENABLE_FORTRAN08
libvftrace_la_LIBADD += mpi/f08-wrapper/libvftr_f08mpiwrap.la \
			mpi/f082vftr/libvftr_mpi_f082vftr.la
endif
endif
endif

libvftr_pause_la_SOURCES = vftr_pause.c
libvftr_pause_la_CFLAGS = -finstrument-functions

libvftrace_la_SOURCES = vftr_scenarios.c \
		       vftr_symbols.c \
                       vftr_hwcounters.c \
		       vftr_hooks.c \
		       vftr_setup.c \
		       vftr_filewrite.c \
		       vftr_fileutils.c \
		       vftr_stringutils.c \
		       vftr_hashing.c \
		       vftr_sorting.c \
		       vftr_stacks.c \
		       vftr_user_stack.c \
		       vftr_functions.c \
		       vftr_regions.c \
                       vftr_allocate.c \
 		       vftr_timer.c \
		       vftr_regex.c \
	 	       vftr_environment.c \
		       vftr_signals.c \
		       vftr_browse.c \
                       vftr_mallinfo.c

if ENABLE_FORTRAN
libvftrace_la_SOURCES += vftrace_mod.F90
endif

libvftrace_la_CPPFLAGS = -D_VERSION=${PACKAGE_VERSION} \
			 -D_BUGREPORT=${PACKAGE_BUGREPORT}
libvftrace_la_CFLAGS =
libvftrace_la_CXXFLAGS =
libvftrace_la_FCFLAGS =
libvftrace_la_LDFLAGS =

libvftrace_la_CFLAGS += -I${srcdir}/mpi/vftr_mpi/ \
			${external_includes}

if ENABLE_HWC
libvftrace_la_CFLAGS += -DHAS_HWC
if ON_VECTOR_ENGINE
libvftrace_la_CFLAGS += -DHAS_SXHWC
endif
if HAS_PAPI
libvftrace_la_CFLAGS += -DHAS_PAPI
endif
endif

if VMAP_OFFSET
libvftrace_la_CFLAGS += -D__VMAP_OFFSET
endif

if WITH_MPI
libvftrace_la_CFLAGS += -D_MPI
libvftrace_la_FCFLAGS += -D_MPI
endif 

if HAS_LIBERTY
libvftrace_la_CFLAGS += -D_LIBERTY_AVAIL
libvftrace_la_LIBADD += -liberty
endif

if USES_INTEL_COMPILER
libvftrace_la_LDFLAGS += -lifcore
endif

if ENABLE_OPENMP

if USES_INTEL_COMPILER
libvftrace_la_CFLAGS += -qopenmp
libvftrace_la_FCFLAGS += -qopenmp
endif

if USES_GNU_COMPILER
libvftrace_la_CFLAGS += -fopenmp
libvftrace_la_FCFLAGS += -fopenmp
endif

if USES_NEC_COMPILER
libvftrace_la_CFLAGS += -fopenmp
libvftrace_la_FCFLAGS += -fopenmp
endif

endif

libvftr_dlopen_la_SOURCES = vftr_dlopen.c

if ENABLE_FORTRAN
include_HEADERS += vftrace.mod
endif

clean-local:
	rm -f *.mod

