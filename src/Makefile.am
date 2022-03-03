lib_LTLIBRARIES = libvftrace.la

#include_HEADERS = vftrace.h
#if ENABLE_FORTRAN
#include_HEADERS += vftrace.mod
#endif

#SUBDIRS = mpi
#
#external_includes = -I$(top_srcdir)/external/tinyexpr/ \
#		    -I$(top_srcdir)/external/jsmn/
#
## external libraries
#libvftrace_la_LIBADD += ${top_builddir}/external/libtinyexpr.la
#
#libvftrace_la_LIBADD += mpi/utils/libvftr_mpi_utils.la
#if WITH_MPI
#libvftrace_la_LIBADD += mpi/vftr_mpi/libvftr_mpi.la
#libvftrace_la_LIBADD += mpi/c-wrapper/libvftr_cmpiwrap.la \
#			mpi/c2vftr/libvftr_mpi_c2vftr.la
#if ENABLE_FORTRAN
#libvftrace_la_LIBADD += mpi/f-wrapper/libvftr_fmpiwrap.la \
#			mpi/f2vftr/libvftr_mpi_f2vftr.la
#if ENABLE_FORTRAN08
#libvftrace_la_LIBADD += mpi/f08-wrapper/libvftr_f08mpiwrap.la \
#			mpi/f082vftr/libvftr_mpi_f082vftr.la
#endif
#endif
#endif

libvftrace_la_SOURCES = 

libvftrace_la_CPPFLAGS = -D_VERSION=${PACKAGE_VERSION} \
			 -D_BUGREPORT=${PACKAGE_BUGREPORT}
libvftrace_la_CFLAGS =
libvftrace_la_CXXFLAGS =
libvftrace_la_FCFLAGS =
libvftrace_la_LDFLAGS =

libvftrace_la_SOURCES += cyghooks.c \
			 vftr_hooks.c \
			 off_hooks.c
libvftrace_la_SOURCES += vftrace_state.c \
			 initialize.c \
			 finalize.c \
			 stacks.c \
			 processes.c \
			 threads.c \
			 timer.c
libvftrace_la_SOURCES += regular_expressions.c \
			 environment.c \
			 ElfFormat.c \
			 symbols.c
libvftrace_la_SOURCES += sorting.c \
			 search.c

libvftrace_la_CFLAGS += -I${srcdir}/mpi/utils/ \
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
#libvftrace_la_LIBADD += -liberty
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

clean-local:
	rm -f *.mod

