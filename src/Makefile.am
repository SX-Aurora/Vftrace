lib_LTLIBRARIES = libvftrace.la
noinst_LTLIBRARIES = libvftr_pause.la

include_HEADERS = vftrace.h
if ENABLE_FORTRAN
include_HEADERS += vftrace.mod
endif

SUBDIRS =
SUBDIRS += sorting
SUBDIRS += mpi

external_includes = -I$(top_srcdir)/external/tinyexpr/ \
		    -I$(top_srcdir)/external/jsmn/


# external libraries
libvftrace_la_LIBADD =
libvftrace_la_LIBADD += ${top_builddir}/external/libtinyexpr.la

#internal libraries
libvftrace_la_LIBADD += libvftr_pause.la
libvftrace_la_LIBADD += sorting/libvftr_sorting.la
libvftrace_la_LIBADD += mpi/utils/libvftr_mpi_utils.la
if ENABLE_MPI
libvftrace_la_LIBADD += mpi/vftr_mpi/libvftr_mpi.la
libvftrace_la_LIBADD += mpi/c-wrapper/libvftr_cmpiwrap.la \
			mpi/c2vftr/libvftr_mpi_c2vftr.la
if ENABLE_FORTRAN
libvftrace_la_LIBADD += mpi/f-wrapper/libvftr_fmpiwrap.la \
			mpi/f2vftr/libvftr_mpi_f2vftr.la
if ENABLE_FORTRAN08
libvftrace_la_LIBADD += mpi/f08-wrapper/libvftr_f08mpiwrap.la \
			mpi/f082vftr/libvftr_mpi_f082vftr.la
endif
endif
endif

if ENABLE_OMP
SUBDIRS += omp
libvftrace_la_LIBADD += omp/libvftr_omp.la
endif

libvftrace_la_SOURCES = 

libvftrace_la_CPPFLAGS = -D_VERSION=${PACKAGE_VERSION} \
			 -D_BUGREPORT=${PACKAGE_BUGREPORT}
libvftrace_la_CFLAGS =
libvftrace_la_CXXFLAGS =
libvftrace_la_FCFLAGS =
libvftrace_la_LDFLAGS =

libvftrace_la_SOURCES += cyghooks.c \
			 vftr_hooks.c \
			 off_hooks.c \
			 pause_hooks.c
libvftrace_la_SOURCES += vftrace_state.c \
			 vftr_initialize.c \
			 vftr_finalize.c \
			 stacks.c \
			 collate_ranks.c \
			 collate_stacks.c \
			 collate_profiles.c \
			 collate_callprofiles.c \
			 collate_hashes.c \
			 collate_vftr_size.c \
			 processes.c \
			 threads.c \
			 threadstacks.c \
			 internal_regions.c \
			 profiling.c \
			 callprofiling.c \
			 sampling.c \
			 timer.c
libvftrace_la_SOURCES += regular_expressions.c \
			 environment.c \
			 ElfFormat.c \
			 symbols.c \
			 dlopen.c
libvftrace_la_SOURCES += search.c \
			 misc_utils.c \
			 range_expand.c \
			 tables.c \
			 exe_info.c \
			 filenames.c \
			 realloc_consts.c \
			 license.c \
			 hashing.c
libvftrace_la_SOURCES += logfile.c \
			 logfile_header.c \
			 logfile_prof_table.c \
			 logfile_stacklist.c \
			 ranklogfile.c \
			 ranklogfile_header.c \
			 ranklogfile_prof_table.c \
			 vfdfiles.c
libvftrace_la_SOURCES += user_regions.c \
			 user_stack.c
if ENABLE_FORTRAN
libvftrace_la_SOURCES += vftrace_mod.F90
endif

libvftrace_la_CFLAGS += -I${srcdir}/mpi/utils/ \
			-I${srcdir}/omp/ \
			-I${srcdir}/sorting \
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

if ENABLE_MPI
libvftrace_la_CFLAGS += -D_MPI
libvftrace_la_FCFLAGS += -D_MPI
endif 

if ENABLE_OMP
libvftrace_la_CFLAGS += -D_OMP
libvftrace_la_FCFLAGS += -D_OMP
endif

if HAS_LIBERTY
libvftrace_la_CFLAGS += -D_LIBERTY
#libvftrace_la_LIBADD += -liberty
endif

#if USES_INTEL_COMPILER
#libvftrace_la_LDFLAGS += -lifcore
#endif

libvftr_pause_la_SOURCES = pause.c
libvftr_pause_la_CFLAGS = -finstrument-functions

clean-local:
	rm -f *.mod
