lib_LTLIBRARIES = libvftrace.la
noinst_LTLIBRARIES = libvftr_pause.la

include_HEADERS = vftrace.h
if ENABLE_FORTRAN
include_HEADERS += vftrace.mod
endif

SUBDIRS =
SUBDIRS += sorting
SUBDIRS += mpi

libvftrace_la_LIBADD =

#external libraries
libvftrace_la_LIBADD += ../external/libcJSON.la

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

SUBDIRS += logfiles
libvftrace_la_LIBADD += logfiles/libvftr_logfiles.la

SUBDIRS += hwprof
libvftrace_la_LIBADD += hwprof/libvftr_hwprof.la

if ENABLE_OMP
SUBDIRS += omp
libvftrace_la_LIBADD += omp/libvftr_omp.la
endif

if ENABLE_CUDAPROF
SUBDIRS += cuda
libvftrace_la_LIBADD += cuda/libvftr_cuda.la
endif

if ENABLE_ACCPROF
SUBDIRS += accprof
libvftrace_la_LIBADD += accprof/libvftr_accprof.la
endif

libvftrace_la_SOURCES = 

libvftrace_la_CPPFLAGS = -D_VERSION=${PACKAGE_VERSION} \
			 -D_BUGREPORT=${PACKAGE_BUGREPORT}
libvftrace_la_CFLAGS =
libvftrace_la_CXXFLAGS =
libvftrace_la_FCFLAGS =
libvftrace_la_LDFLAGS =

libvftrace_la_SOURCES += cyghooks.c \
			 pre_hooks.c \
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
			 collated_profiling.c \
			 callprofiling.c \
			 collated_callprofiling.c \
			 sampling.c \
			 timer.c \
                         signal_handling.c
libvftrace_la_SOURCES += regular_expressions.c \
			 configuration.c \
			 configuration_print.c \
			 configuration_defaults.c \
			 configuration_parse.c \
			 configuration_advisor.c \
			 configuration_assert.c \
			 elf_reader.c \
			 symbols.c
libvftrace_la_SOURCES += search.c \
			 misc_utils.c \
			 levenshtein.c \
			 bool_strings.c \
			 range_expand.c \
			 tables.c \
			 exe_info.c \
			 filenames.c \
			 realloc_consts.c \
			 license.c \
			 hashing.c

libvftrace_la_SOURCES += vfdfiles.c 

libvftrace_la_SOURCES += user_regions.c \
			 user_stack.c
if ENABLE_FORTRAN
libvftrace_la_SOURCES += vftrace_mod.F90
endif

if ENABLE_DYNLIB_TRACING
libvftrace_la_SOURCES += dlopen.c
endif

libvftrace_la_CFLAGS += -I${srcdir}/../external/cJSON/
libvftrace_la_CFLAGS += -I${srcdir}/mpi/utils/ \
			-I${srcdir}/omp/ \
                        -I${srcdir}/cuda/ \
			-I${srcdir}/accprof \
		        -I${srcdir}/hwprof \
			-I${srcdir}/sorting \
                        -I${srcdir}/logfiles \
                        -I${srcdir}/../external/tinyexpr

if ENABLE_MPI
libvftrace_la_CFLAGS += -D_MPI
libvftrace_la_FCFLAGS += -D_MPI
endif 

if ENABLE_OMP
libvftrace_la_CFLAGS += -D_OMP
libvftrace_la_FCFLAGS += -D_OMP
endif

if ON_VECTOR_ENGINE
libvftrace_la_CFLAGS += -D_ON_VE
endif

if ENABLE_CUDAPROF
libvftrace_la_CFLAGS += -D_CUDA
endif

if ENABLE_ACCPROF
libvftrace_la_CFLAGS += -D_ACCPROF
endif

if ENABLE_PAPI
libvftrace_la_CFLAGS += -D_PAPI_AVAIL
endif

if HAS_LIBERTY
libvftrace_la_CFLAGS += -D_LIBERTY
endif

if ENABLE_SHARED
libvftrace_la_LIBADD += -ldl
endif

if SELF_PROFILING
libvftrace_la_CPPFLAGS += -D_SELF_PROFILE_VFTRACE
libvftrace_la_SOURCES += self_profile.c
endif

libvftr_pause_la_SOURCES = pause.c
libvftr_pause_la_CFLAGS = $(FINSTRUMENT_FLAG) -I${srcdir}/hwprof -I${srcdir}/../external/tinyexpr

CLEANFILES = *.mod
