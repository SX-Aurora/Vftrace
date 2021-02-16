lib_LTLIBRARIES = libvftrace.la libvftr_dlopen.la
noinst_LTLIBRARIES = libvftr_pause.la

include_HEADERS = vftrace.h

SUBDIRS = vftr_mpi


libvftrace_la_LIBADD = libvftr_pause.la \
		       vftr_mpi/libvftr_mpi.la
if WITH_MPI
libvftrace_la_LIBADD += vftr_mpi/c-wrapper/libvftr_cmpiwrap.la \
		        vftr_mpi/f-wrapper/libvftr_fmpiwrap.la \
		        vftr_mpi/f08-wrapper/libvftr_f08mpiwrap.la
endif

libvftr_pause_la_SOURCES = vftr_pause.c
libvftr_pause_la_CFLAGS = -finstrument-functions

libvftrace_la_SOURCES = demangle.cpp \
		       vftr_scenarios.c \
		       vftr_symbols.c \
                       vftr_hwcounters.c \
		       vftr_hooks.c \
		       vftr_setup.c \
		       vftr_filewrite.c \
		       vftr_fileutils.c \
		       vftr_hashing.c \
		       vftr_sorting.c \
		       vftr_stacks.c \
		       vftr_user_stack.c \
		       vftr_functions.c \
		       vftr_regions.c \
                       vftr_allocate.c \
 		       vftr_timer.c \
		       tinyexpr.c \
		       vftr_regex.c \
	 	       vftr_environment.c \
		       vftr_signals.c \
		       vftr_browse.c \
		       vftrace_mod.F90


libvftrace_la_CFLAGS =
libvftrace_la_CXXFLAGS =
libvftrace_la_FCFLAGS =
libvftrace_la_LDFLAGS =

libvftrace_la_CFLAGS += -I${srcdir}/vftr_mpi/

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

if USES_INTEL_MPI
libvftrace_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftrace_la_FCFLAGS += -DPMPI_MODULE=mpi
endif

if USES_INTEL_COMPILER
libvftrace_la_LDFLAGS += -lifcore
endif

libvftr_dlopen_la_SOURCES = vftr_dlopen.c

nodist_include_HEADERS = vftrace.mod
vftrace.mod: vftrace_mod.F90

clean-local:
	rm -f *.mod

