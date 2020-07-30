SRCPATH = src

ACLOCAL_AMFLAGS="-Im4"

lib_LTLIBRARIES = libvftrace.la libvftr_dlopen.la
noinst_LTLIBRARIES = libvftr_mpi.la \
                     libvftr_mpiwrap.la \
                     libvftr_pause.la

include_HEADERS = $(SRCPATH)/vftrace.h

MANDIR = share_src/man
man_MANS = $(MANDIR)/man1/vftrace.1

libvftrace_la_LIBADD = libvftr_mpi.la \
                       libvftr_mpiwrap.la \
		       libvftr_pause.la

libvftr_mpi_la_SOURCES = $(SRCPATH)/vftr_sync_messages.c \
                         $(SRCPATH)/vftr_async_messages.c \
		         $(SRCPATH)/vftr_mpi_utils.c \
		         $(SRCPATH)/vftr_mpi_environment.c \
		         $(SRCPATH)/vftr_mpi_environment_F.c \
		         $(SRCPATH)/vftr_mpi_environment_F08.c \
		         $(SRCPATH)/vftr_mpi_point2point.c \
		         $(SRCPATH)/vftr_mpi_point2point_F.c \
		         $(SRCPATH)/vftr_mpi_collective.c \
		         $(SRCPATH)/vftr_mpi_collective_F.c \
		         $(SRCPATH)/vftr_mpi_onesided.c \
		         $(SRCPATH)/vftr_mpi_onesided_F.c \
		         $(SRCPATH)/vftr_mpi_testwait.c \
		         $(SRCPATH)/vftr_mpi_testwait_F.c \
                         $(SRCPATH)/vftr_mpi_buf_addr_const.c \
                         $(SRCPATH)/vftr_mpi_buf_addr_const_F.F90

libvftr_mpi_la_CFLAGS =
libvftr_mpi_la_FCFLAGS =
if MPI_PROFILING
libvftr_mpi_la_CFLAGS += -D_MPI
libvftr_mpi_la_FCFLAGS += -D_MPI
endif

if USES_INTEL_MPI
libvftr_mpi_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftr_mpi_la_FCFLAGS += -DPMPI_MODULE=mpi
endif

libvftr_mpiwrap_la_SOURCES = $(SRCPATH)/vftr_mpi_environment_cwrap.c \
			     $(SRCPATH)/vftr_mpi_environment_c2F.F90 \
			     $(SRCPATH)/vftr_mpi_environment_Fwrap.F90 \
			     $(SRCPATH)/vftr_mpi_environment_c2F08.F90 \
			     $(SRCPATH)/vftr_mpi_environment_F08wrap.F90 \
			     $(SRCPATH)/vftr_mpi_point2point_cwrap.c \
			     $(SRCPATH)/vftr_mpi_point2point_c2F.F90 \
			     $(SRCPATH)/vftr_mpi_point2point_Fwrap.F90 \
			     $(SRCPATH)/vftr_mpi_point2point_F08wrap.F90 \
			     $(SRCPATH)/vftr_mpi_collective_cwrap.c \
			     $(SRCPATH)/vftr_mpi_collective_c2F.F90 \
			     $(SRCPATH)/vftr_mpi_collective_Fwrap.F90 \
			     $(SRCPATH)/vftr_mpi_collective_F08wrap.F90 \
			     $(SRCPATH)/vftr_mpi_onesided_cwrap.c \
			     $(SRCPATH)/vftr_mpi_onesided_c2F.F90 \
			     $(SRCPATH)/vftr_mpi_onesided_Fwrap.F90 \
			     $(SRCPATH)/vftr_mpi_testwait_cwrap.c \
                             $(SRCPATH)/vftr_mpi_testwait_c2F.F90 \
			     $(SRCPATH)/vftr_mpi_testwait_Fwrap.F90

libvftr_mpiwrap_la_CFLAGS = -finstrument-functions
libvftr_mpiwrap_la_FCFLAGS = -finstrument-functions
if MPI_PROFILING
libvftr_mpiwrap_la_CFLAGS += -D_MPI
libvftr_mpiwrap_la_FCFLAGS += -D_MPI
endif 

if USES_INTEL_MPI
libvftr_mpiwrap_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftr_mpiwrap_la_FCFLAGS += -DPMPI_MODULE=mpi
endif

libvftr_pause_la_SOURCES = $(SRCPATH)/vftr_pause.c
libvftr_pause_la_CFLAGS = -finstrument-functions

libvftrace_la_SOURCES = $(SRCPATH)/demangle.cpp \
		       $(SRCPATH)/vftr_scenarios.c \
		       $(SRCPATH)/vftr_symbols.c \
                       $(SRCPATH)/vftr_hwcounters.c \
		       $(SRCPATH)/vftr_hooks.c \
		       $(SRCPATH)/vftr_setup.c \
		       $(SRCPATH)/vftr_filewrite.c \
		       $(SRCPATH)/vftr_fileutils.c \
		       $(SRCPATH)/vftr_hashing.c \
		       $(SRCPATH)/vftr_sorting.c \
		       $(SRCPATH)/vftr_stacks.c \
		       $(SRCPATH)/vftr_user_stack.c \
		       $(SRCPATH)/vftr_functions.c \
		       $(SRCPATH)/vftr_regions.c \
 		       $(SRCPATH)/vftr_timer.c \
		       $(SRCPATH)/tinyexpr.c \
		       $(SRCPATH)/vftr_regex.c \
	 	       $(SRCPATH)/vftr_environment.c \
		       $(SRCPATH)/vftr_signals.c \
		       $(SRCPATH)/vftr_loadbalance.c \
		       $(SRCPATH)/vftrace_mod.F90


libvftrace_la_CFLAGS =
libvftrace_la_CXXFLAGS =
libvftrace_la_FCFLAGS =
libvftrace_la_LDFLAGS =

if PAPI_AVAIL
libvftrace_la_CFLAGS += -DHAS_PAPI -I$(PAPI_HOME)/include
libvftrace_la_LDFLAGS += -L$(PAPI_HOME)/lib -lpapi
endif

if VEPERF_AVAIL
libvftrace_la_CFLAGS += -DHAS_VEPERF -I$(VEPERF_HOME)/include
libvftrace_la_LDFLAGS +=  -L$(VEPERF_HOME)/lib -lveperf_ext
endif

if VMAP_OFFSET
libvftrace_la_CFLAGS += -D__VMAP_OFFSET
endif

if MPI_PROFILING
libvftrace_la_CFLAGS += -D_MPI
libvftrace_la_FCFLAGS += -D_MPI
endif 

if USES_INTEL_MPI
libvftrace_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftrace_la_FCFLAGS += -DPMPI_MODULE=mpi
endif

libvftr_dlopen_la_SOURCES = $(SRCPATH)/vftr_dlopen.c

nodist_include_HEADERS = vftrace.mod
vftrace.mod: $(SRCPATH)/vftrace_mod.F90

clean-local:
	rm -f *.mod
