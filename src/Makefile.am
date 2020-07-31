lib_LTLIBRARIES = libvftrace.la libvftr_dlopen.la
noinst_LTLIBRARIES = libvftr_mpi.la \
                     libvftr_mpiwrap.la \
                     libvftr_pause.la

include_HEADERS = vftrace.h

bin_PROGRAMS = tracedump

libvftrace_la_LIBADD = libvftr_mpi.la \
                       libvftr_mpiwrap.la \
		       libvftr_pause.la

libvftr_mpi_la_SOURCES = vftr_sync_messages.c \
                         vftr_async_messages.c \
		         vftr_mpi_utils.c \
		         vftr_mpi_environment.c \
		         vftr_mpi_environment_F.c \
		         vftr_mpi_environment_F08.c \
		         vftr_mpi_point2point.c \
		         vftr_mpi_point2point_F.c \
		         vftr_mpi_collective.c \
		         vftr_mpi_collective_F.c \
		         vftr_mpi_onesided.c \
		         vftr_mpi_onesided_F.c \
		         vftr_mpi_testwait.c \
		         vftr_mpi_testwait_F.c \
                         vftr_mpi_buf_addr_const.c \
                         vftr_mpi_buf_addr_const_F.F90

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

libvftr_mpiwrap_la_SOURCES = vftr_mpi_environment_cwrap.c \
			     vftr_mpi_environment_c2F.F90 \
			     vftr_mpi_environment_Fwrap.F90 \
			     vftr_mpi_environment_c2F08.F90 \
			     vftr_mpi_environment_F08wrap.F90 \
			     vftr_mpi_point2point_cwrap.c \
			     vftr_mpi_point2point_c2F.F90 \
			     vftr_mpi_point2point_Fwrap.F90 \
			     vftr_mpi_collective_cwrap.c \
			     vftr_mpi_collective_c2F.F90 \
			     vftr_mpi_collective_Fwrap.F90 \
			     vftr_mpi_onesided_cwrap.c \
			     vftr_mpi_onesided_c2F.F90 \
			     vftr_mpi_onesided_Fwrap.F90 \
			     vftr_mpi_testwait_cwrap.c \
                             vftr_mpi_testwait_c2F.F90 \
			     vftr_mpi_testwait_Fwrap.F90

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
 		       vftr_timer.c \
		       tinyexpr.c \
		       vftr_regex.c \
	 	       vftr_environment.c \
		       vftr_signals.c \
		       vftr_loadbalance.c \
		       vftrace_mod.F90


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

libvftr_dlopen_la_SOURCES = vftr_dlopen.c

tracedump_SOURCES = tracedump.c \
		    vftr_mpi_utils.c

nodist_include_HEADERS = vftrace.mod
vftrace.mod: vftrace_mod.F90

clean-local:
	rm -f *.mod

