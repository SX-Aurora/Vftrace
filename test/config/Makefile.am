if ENABLE_MPI
if USES_OPEN_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_openmpi.sh;
else
if USES_NEC_MPI
if ON_VECTOR_ENGINE
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necvhmpi.sh;
endif
else
if USES_INTEL_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_intelmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_genericmpi.sh;
endif
endif
endif
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_nompi.sh;
endif

if ENABLE_SHARED
   AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/set_shared.sh;
else
   AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/set_noshared.sh;
endif

AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/unset_vftr_env.sh;

AM_CPPFLAGS =
AM_CFLAGS = -I$(top_srcdir)/src/
AM_FCFLAGS = -I$(top_srcdir)/src/
AM_CXXFLAGS = -I$(top_srcdir)/src/
AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if ENABLE_MPI
  AM_CFLAGS += -D_MPI
  AM_CXXFLAGS += -D_MPI
if ENABLE_FORTRAN
  AM_FCFLAGS += -D_MPI
endif
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

if ENABLE_OMP
  AM_CFLAGS += -D_OMP
  AM_CXXFLAGS += -D_OMP
if ENABLE_FORTRAN
  AM_FCFLAGS += -D_OMP
endif
endif

if SELF_PROFILING
   AM_CPPFLAGS += -D_SELF_PROFILE_VFTRACE
endif

check_PROGRAMS = 
dist_check_SCRIPTS =

check_PROGRAMS += defaults \
		  advisor \
		  advisor2

defaults_SOURCES = $(srcdir)/defaults.c
defaults_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
defaults_LDFLAGS = $(AM_LDFLAGS)

advisor_SOURCES = $(srcdir)/advisor.c
advisor_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
advisor_LDFLAGS = $(AM_LDFLAGS)

advisor2_SOURCES = $(srcdir)/little_tasks.c
advisor2_CFLAGS = $(AM_CFLAGS) -finstrument-functions
advisor2_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += defaults.sh 
#		      advisor.sh \
#		      advisor2.sh

if !SELF_PROFILING
check_PROGRAMS += off
endif
check_PROGRAMS += sampling \
		  out_directory \
		  outfile_basename \
		  logfile_for_ranks \
		  print_config \
		  ranks_in_mpi_profile \
		  show_calltime_imbalances

off_SOURCES = $(srcdir)/little_tasks.c
off_CFLAGS = $(AM_CFLAGS) -finstrument-functions
off_LDFLAGS = $(AM_LDFLAGS)

sampling_SOURCES = $(srcdir)/little_tasks.c
sampling_CFLAGS = $(AM_CFLAGS) -finstrument-functions
sampling_LDFLAGS = $(AM_LDFLAGS)

out_directory_SOURCES = $(srcdir)/little_tasks.c
out_directory_CFLAGS = $(AM_CFLAGS) -finstrument-functions
out_directory_LDFLAGS = $(AM_LDFLAGS)

outfile_basename_SOURCES = $(srcdir)/little_tasks.c
outfile_basename_CFLAGS = $(AM_CFLAGS) -finstrument-functions
outfile_basename_LDFLAGS = $(AM_LDFLAGS)

logfile_for_ranks_SOURCES = $(srcdir)/little_tasks.c
logfile_for_ranks_CFLAGS = $(AM_CFLAGS) -finstrument-functions
logfile_for_ranks_LDFLAGS = $(AM_LDFLAGS)

print_config_SOURCES = $(srcdir)/little_tasks.c
print_config_CFLAGS = $(AM_CFLAGS) -finstrument-functions
print_config_LDFLAGS = $(AM_LDFLAGS)

ranks_in_mpi_profile_SOURCES = $(srcdir)/mpi_tasks.c
ranks_in_mpi_profile_CFLAGS = $(AM_CFLAGS) -finstrument-functions
ranks_in_mpi_profile_LDFLAGS = $(AM_LDFLAGS)

show_calltime_imbalances_SOURCES = $(srcdir)/little_tasks.c
show_calltime_imbalances_CFLAGS = $(AM_CFLAGS) -finstrument-functions
show_calltime_imbalances_LDFLAGS = $(AM_LDFLAGS)

if !SELF_PROFILING
dist_check_SCRIPTS += off.sh
endif
dist_check_SCRIPTS += sampling.sh \
		      out_directory.sh \
		      outfile_basename.sh \
		      logfile_for_ranks.sh \
		      print_config.sh \
		      ranks_in_mpi_profile.sh \
		      show_calltime_imbalances.sh

if ENABLE_FORTRAN
check_PROGRAMS += strip_module_names

strip_module_names_SOURCES = $(srcdir)/little_ftasks.F90
strip_module_names_FCFLAGS = $(AM_FCFLAGS) -finstrument-functions
strip_module_names_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += strip_module_names.sh
endif

if HAS_LIBERTY
check_PROGRAMS += demangle_cxx

demangle_cxx_SOURCES = $(srcdir)/little_cxxtasks.cpp
demangle_cxx_CXXFLAGS = $(AM_CXXFLAGS) -finstrument-functions
demangle_cxx_LDFLAGS = $(AM_LDFLAGS) -liberty

dist_check_SCRIPTS += demangle_cxx.sh
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)

CLEANFILES = vftrace_self_profile-* \
	     *.json
