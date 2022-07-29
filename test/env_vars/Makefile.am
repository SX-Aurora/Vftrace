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
AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/unset_vftr_env.sh;

AM_CFLAGS = -I$(top_srcdir)/src/
AM_FCFLAGS = -I$(top_srcdir)/src/
AM_CXXFLAGS = -I$(top_srcdir)/src/
AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if ENABLE_MPI
  AM_CFLAGS += -D_MPI
if ENABLE_FORTRAN
  AM_FCFLAGS += -D_MPI
endif
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

if ENABLE_OMP
  AM_CFLAGS += -D_OMP
if ENABLE_FORTRAN
  AM_FCFLAGS += -D_OMP
endif
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

dist_check_SCRIPTS += defaults.sh \
		      advisor.sh \
		      advisor2.sh

check_PROGRAMS += vftr_off \
		  vftr_sampling \
		  vftr_out_directory \
		  vftr_logfile_basename \
		  vftr_logfile_for_ranks \
		  vftr_logfile_for_ranks_assert \
		  vftr_ranks_in_mpi_profile \
		  vftr_ranks_in_mpi_profile_assert \
		  vftr_sampletime_assert \
		  vftr_scenario_file_assert \
		  vftr_sort_profile_table_assert

vftr_off_SOURCES = $(srcdir)/little_tasks.c
vftr_off_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_off_LDFLAGS = $(AM_LDFLAGS)

vftr_sampling_SOURCES = $(srcdir)/little_tasks.c
vftr_sampling_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_sampling_LDFLAGS = $(AM_LDFLAGS)

vftr_out_directory_SOURCES = $(srcdir)/little_tasks.c
vftr_out_directory_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_out_directory_LDFLAGS = $(AM_LDFLAGS)

vftr_logfile_basename_SOURCES = $(srcdir)/little_tasks.c
vftr_logfile_basename_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_logfile_basename_LDFLAGS = $(AM_LDFLAGS)

vftr_logfile_for_ranks_SOURCES = $(srcdir)/little_tasks.c
vftr_logfile_for_ranks_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_logfile_for_ranks_LDFLAGS = $(AM_LDFLAGS)

vftr_logfile_for_ranks_assert_SOURCES = $(srcdir)/little_tasks.c
vftr_logfile_for_ranks_assert_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_logfile_for_ranks_assert_LDFLAGS = $(AM_LDFLAGS)

vftr_ranks_in_mpi_profile_SOURCES = $(srcdir)/mpi_tasks.c
vftr_ranks_in_mpi_profile_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_ranks_in_mpi_profile_LDFLAGS = $(AM_LDFLAGS)

vftr_ranks_in_mpi_profile_assert_SOURCES = $(srcdir)/mpi_tasks.c
vftr_ranks_in_mpi_profile_assert_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_ranks_in_mpi_profile_assert_LDFLAGS = $(AM_LDFLAGS)

vftr_sampletime_assert_SOURCES = $(srcdir)/little_tasks.c
vftr_sampletime_assert_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_sampletime_assert_LDFLAGS = $(AM_LDFLAGS)

vftr_scenario_file_assert_SOURCES = $(srcdir)/little_tasks.c
vftr_scenario_file_assert_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_scenario_file_assert_LDFLAGS = $(AM_LDFLAGS)

vftr_sort_profile_table_assert_SOURCES = $(srcdir)/little_tasks.c
vftr_sort_profile_table_assert_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_sort_profile_table_assert_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += vftr_off.sh \
		      vftr_sampling.sh \
		      vftr_out_directory.sh \
		      vftr_logfile_basename.sh \
		      vftr_logfile_for_ranks.sh \
		      vftr_logfile_for_ranks_assert.sh \
		      vftr_ranks_in_mpi_profile.sh \
		      vftr_ranks_in_mpi_profile_assert.sh \
		      vftr_sampletime_assert.sh \
		      vftr_scenario_file_assert.sh \
		      vftr_sort_profile_table_assert.sh

if ENABLE_FORTRAN
check_PROGRAMS += vftr_strip_module_names

vftr_strip_module_names_SOURCES = $(srcdir)/little_ftasks.F90
vftr_strip_module_names_FCFLAGS = $(AM_FCFLAGS) -finstrument-functions
vftr_strip_module_names_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += vftr_strip_module_names.sh
endif

if HAS_LIBERTY
check_PROGRAMS += vftr_demangle_cxx

vftr_demangle_cxx_SOURCES = $(srcdir)/little_cxxtasks.cpp
vftr_demangle_cxx_CXXFLAGS = $(AM_CXXFLAGS) -finstrument-functions
vftr_demangle_cxx_LDFLAGS = $(AM_LDFLAGS) -liberty

dist_check_SCRIPTS += vftr_demangle_cxx.sh
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
