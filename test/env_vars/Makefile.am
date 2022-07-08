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

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if ENABLE_MPI
  AM_CFLAGS += -D_MPI
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

if ENABLE_OMP
  AM_CFLAGS += -D_OMP
endif

check_PROGRAMS = 
dist_check_SCRIPTS =

check_PROGRAMS += defaults \
		  advisor

defaults_SOURCES = $(srcdir)/defaults.c
defaults_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
defaults_LDFLAGS = $(AM_LDFLAGS)

advisor_SOURCES = $(srcdir)/advisor.c
advisor_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
advisor_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += defaults.sh \
		      advisor.sh

check_PROGRAMS += vftr_off \
		  vftr_sampling \
		  vftr_regions_precise \
		  vftr_out_directory \
		  vftr_logfile_basename

vftr_off_SOURCES = $(srcdir)/little_tasks.c
vftr_off_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_off_LDFLAGS = $(AM_LDFLAGS)

vftr_sampling_SOURCES = $(srcdir)/little_tasks.c
vftr_sampling_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_sampling_LDFLAGS = $(AM_LDFLAGS)

vftr_regions_precise_SOURCES = $(srcdir)/little_tasks.c
vftr_regions_precise_CFLAGS = $(AM_CFLAGS) -finstrument-functions -D_REGIONS
vftr_regions_precise_LDFLAGS = $(AM_LDFLAGS)

vftr_out_directory_SOURCES = $(srcdir)/little_tasks.c
vftr_out_directory_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_out_directory_LDFLAGS = $(AM_LDFLAGS)

vftr_logfile_basename_SOURCES = $(srcdir)/little_tasks.c
vftr_logfile_basename_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_logfile_basename_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += vftr_off.sh \
		      vftr_sampling.sh \
		      vftr_regions_precise.sh \
		      vftr_out_directory.sh \
		      vftr_logfile_basename.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
