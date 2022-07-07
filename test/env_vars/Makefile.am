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

check_PROGRAMS += parse_and_print

parse_and_print_SOURCES = $(srcdir)/parse_and_print.c
parse_and_print_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
parse_and_print_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += defaults.sh \
		      advisor.sh

check_PROGRAMS += vftr_off

vftr_off_SOURCES = $(srcdir)/little_tasks.c
vftr_off_CFLAGS = $(AM_CFLAGS) -finstrument-functions
vftr_off_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += vftr_off.sh


TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
