if WITH_MPI
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

AM_CFLAGS = -O3 -finstrument-functions
AM_CFLAGS += -I$(top_srcdir)/src/ -I$(top_srcdir)/src/omp/

AM_FCFLAGS = -O3 -finstrument-functions

if WITH_MPI
AM_CFLAGS += -D_MPI
AM_FCFLAGS += -D_MPI
endif 

if WITH_OMP
AM_CFLAGS += -D_OMP
AM_FCFLAGS += -D_OMP
endif

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

################
# System Tests #
################
check_PROGRAMS = 
dist_noinst_SCRIPTS = 
check_PROGRAMS += start_tool \
                  initialize

start_tool_SOURCES = $(srcdir)/start_tool.c
start_tool_CFLAGS  = $(AM_CFLAGS)
start_tool_LDFLAGS = $(AM_LDFLAGS)

initialize_SOURCES = $(srcdir)/initialize.c
initialize_CFLAGS  = $(AM_CFLAGS)
initialize_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += start_tool.sh \
                       initialize.sh

check_PROGRAMS += thread_num \
                  thread_level \
                  ancestor_thread_num

thread_num_SOURCES = $(srcdir)/thread_num.c
thread_num_CFLAGS  = $(AM_CFLAGS)
thread_num_LDFLAGS = $(AM_LDFLAGS)

thread_level_SOURCES = $(srcdir)/thread_level.c
thread_level_CFLAGS  = $(AM_CFLAGS)
thread_level_LDFLAGS = $(AM_LDFLAGS)

ancestor_thread_num_SOURCES = $(srcdir)/ancestor_thread_num.c
ancestor_thread_num_CFLAGS  = $(AM_CFLAGS)
ancestor_thread_num_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += thread_num.sh \
                       thread_level.sh \
                       ancestor_thread_num.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)