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

AM_CFLAGS = -I$(top_srcdir)/src/ -I$(top_srcdir)/src/vftr_mpi

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if WITH_MPI
  AM_CFLAGS += -D_MPI
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

if ENABLE_OPENMP
if USES_INTEL_COMPILER
  AM_CFLAGS += -qopenmp
endif
if USES_GNU_COMPILER
  AM_CFLAGS += -fopenmp
endif
if USES_NEC_COMPILER
  AM_CFLAGS += -fopenmp
endif
endif

check_PROGRAMS = test_vftrace

test_vftrace_SOURCES = $(srcdir)/test_vftrace.c
test_vftrace_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
test_vftrace_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS = this_passes.sh \
		     this_fails.sh \
	 	     vftr_functions_test_3.sh \
	 	     vftr_functions_test_4.sh \
	 	     vftr_functions_test_5.sh \
		     vftr_stacks_test_1.sh \
		     vftr_browse_test_1.sh

if WITH_MPI
dist_check_SCRIPTS += vftr_stacks_test_2.sh
endif

if ENABLE_HWC
if ON_VECTOR_ENGINE
dist_check_SCRIPTS += vftr_sxhwc_test_1.sh
endif
endif

check_PROGRAMS += radixsort_uint64 \
		  sort_integer_ascending \
		  sort_integer_descending \
		  sort_double_ascending \
		  sort_double_descending \
		  sort_double_copy_ascending \
		  sort_double_copy_descending \
                  symbols_test_1 \
                  environment_1 \
                  environment_2 \
                  setup_1 \
                  setup_2 \
                  filewrite_1 \
                  filewrite_2 \
                  scenario_1 \
                  scenario_2 \
                  scenario_3 \
                  functions_1 \
                  functions_2

radixsort_uint64_SOURCES = $(srcdir)/radixsort_uint64.c
radixsort_uint64_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
radixsort_uint64_LDFLAGS = $(AM_LDFLAGS)

sort_integer_ascending_SOURCES = $(srcdir)/sort_integer_ascending.c
sort_integer_ascending_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
sort_integer_ascending_LDFLAGS = $(AM_LDFLAGS)

sort_integer_descending_SOURCES = $(srcdir)/sort_integer_descending.c
sort_integer_descending_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
sort_integer_descending_LDFLAGS = $(AM_LDFLAGS)

sort_double_ascending_SOURCES = $(srcdir)/sort_double_ascending.c
sort_double_ascending_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
sort_double_ascending_LDFLAGS = $(AM_LDFLAGS)

sort_double_descending_SOURCES = $(srcdir)/sort_double_descending.c
sort_double_descending_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
sort_double_descending_LDFLAGS = $(AM_LDFLAGS)

sort_double_copy_ascending_SOURCES = $(srcdir)/sort_double_copy_ascending.c
sort_double_copy_ascending_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
sort_double_copy_ascending_LDFLAGS = $(AM_LDFLAGS)

sort_double_copy_descending_SOURCES = $(srcdir)/sort_double_copy_descending.c
sort_double_copy_descending_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
sort_double_copy_descending_LDFLAGS = $(AM_LDFLAGS)

symbols_test_1_SOURCES = $(srcdir)/symbols_test_1.c
symbols_test_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_test_1_LDFLAGS = $(AM_LDFLAGS)

environment_1_SOURCES = $(srcdir)/environment_1.c
environment_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
environment_1_LDFLAGS = $(AM_LDFLAGS)

environment_2_SOURCES = $(srcdir)/environment_2.c
environment_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
environment_2_LDFLAGS = $(AM_LDFLAGS)

setup_1_SOURCES = $(srcdir)/setup_1.c
setup_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
setup_1_LDFLAGS = $(AM_LDFLAGS)

setup_2_SOURCES = $(srcdir)/setup_2.c
setup_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
setup_2_LDFLAGS = $(AM_LDFLAGS)

filewrite_1_SOURCES = $(srcdir)/filewrite_1.c
filewrite_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
filewrite_1_LDFLAGS = $(AM_LDFLAGS)

filewrite_2_SOURCES = $(srcdir)/filewrite_2.c
filewrite_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
filewrite_2_LDFLAGS = $(AM_LDFLAGS)

scenario_1_SOURCES = $(srcdir)/scenario_1.c
scenario_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
scenario_1_LDFLAGS = $(AM_LDFLAGS)

scenario_2_SOURCES = $(srcdir)/scenario_2.c
scenario_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
scenario_2_LDFLAGS = $(AM_LDFLAGS)

scenario_3_SOURCES = $(srcdir)/scenario_3.c
scenario_3_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
scenario_3_LDFLAGS = $(AM_LDFLAGS)

functions_1_SOURCES = $(srcdir)/functions_1.c
functions_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
functions_1_LDFLAGS = $(AM_LDFLAGS)

functions_2_SOURCES = $(srcdir)/functions_2.c
functions_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
functions_2_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS += radixsort_uint64.sh \
		      sort_integer_ascending.sh \
		      sort_integer_descending.sh \
		      sort_double_ascending.sh \
		      sort_double_descending.sh \
		      sort_double_copy_ascending.sh \
		      sort_double_copy_descending.sh \
                      symbols_test_1.sh \
                      environment_1.sh \
                      environment_2.sh \
                      setup_1.sh \
                      setup_2.sh \
                      filewrite_1.sh \
                      filewrite_2.sh \
                      scenario_1.sh \
                      scenario_2.sh \
                      scenario_3.sh \
                      functions_1.sh \
                      functions_2.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
XFAIL_TESTS = this_fails.sh \
	      scenario_3.sh
