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

AM_CFLAGS = -I$(top_srcdir)/src/ -I$(top_srcdir)/src/mpi/vftr_mpi

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if WITH_MPI
  AM_CFLAGS += -D_MPI
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

check_PROGRAMS = test_vftrace

test_vftrace_SOURCES = $(srcdir)/test_vftrace.c
test_vftrace_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
test_vftrace_LDFLAGS = $(AM_LDFLAGS)


dist_check_SCRIPTS = this_passes.sh \
		     this_fails.sh \
		     vftr_symbols_test_1.sh \
		     vftr_environment_test_1.sh \
		     vftr_environment_test_2.sh \
		     vftr_setup_test_1.sh \
		     vftr_setup_test_2.sh \
		     vftr_filewrite_test_1.sh \
		     vftr_filewrite_test_2.sh \
		     vftr_scenario_test_1.sh \
		     vftr_scenario_test_2.sh \
		     vftr_scenario_test_3.sh \
	 	     vftr_functions_test_1.sh \
	 	     vftr_functions_test_2.sh \
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
		  sort_double_copy_descending

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

dist_check_SCRIPTS += radixsort_uint64.sh \
		      sort_integer_ascending.sh \
		      sort_integer_descending.sh \
		      sort_double_ascending.sh \
		      sort_double_descending.sh \
		      sort_double_copy_ascending.sh \
		      sort_double_copy_descending.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
XFAIL_TESTS = this_fails.sh \
	      vftr_scenario_test_3.sh
