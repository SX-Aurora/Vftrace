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

AM_CFLAGS = -I$(top_srcdir)/src/ -I$(top_srcdir)/src/mpi/utils/

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

check_PROGRAMS = radixsort_uint64 \
		 sort_integer_ascending \
		 sort_integer_descending \
		 sort_double_ascending \
		 sort_double_descending \
		 sort_double_copy_ascending \
		 sort_double_copy_descending \
		 symbols_1 \
		 environment_1 \
		 environment_2 \
		 filewrite_1 \
		 stacktree 
                 #scenario_1 \
                 #scenario_2 \
                 #scenario_3 \
		 #tables \
                 #browse_1 \
		 #no_instrument_attribute

if WITH_MPI
check_PROGRAMS += stacks_2 \
		  stacks_3 \
		  no_mpi_init_C
if ENABLE_FORTRAN
check_PROGRAMS += no_mpi_init_F
endif
endif

if ENABLE_HWC
if ON_VECTOR_ENGINE
check_PROGRAMS += sxhwc_1
endif
endif

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

symbols_1_SOURCES = $(srcdir)/symbols_1.c \
		    $(srcdir)/protofuncts.c
symbols_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_1_LDFLAGS = $(AM_LDFLAGS)

environment_1_SOURCES = $(srcdir)/environment_1.c
environment_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
environment_1_LDFLAGS = $(AM_LDFLAGS)

environment_2_SOURCES = $(srcdir)/environment_2.c
environment_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
environment_2_LDFLAGS = $(AM_LDFLAGS)

filewrite_1_SOURCES = $(srcdir)/filewrite_1.c
filewrite_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
filewrite_1_LDFLAGS = $(AM_LDFLAGS)

stacktree_SOURCES = $(srcdir)/stacktree.c
stacktree_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stacktree_LDFLAGS = $(AM_LDFLAGS)

#scenario_1_SOURCES = $(srcdir)/scenario_1.c
#scenario_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
#scenario_1_LDFLAGS = $(AM_LDFLAGS)
#
#scenario_2_SOURCES = $(srcdir)/scenario_2.c
#scenario_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
#scenario_2_LDFLAGS = $(AM_LDFLAGS)
#
#scenario_3_SOURCES = $(srcdir)/scenario_3.c
#scenario_3_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/external/tinyexpr
#scenario_3_LDFLAGS = $(AM_LDFLAGS)
#
#tables_SOURCES = $(srcdir)/tables.c
#tables_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#tables_LDFLAGS = $(AM_LDFLAGS)
#
#stacks_1_SOURCES = $(srcdir)/stacks_1.c
#stacks_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#stacks_1_LDFLAGS = $(AM_LDFLAGS)
#
#if WITH_MPI
#stacks_2_SOURCES = $(srcdir)/stacks_2.c
#stacks_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#stacks_2_LDFLAGS = $(AM_LDFLAGS)
#
#stacks_3_SOURCES = $(srcdir)/stacks_3.c
#stacks_3_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#stacks_3_LDFLAGS = $(AM_LDFLAGS)
#
#no_mpi_init_C_SOURCES = $(srcdir)/no_mpi_init.c
#no_mpi_init_C_CFLAGS = $(AM_CFLAGS) -finstrument-functions
#no_mpi_init_C_LDFLAGS = $(AM_LDFLAGS)
#
#if ENABLE_FORTRAN
#no_mpi_init_F_SOURCES = $(srcdir)/no_mpi_init.F90
#no_mpi_init_F_FCFLAGS = $(AM_FFLAGS) -finstrument-functions
#no_mpi_init_F_LDFLAGS = $(AM_LDFLAGS)
#endif
#endif
#
#browse_1_SOURCES = $(srcdir)/browse_1.c
#browse_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#browse_1_LDFLAGS = $(AM_LDFLAGS)
#
#no_instrument_attribute_SOURCES = $(srcdir)/no_instrument_attribute.c
#no_instrument_attribute_CFLAGS = $(AM_CFLAGS) -finstrument-functions
#no_instrument_attribute_LDFLAGS = $(AM_LDFLAGS)
#
#if ENABLE_HWC
#if ON_VECTOR_ENGINE
#sxhwc_1_SOURCES = $(srcdir)/sxhwc_1.c
#sxhwc_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
#sxhwc_1_LDFLAGS = $(AM_LDFLAGS)
#endif
#endif

dist_check_SCRIPTS = radixsort_uint64.sh \
		      sort_integer_ascending.sh \
		      sort_integer_descending.sh \
		      sort_double_ascending.sh \
		      sort_double_descending.sh \
		      sort_double_copy_ascending.sh \
		      sort_double_copy_descending.sh \
                      symbols_1.sh \
                      environment_1.sh \
                      environment_2.sh \
                      filewrite_1.sh \
                      stacktree.sh \
                      scenario_1.sh \
                      scenario_2.sh \
                      scenario_3.sh \
                      stacks_1.sh \
                      browse_1.sh \
		      no_instrument_attribute.sh

if ENABLE_HWC
if ON_VECTOR_ENGINE
dist_check_SCRIPTS += sxhwc_1.sh
endif
endif

if WITH_MPI
dist_check_SCRIPTS += stacks_2.sh \
		      stacks_3.sh \
		      no_mpi_init_C.sh
if ENABLE_FORTRAN
dist_check_SCRIPTS += no_mpi_init_F.sh
endif
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
XFAIL_TESTS = scenario_3.sh

if ENABLE_HWC
if ON_VECTOR_ENGINE
XFAIL_TESTS += sxhwc_1.sh
endif
endif
