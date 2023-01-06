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

AM_CFLAGS = -I$(top_srcdir)/src \
            -I$(top_srcdir)/src/papi \
            -I$(top_srcdir)/external/tinyexpr

AM_CFLAGS += -D_PAPI_AVAIL

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

if ENABLE_MPI
  AM_CFLAGS += -D_MPI -I$(top_srcdir)/src/mpi/utils
endif

check_PROGRAMS = papi_init_test \
                 calculator \
                 print_config \
                 papi_region1 \
                 papi_region2

if ENABLE_MPI
check_PROGRAMS += collate_papiprofiles_parallel
endif

papi_init_test_SOURCES = $(srcdir)/papi_init_test.c
papi_init_test_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
papi_init_test_LDFLAGS = $(AM_LDFLAGS)

calculator_SOURCES = $(srcdir)/calculator.c
calculator_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
calculator_LDFLAGS = $(AM_LDFLAGS)

print_config_SOURCES = $(srcdir)/little_tasks.c
print_config_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
print_config_LDFLAGS = $(AM_LDFLAGS)

papi_region1_SOURCES = $(srcdir)/papi_region1.c
papi_region1_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
papi_region1_LDFLAGS = $(AM_LDFLAGS)

papi_region2_SOURCES = $(srcdir)/papi_region2.c
papi_region2_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
papi_region2_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collate_papiprofiles_parallel_SOURCES = $(srcdir)/collate_papiprofiles_parallel.c
collate_papiprofiles_parallel_CFLAGS = $(AM_CFLAGS)
collate_papiprofiles_parallel_LDFLAGS = $(AM_LDFLAGS)
endif

dist_check_SCRIPTS = papi_init.sh \
                     calculator.sh \
                     print_config.sh \
                     papi_region1.sh \
                     papi_region2.sh

if ENABLE_MPI
dist_check_SCRIPTS += collate_papiprofiles_parallel.sh
endif

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
