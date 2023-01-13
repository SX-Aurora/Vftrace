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
            -I$(top_srcdir)/src/hwprof \
            -I$(top_srcdir)/external/tinyexpr

AM_CFLAGS += -D_HWPROF

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

if ENABLE_MPI
  AM_CFLAGS += -D_MPI -I$(top_srcdir)/src/mpi/utils
endif

check_PROGRAMS = papi_init_test \
                 calculator \
                 print_config \
                 print_config_default_scenario \
                 hwprof_region1 \
                 hwprof_region2 \
                 hwprof_region3

if ENABLE_MPI
check_PROGRAMS += collate_hwprofiles_parallel
check_PROGRAMS += collate_hwprofiles_parallel_2
endif

papi_init_test_SOURCES = $(srcdir)/papi_init_test.c
papi_init_test_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
papi_init_test_LDFLAGS = $(AM_LDFLAGS)

calculator_SOURCES = $(srcdir)/calculator.c
calculator_CFLAGS = $(AM_CFLAGS)
calculator_LDFLAGS = $(AM_LDFLAGS)

print_config_SOURCES = $(srcdir)/print_config.c
print_config_CFLAGS = $(AM_CFLAGS)
print_config_LDFLAGS = $(AM_LDFLAGS)

print_config_default_scenario_SOURCES = $(srcdir)/print_config.c
print_config_default_scenario_CFLAGS = $(AM_CFLAGS)
print_config_default_scenario_LDFLAGS = $(AM_LDFLAGS)

hwprof_region1_SOURCES = $(srcdir)/hwprof_region1.c
hwprof_region1_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
hwprof_region1_LDFLAGS = $(AM_LDFLAGS)

hwprof_region2_SOURCES = $(srcdir)/hwprof_region2.c
hwprof_region2_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
hwprof_region2_LDFLAGS = $(AM_LDFLAGS)

hwprof_region3_SOURCES = $(srcdir)/hwprof_region2.c
hwprof_region3_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
hwprof_region3_LDFLAGS = $(AM_LDFLAGS)


if ENABLE_MPI
collate_hwprofiles_parallel_SOURCES = $(srcdir)/collate_hwprofiles_parallel.c
collate_hwprofiles_parallel_CFLAGS = $(AM_CFLAGS)
collate_hwprofiles_parallel_LDFLAGS = $(AM_LDFLAGS)

collate_hwprofiles_parallel_2_SOURCES = $(srcdir)/collate_hwprofiles_parallel_2.c
collate_hwprofiles_parallel_2_CFLAGS = $(AM_CFLAGS)
collate_hwprofiles_parallel_2_LDFLAGS = $(AM_LDFLAGS)

endif

dist_check_SCRIPTS = papi_init.sh \
                     calculator.sh \
                     print_config.sh \
                     print_config_default_scenario.sh \
                     hwprof_region1.sh \
                     hwprof_region2.sh \
                     hwprof_region3.sh

if ENABLE_MPI
dist_check_SCRIPTS += collate_hwprofiles_parallel.sh
dist_check_SCRIPTS += collate_hwprofiles_parallel_2.sh
endif

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
