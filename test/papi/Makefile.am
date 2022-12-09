AM_CFLAGS = -I$(top_srcdir)/src \
            -I$(top_srcdir)/src/cuda

AM_CFLAGS += -D_PAPI_AVAIL

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

check_PROGRAMS = papi_init_test \
                 calculator \
                 print_config

papi_init_test_SOURCES = $(srcdir)/papi_init_test.c
papi_init_test_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
papi_init_test_LDFLAGS = $(AM_LDFLAGS)

calculator_SOURCES = $(srcdir)/calculator.c
calculator_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
calculator_LDFLAGS = $(AM_LDFLAGS)

print_config_SOURCES = $(srcdir)/little_tasks.c
print_config_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
print_config_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS = papi_init.sh \
                     calculator.sh \
                     print_config.sh

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
