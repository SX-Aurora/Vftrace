AM_CFLAGS = -I$(top_srcdir)/src \
            -I$(top_srcdir)/src/cuda

AM_CFLAGS += -D_PAPI_AVAIL

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

check_PROGRAMS = print_config

print_config_SOURCES = $(srcdir)/little_tasks.c
print_config_CFLAGS = $(AM_CFLAGS) $(FINSTRUMENT_FLAG)
print_config_LDFLAGS = $(AM_LDFLAGS)

dist_check_SCRIPTS = print_config.sh

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
