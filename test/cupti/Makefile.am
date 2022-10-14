AM_CFLAGS = -I$(top_srcdir)/src \
            -I$(top_srcdir)/src/cupti

AM_CFLAGS += -D_CUPTI

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

check_PROGRAMS = gpu_info \
                 accumulate_cuptiprof \
                 print_cupti_table

gpu_info_SOURCES = $(srcdir)/gpu_info.c

accumulate_cuptiprof_SOURCES = $(srcdir)/accumulate_cuptiprof.c \
                               $(srcdir)/dummy_stacktree.c

print_cupti_table_SOURCES = $(srcdir)/print_cupti_table.c \
                            $(srcdir)/dummy_stacktree.c



dist_check_SCRIPTS = gpu_info.sh \
                     accumulate_cuptiprof.sh \
                     sort_cupti_none.sh \
                     sort_cupti_time.sh \
                     sort_cupti_memcpy.sh \
                     sort_cupti_cbid.sh \
                     sort_cupti_calls.sh
                         

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
