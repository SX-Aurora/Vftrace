AM_CFLAGS = -I$(top_srcdir)/src \
            -I$(top_srcdir)/src/cuda

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

check_PROGRAMS = gpu_info \
                 accumulate_cudaprof \
                 print_cuda_table \
                 cbid_names \
                 callback_subscribe

gpu_info_SOURCES = $(srcdir)/gpu_info.c

accumulate_cudaprof_SOURCES = $(srcdir)/accumulate_cudaprof.c \
                               $(srcdir)/dummy_stacktree.c

print_cuda_table_SOURCES = $(srcdir)/print_cuda_table.c \
                            $(srcdir)/dummy_stacktree.c

cbid_names_SOURCES = $(srcdir)/cbid_names.c

callback_subscribe_SOURCES = $(srcdir)/callback_subscribe.c

dist_check_SCRIPTS = gpu_info.sh \
                     accumulate_cudaprof.sh \
                     sort_cuda_none.sh \
                     sort_cuda_time.sh \
                     sort_cuda_memcpy.sh \
                     sort_cuda_cbid.sh \
                     sort_cuda_calls.sh \
                     cbid_names.sh \
                     callback_subscribe.sh
                         

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
