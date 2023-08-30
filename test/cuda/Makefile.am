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
            -I${top_srcdir}/src/hwprof \
            -I$(top_srcdir)/src/cuda

AM_CFLAGS += -D_CUDA

if ENABLE_MPI
AM_CFLAGS += -D_MPI -I$(top_srcdir)/src/mpi/utils
endif

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace

check_PROGRAMS = gpu_info \
                 accumulate_cudaprof \
                 print_cuda_table \
                 cbid_names \
                 callback_subscribe

if ENABLE_MPI
check_PROGRAMS += collate_cudaprofiles_parallel
endif

gpu_info_SOURCES = $(srcdir)/gpu_info.c

accumulate_cudaprof_SOURCES = $(srcdir)/accumulate_cudaprof.c \
                               $(srcdir)/dummy_stacktree.c

print_cuda_table_SOURCES = $(srcdir)/print_cuda_table.c \
                            $(srcdir)/dummy_stacktree.c

cbid_names_SOURCES = $(srcdir)/cbid_names.c

callback_subscribe_SOURCES = $(srcdir)/callback_subscribe.c

if ENABLE_MPI
collate_cudaprofiles_parallel_SOURCES = $(srcdir)/collate_cudaprofiles_parallel.c \
                                        $(srcdir)/dummy_stacktree.c
collate_cudaprofiles_parallel_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collate_cudaprofiles_parallel_LDFLAGS = $(AM_LDFLAGS)
endif

dist_check_SCRIPTS = gpu_info.sh \
                     accumulate_cudaprof.sh \
                     sort_cuda_none.sh \
                     sort_cuda_time.sh \
                     sort_cuda_memcpy.sh \
                     sort_cuda_cbid.sh \
                     sort_cuda_calls.sh \
                     cbid_names.sh \
                     callback_subscribe.sh
                         
if ENABLE_MPI
dist_check_SCRIPTS += collate_cudaprofiles_parallel.sh
endif

TEST_EXTENSIONS = .sh
TESTS = $(dist_check_SCRIPTS)
