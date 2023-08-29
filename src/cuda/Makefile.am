noinst_LTLIBRARIES = libvftr_cuda.la

libvftr_cuda_la_SOURCES = 

libvftr_cuda_la_SOURCES += cupti_init_final.c \
                            cupti_vftr_callbacks.c \
                            cudaprofiling.c \
                            collated_cudaprofiling.c \
                            collate_cudaprofiles.c \
                            cuda_logfile.c \
                            cuda_ranklogfile.c \
                            cuda_utils.c

libvftr_cuda_la_CFLAGS = -I${srcdir}/../ \
                         -I${srcdir}/../sorting \
                         -I${srcdir}/../hwprof \
                         -I${srcdir}/../../external/tinyexpr \
                         -D_CUDA

if ENABLE_MPI
libvftr_cuda_la_CFLAGS += -D_MPI
libvftr_cuda_la_CFLAGS += -I${srcdir}/../mpi/utils
libvftr_cuda_la_CFLAGS += -I${srcdir}/../mpi/vftr_mpi
endif

if HAS_LIBERTY
libvftr_cuda_la_CFLAGS += -D_LIBERTY
endif

if SELF_PROFILING
libvftr_cuda_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif
