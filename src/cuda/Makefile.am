noinst_LTLIBRARIES = libvftr_cuda.la

libvftr_cuda_la_SOURCES = 

libvftr_cuda_la_SOURCES += cupti_init_final.c \
                            cupti_vftr_callbacks.c \
                            cudaprofiling.c \
                            collate_cudaprofiles.c \
                            cuda_logfile.c \
                            cuda_ranklogfile.c \
                            cuda_utils.c

libvftr_cuda_la_CFLAGS =

if SELF_PROFILING
libvftr_cuda_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif
