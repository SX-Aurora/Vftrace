noinst_LTLIBRARIES = libvftr_logfiles.la

libvftr_logfiles_la_SOURCES = logfile_common.c \
                              logfile.c \
                              logfile_summary.c \
                              logfile_prof_table.c \
                              logfile_stacklist.c \
                              ranklogfile.c \
                              ranklogfile_summary.c \
                              ranklogfile_prof_table.c 

libvftr_logfiles_la_CFLAGS = -I${srcdir}/.. -I${srcdir}/../sorting
libvftr_logfiles_la_CPPFLAGS = 
libvftr_logfiles_la_CFLAGS += -I${srcdir}/../mpi/utils/ \
			      -I${srcdir}/../omp/ \
                              -I${srcdir}/../cuda/ \
			      -I${srcdir}/../accprof \
		              -I${srcdir}/../hwprof \
			      -I${srcdir}/../sorting \
                              -I${srcdir}/../../external/tinyexpr

if ENABLE_MPI
libvftr_logfiles_la_CFLAGS += -D_MPI
endif

if ENABLE_OMP
libvftr_logfiles_la_CFLAGS += -D_OMP
endif

if ON_VECTOR_ENGINE
libvftr_logfiles_la_CFLAGS += -D_ON_VE
endif

if ENABLE_CUDAPROF
libvftr_logfiles_la_CFLAGS += -D_CUDA
endif

if ENABLE_ACCPROF
libvftr_logfiles_la_CFLAGS += -D_ACCPROF
endif

if ENABLE_PAPI
libvftr_logfiles_la_CFLAGS += -D_PAPI_AVAIL
endif

if SELF_PROFILING
libvftr_logfiles_la_CPPFLAGS += -D_SELF_PROFILE_VFTRACE
libvftr_logfiles_la_SOURCES += self_profile.c
endif

