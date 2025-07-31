noinst_LTLIBRARIES = libvftr_hwprof.la

libvftr_hwprof_la_SOURCES =

libvftr_hwprof_la_SOURCES += hwprof_init_final.c \
                           calculator.c \
                           hwprofiling.c \
			   collate_hwprofiles.c \
                           hwprof_logfile.c \
		  	   hwprof_ranklogfile.c \
                           hwprof_dummy.c

if ENABLE_PAPI
libvftr_hwprof_la_SOURCES += hwprof_papi.c
endif

if ON_VECTOR_ENGINE
libvftr_hwprof_la_SOURCES += hwprof_ve.c
endif

libvftr_hwprof_la_SOURCES += hwprof_likwid.c

libvftr_hwprof_la_CFLAGS = -I${srcdir}/../ -I${srcdir}/../sorting -I${srcdir}/../../external/tinyexpr 
if ENABLE_PAPI
# -D_PAPI conflicts with PAPI headers.
#  The compiler flag for PAPI is therefore called PAPI_AVAIL.
libvftr_hwprof_la_CFLAGS += -D_PAPI_AVAIL
endif

if ON_VECTOR_ENGINE
libvftr_hwprof_la_CFLAGS += -D_ON_VE
endif

if ENABLE_MPI
libvftr_hwprof_la_CFLAGS += -D_MPI -I${srcdir}/../mpi/utils
endif

if HAS_LIBERTY
libvftr_hwprof_la_CFLAGS += -D_LIBERTY
endif

if SELF_PROFILING
libvftr_hwprof_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif

if ENABLE_CUDAPROF
libvftr_hwprof_la_CFLAGS += -D_CUDA -I${srcdir}/../cuda
endif

if ENABLE_ACCPROF
libvftr_hwprof_la_CFLAGS += -D_ACCPROF -I${srcdir}/../accprof
endif

libvftr_hwprof_la_LIBADD = ../../external/libtinyexpr.la
