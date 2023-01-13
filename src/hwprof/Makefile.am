noinst_LTLIBRARIES = libvftr_hwprof.la

libvftr_hwprof_la_SOURCES =

libvftr_hwprof_la_SOURCES += papi_init_final.c \
                           hwprof_init_final.c \
                           calculator.c \
                           hwprofiling.c \
			   collate_hwprofiles.c \
                           hwprof_logfile.c \
		  	   hwprof_ranklogfile.c

# -D_PAPI conflicts with PAPI headers.
libvftr_hwprof_la_CFLAGS = -I${srcdir}/../ -I${srcdir}/../sorting -I${srcdir}/../../external/tinyexpr -D_HWPROF 
#  The compiler flag for PAPI is therefore called PAPI_AVAIL.
if ENABLE_PAPI
libvftr_hwprof_la_CFLAGS += -D_PAPI_AVAIL
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

libvftr_hwprof_la_LIBADD = ../../external/libtinyexpr.la
