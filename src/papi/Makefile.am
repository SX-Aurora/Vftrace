noinst_LTLIBRARIES = libvftr_papi.la

libvftr_papi_la_SOURCES =

libvftr_papi_la_SOURCES += papi_init_final.c \
                           papi_calculator.c \
                           papiprofiling.c \
			   collate_papiprofiles.c \
                           papi_logfile.c

# -D_PAPI conflicts with PAPI headers.
libvftr_papi_la_CFLAGS = -I${srcdir}/../ -I${srcdir}/../sorting -I${srcdir}/../../external/tinyexpr -D_PAPI_AVAIL 

if HAS_LIBERTY
libvftr_papi_la_CFLAGS += -D_LIBERTY
endif

if SELF_PROFILING
libvftr_papi_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif

libvftr_papi_la_LIBADD = ../../external/libtinyexpr.la
