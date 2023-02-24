noinst_LTLIBRARIES = libvftr_accprof.la

libvftr_accprof_la_SOURCES =

libvftr_accprof_la_SOURCES += accprof_callbacks.c \
			      accprof_init_final.c \
			      accprof_events.c \
			      accprofiling.c \
			      collate_accprofiles.c \
                              collated_accprofiling.c \
			      accprof_logfile.c \
			      accprof_ranklogfile.c

libvftr_accprof_la_CFLAGS = -I${srcdir}/../ -I${srcdir}/../sorting -D_ACCPROF

libvftr_accprof_la_CFLAGS += -I${srcdir}/../hwprof -I${srcdir}/../../external/tinyexpr

if ENABLE_MPI
libvftr_accprof_la_CFLAGS += -D_MPI
libvftr_accprof_la_CFLAGS += -I${srcdir}/../mpi/utils
libvftr_accprof_la_CFLAGS += -I${srcdir}/../mpi/vftr_mpi
endif
