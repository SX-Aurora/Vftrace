noinst_LTLIBRARIES = libvftr_mpi_utils.la

libvftr_mpi_utils_la_SOURCES = mpi_util_types.c

if WITH_MPI
libvftr_mpi_utils_la_SOURCES += mpi_logging.c \
                                mpi_logging_F.F90

libvftr_mpi_utils_la_SOURCES += sync_time.c

libvftr_mpi_utils_la_SOURCES += rank_translate.c
endif

libvftr_mpi_utils_la_CFLAGS =
libvftr_mpi_utils_la_FCFLAGS =

libvftr_mpi_utils_la_CFLAGS += -I${srcdir}/../../
libvftr_mpi_utils_la_CFLAGS += -I${srcdir}/../vftr_mpi/
