noinst_LTLIBRARIES = libvftr_mpi_utils.la

libvftr_mpi_utils_la_SOURCES = mpi_util_types.c

if WITH_MPI
libvftr_mpi_utils_la_SOURCES += mpi_logging.c \
                                mpi_logging_F.F90

libvftr_mpi_utils_la_SOURCES += mpi_buf_addr_const.c \
				status_utils.c \
				request_utils.c

libvftr_mpi_utils_la_SOURCES += sync_time.c

libvftr_mpi_utils_la_SOURCES += rank_translate.c

libvftr_mpi_utils_la_SOURCES += sync_messages.c

libvftr_mpi_utils_la_SOURCES += requests.c \
				p2p_requests.c \
				persistent_requests.c \
				collective_requests.c \
				onesided_requests.c \
				clear_mpi_requests.c

endif

libvftr_mpi_utils_la_CFLAGS =
libvftr_mpi_utils_la_FCFLAGS =
if WITH_MPI
libvftr_mpi_utils_la_CFLAGS += -D_MPI
endif

libvftr_mpi_utils_la_CFLAGS += -I${srcdir}/../../
libvftr_mpi_utils_la_CFLAGS += -I${srcdir}/../vftr_mpi/
