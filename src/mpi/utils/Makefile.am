noinst_LTLIBRARIES = libvftr_mpi_utils.la

libvftr_mpi_utils_la_SOURCES = mpi_util_types.c

if WITH_MPI
libvftr_mpi_utils_la_SOURCES += mpi_logging.c \
				mpi_buf_addr_const.c \
				sync_time.c
if ENABLE_FORTRAN
libvftr_mpi_utils_la_SOURCES += mpi_logging_F.F90 \
				mpi_buf_addr_const_F.F90 \
				sync_time_F2c.c \
				sync_time_F.F90
if ENABLE_FORTRAN08
libvftr_mpi_utils_la_SOURCES += mpi_logging_F08.F90 \
                                mpi_buf_addr_const_F08.F90 \
                                sync_time_F082c.c \
                                sync_time_F08.F90
endif
endif

libvftr_mpi_utils_la_SOURCES += status_utils.c \
				request_utils.c


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
libvftr_mpi_utils_la_FCFLAGS +=
endif

libvftr_mpi_utils_la_CFLAGS += -I${srcdir}/../../
libvftr_mpi_utils_la_CFLAGS += -I${srcdir}/../vftr_mpi/