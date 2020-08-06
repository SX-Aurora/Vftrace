ACLOCAL_AMFLAGS="-I m4"

noinst_LTLIBRARIES = libvftr_mpi.la

SUBDIRS = c-wrapper \
	  f-wrapper \
	  f08-wrapper

libvftr_mpi_la_SOURCES = vftr_async_messages.c \
			 vftr_mpi_buf_addr_const.c \
			 vftr_mpi_buf_addr_const_F.F90 \
			 vftr_mpi_collective.c \
			 vftr_mpi_collective_F.c \
			 vftr_mpi_environment.c \
			 vftr_mpi_environment_F08.c \
			 vftr_mpi_environment_F.c \
			 vftr_mpi_onesided.c \
			 vftr_mpi_onesided_F.c \
			 vftr_mpi_point2point.c \
			 vftr_mpi_point2point_F.c \
			 vftr_mpi_testwait.c \
			 vftr_mpi_testwait_F.c \
			 vftr_mpi_utils.c \
			 vftr_sync_messages.c

libvftr_mpi_la_CFLAGS =
libvftr_mpi_la_FCFLAGS =

libvftr_mpi_la_CFLAGS += -I${srcdir}/../

if MPI_PROFILING
libvftr_mpi_la_CFLAGS += -D_MPI
libvftr_mpi_la_FCFLAGS += -D_MPI
endif

clean-local:
	rm -f *.mod
