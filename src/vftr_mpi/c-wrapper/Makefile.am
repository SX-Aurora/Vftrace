noinst_LTLIBRARIES = libvftr_cmpiwrap.la

libvftr_cmpiwrap_la_SOURCES = vftr_mpi_environment_cwrap.c \
			      vftr_mpi_point2point_cwrap.c \
			      vftr_mpi_collective_cwrap.c \
			      vftr_mpi_onesided_cwrap.c \
			      vftr_mpi_testwait_cwrap.c

libvftr_cmpiwrap_la_CFLAGS = -I${srcdir}/../
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../../
libvftr_cmpiwrap_la_CFLAGS += -finstrument-functions
if MPI_PROFILING
libvftr_cmpiwrap_la_CFLAGS += -D_MPI
endif
