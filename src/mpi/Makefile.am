SUBDIRS = utils

if MPI_TRACE_MINIMAL
SUBDIRS +=vftr_mpi

SUBDIRS += c2vftr \
	   c-wrapper
if ENABLE_FORTRAN
SUBDIRS += f2vftr \
	   f-wrapper
if MPI_TRACE_F08
SUBDIRS += f082vftr \
	   f08-wrapper
endif
endif
endif
