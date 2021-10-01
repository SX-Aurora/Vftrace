SUBDIRS = utils

if WITH_MPI
SUBDIRS +=vftr_mpi

SUBDIRS += c2vftr \
	   c-wrapper
if ENABLE_FORTRAN
SUBDIRS += f2vftr \
	   f-wrapper
if ENABLE_FORTRAN08
SUBDIRS += f082vftr \
	   f08-wrapper
endif
endif
endif
