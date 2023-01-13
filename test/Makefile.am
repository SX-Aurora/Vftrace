SUBDIRS =
SUBDIRS += units
SUBDIRS += config
SUBDIRS += user_functions
if ENABLE_OMP
SUBDIRS += omp
endif

if ENABLE_MPI 
SUBDIRS += mpi
endif

if ENABLE_CUDAPROF
SUBDIRS += cuda
endif

if ENABLE_ACCPROF
SUBDIRS += openacc
endif

if ENABLE_PAPI
SUBDIRS += hwprof
endif
