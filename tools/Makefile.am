bin_PROGRAMS = tracedump mpi_stacks

tracedump_SOURCES = tracedump.c
mpi_stacks_SOURCES = mpi_stacks.c

tracedump_CFLAGS = -I${srcdir}/../src/
tracedump_CFLAGS += -I${srcdir}/../src/vftr_mpi/
tracedump_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

mpi_stacks_CFLAGS = -I${srcdir}/../src/
mpi_stacks_CFLAGS += -I${srcdir}/../src/vftr_mpi/
mpi_stacks_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

if USES_NEC_MPI
  tracedump_LDFLAGS += -mpiprof
  mpi_stacks_LDFLAGS += -mpiprof
endif
