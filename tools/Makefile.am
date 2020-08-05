bin_PROGRAMS = tracedump

tracedump_SOURCES = tracedump.c

tracedump_CFLAGS = -I${srcdir}/../src/
tracedump_CFLAGS += -I${srcdir}/../src/vftr_mpi/
tracedump_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

if USES_NEC_MPI
  tracedump_LDFLAGS += -mpiprof
endif
