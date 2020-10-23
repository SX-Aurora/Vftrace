bin_PROGRAMS = tracedump \
	       show_function_stacks \
	       mpi_crosscheck \
	       mpi_stack_crosscheck \
	       filter_stacks

tracedump_SOURCES = tracedump.c vftr_vfd_utils.c
show_function_stacks_SOURCES = show_function_stacks.c vftr_vfd_utils.c
mpi_crosscheck_SOURCES = mpi_crosscheck.c
mpi_stack_crosscheck_SOURCES = mpi_stack_crosscheck.c
filter_stacks_SOURCES = filter_stacks.c

tracedump_CFLAGS = -I${srcdir}/../src/
tracedump_CFLAGS += -I${srcdir}/../src/vftr_mpi/
tracedump_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

show_function_stacks_CFLAGS = -I${srcdir}/../src/
show_function_stacks_CFLAGS += -I${srcdir}/../src/vftr_mpi/
show_function_stacks_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

filter_stacks_CFLAGS = -I${srcdir}/../src/

if USES_NEC_MPI
  tracedump_LDFLAGS += -mpiprof
  show_function_stacks_LDFLAGS += -mpiprof
endif
