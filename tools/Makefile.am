bin_PROGRAMS = vftrace_vfd_dump \
	       vftrace_show_function_stacks \
	       vftrace_mpi_table_crosscheck \
	       vftrace_mpi_stack_crosscheck \
	       vftrace_filter_stacks \
	       vftrace_diff_profiles

vftrace_mpi_table_crosscheck_SOURCES = vftrace_mpi_table_crosscheck.c
vftrace_mpi_stack_crosscheck_SOURCES = vftrace_mpi_stack_crosscheck.c

vftrace_vfd_dump_SOURCES = vftrace_vfd_dump.c vftr_vfd_utils.c
vftrace_vfd_dump_CFLAGS = -I${srcdir}/../src/
vftrace_vfd_dump_CFLAGS += -I${srcdir}/../src/vftr_mpi/
vftrace_vfd_dump_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

vftrace_show_function_stacks_SOURCES = vftrace_show_function_stacks.c vftr_vfd_utils.c
vftrace_show_function_stacks_CFLAGS = -I${srcdir}/../src/
vftrace_show_function_stacks_CFLAGS += -I${srcdir}/../src/vftr_mpi/
vftrace_show_function_stacks_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

vftrace_filter_stacks_SOURCES = vftrace_filter_stacks.c
vftrace_filter_stacks_CFLAGS = -I${srcdir}/../src/
vftrace_filter_stacks_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

vftrace_diff_profiles_SOURCES = vftrace_diff_profiles.c
vftrace_diff_profiles_CFLAGS = -I${srcdir}/../src
vftrace_diff_profiles_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

if USES_NEC_MPI
  vftrace_vfd_dump_LDFLAGS += -mpiprof
  vftrace_show_function_stacks_LDFLAGS += -mpiprof
  vftrace_filter_stacks_LDFLAGS += -mpiprof
  vftrace_diff_profiles_LDFLAGS += -mpiprof
endif

if ENABLE_OPENMP

if USES_INTEL_COMPILER
vftrace_vfd_dump_CFLAGS += -qopenmp
vftrace_show_function_stacks_CFLAGS += -qopenmp
vftrace_filter_stacks_CFLAGS += -qopenmp
vftrace_diff_profiles_CFLAGS += -qopenmp
endif

if USES_GNU_COMPILER
vftrace_vfd_dump_CFLAGS += -fopenmp
vftrace_show_function_stacks_CFLAGS += -fopenmp
vftrace_filter_stacks_CFLAGS += -fopenmp
vftrace_diff_profiles_CFLAGS += -fopenmp
endif

if USES_NEC_COMPILER
vftrace_vfd_dump_CFLAGS += -fopenmp
vftrace_show_function_stacks_CFLAGS += -fopenmp
vftrace_filter_stacks_CFLAGS += -fopenmp
vftrace_diff_profiles_CFLAGS += -fopenmp
endif

endif

