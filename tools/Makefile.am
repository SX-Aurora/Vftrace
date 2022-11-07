SUBDIRS =
if SELF_PROFILING
SUBDIRS += self_profiling_analyzer
endif
SUBDIRS += config_tools

bin_PROGRAMS = vftrace_vfd_dump \
	       vftrace_diff_profiles

vftrace_vfd_dump_SOURCES = vftrace_vfd_dump.c \
			   vfd_dump_cmd_options.c \
			   vfd_utils.c
vftrace_vfd_dump_CFLAGS = -I${srcdir}/../src/
vftrace_vfd_dump_CFLAGS += -I${srcdir}/../src/mpi/utils/
vftrace_vfd_dump_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

vftrace_diff_profiles_SOURCES = vftrace_diff_profiles.c
vftrace_diff_profiles_CFLAGS = -I${srcdir}/../src
vftrace_diff_profiles_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

if USES_NEC_MPI
  vftrace_vfd_dump_LDFLAGS += -mpiprof
  vftrace_diff_profiles_LDFLAGS += -mpiprof
endif
