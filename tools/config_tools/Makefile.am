bin_PROGRAMS = vftrace_generate_default_config \
	       vftrace_check_config

vftrace_generate_default_config_SOURCES = vftrace_generate_default_config.c
vftrace_generate_default_config_CFLAGS = -I${srcdir}/../../src/
vftrace_generate_default_config_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace

vftrace_check_config_SOURCES = vftrace_check_config.c
vftrace_check_config_CFLAGS = -I${srcdir}/../../src
vftrace_check_config_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace

if USES_NEC_MPI
  vftrace_generate_default_config_LDFLAGS += -mpiprof
  vftrace_check_config_LDFLAGS += -mpiprof
endif
