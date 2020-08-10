noinst_PROGRAMS = test_vftrace

test_vftrace_SOURCES = $(srcdir)/test_vftrace.c
test_vftrace_CFLAGS = -I$(top_srcdir)/src/ -I$(top_srcdir)/src/vftr_mpi
test_vftrace_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if USES_NEC_MPI
  test_vftrace_LDFLAGS += -mpiprof
endif

dist_noinst_DATA = $(srcdir)/ref_input/* \
		   $(srcdir)/ref_output/*

dist_noinst_SCRIPTS = this_passes.sh \
		      this_fails.sh \
		      vftr_symbols_test_1.sh \
		      vftr_environment_test_1.sh \
		      vftr_setup_test_1.sh \
		      vftr_setup_test_2.sh 
# deactivated for now. 
# Needs investigation
#		      vftr_filewrite_test_1.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
XFAIL_TESTS = this_fails.sh
