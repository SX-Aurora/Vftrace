if MPI_PROFILING
  AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necmpi.sh;
#   AM_TESTS_ENVIRONMENT=export HAS_MPI="YES"; \
#                        export MPI_EXEC="mpirun"; \
# 		       export NP="-np";  
else
  AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_nompi.sh;
endif

AM_CFLAGS = -I$(top_srcdir)/src/ -I$(top_srcdir)/src/vftr_mpi

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if VEPERF_AVAIL
AM_LDFLAGS += -L$(VEPERF_HOME)/lib -lveperf
AM_CFLAGS += -DHAS_VEPERF -I$(VEPERF_HOME)/inclide
endif

if MPI_PROFILING
  AM_CFLAGS += -D_MPI
endif

if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

check_PROGRAMS = test_vftrace

test_vftrace_SOURCES = $(srcdir)/test_vftrace.c
test_vftrace_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
test_vftrace_LDFLAGS = $(AM_LDFLAGS)


dist_check_SCRIPTS = this_passes.sh \
		      this_fails.sh \
		      vftr_symbols_test_1.sh \
		      vftr_environment_test_1.sh \
		      vftr_setup_test_1.sh \
		      vftr_setup_test_2.sh \
		      vftr_filewrite_test_1.sh \
		      vftr_filewrite_test_2.sh \
		      vftr_scenario_test_1.sh \
		      vftr_scenario_test_2.sh \
		      vftr_scenario_test_3.sh \
	 	      vftr_functions_test_1.sh \
	 	      vftr_functions_test_2.sh \
	 	      vftr_functions_test_3.sh \
	 	      vftr_functions_test_4.sh \
	 	      vftr_functions_test_5.sh \
		      vftr_stacks_test_1.sh

if MPI_PROFILING
dist_check_SCRIPTS += vftr_stacks_test_2.sh
endif

if VEPERF_AVAIL
dist_check_SCRIPTS += vftr_veperf_test_1.sh \
		      vftr_veperf_test_2.sh
endif



TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)
XFAIL_TESTS = this_fails.sh \
	      vftr_scenario_test_3.sh
