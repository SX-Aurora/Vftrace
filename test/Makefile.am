VPATH=${builddir}
bin_PROGRAMS = test_vftrace

SUBDIRS = .
if MPI_PROFILING 
SUBDIRS += c-mpi
#SUBDIRS += c-mpi f-mpi f08-mpi
endif

BUILT_SOURCES = ref_input ref_output run

test_vftrace_SOURCES = ${srcdir}/test_vftrace.c
test_vftrace_CFLAGS = -I${srcdir}/../src/ -I${srcdir}/../src/vftr_mpi
test_vftrace_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

if USES_NEC_MPI
  test_vftrace_LDFLAGS += -mpiprof
endif

run: $(top_srcdir)/test/run
	mkdir -p run
	for f in $</*.sh; do cp $$f $@; chmod u+x $@/`basename $$f`; done

ref_input: $(top_srcdir)/test/ref_input
	mkdir -p ref_input
	for f in $</*; do cp $$f $@; done

ref_output: $(top_srcdir)/test/ref_output
	mkdir -p ref_output
	for f in $</*.out; do cp $$f $@; done

XFAIL_TESTS = ./run/this_fails.sh
 
TESTS = ./run/this_passes.sh \
	./run/vftr_symbols_test_1.sh \
	./run/vftr_environment_test_1.sh \
	./run/vftr_setup_test_1.sh \
	./run/vftr_setup_test_2.sh \
	./run/vftr_filewrite_test_1.sh

TESTS += $(XFAIL_TESTS)

clean-local:
	rm -rf ref_output
	rm -rf ref_input
	rm -rf run
