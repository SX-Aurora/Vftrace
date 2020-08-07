VPATH=${builddir}
bin_PROGRAMS = init_finalize \
	       send_recv 

BUILT_SOURCES = run
#BUILT_SOURCES = ref_input ref_output run

init_finalize_SOURCES = ${srcdir}/init_finalize.c
init_finalize_CFLAGS  = -O3 -finstrument-functions
init_finalize_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

send_recv_SOURCES = ${srcdir}/send_recv.c
send_recv_CFLAGS  = -O3 -finstrument-functions
send_recv_LDFLAGS = -L${top_builddir}/src/.libs -lvftrace -lm

LDADD = -L${top_builddir}/src/.libs -lvftrace -lm

if USES_NEC_MPI
  init_finalize_LDFLAGS += -mpiprof
  send_recv_LDFLAGS += -mpiprof
endif

run: $(top_srcdir)/test/c-mpi/run
	mkdir -p run
	for f in $</*.sh; do cp $$f $@; chmod u+x $@/`basename $$f`; done

ref_input: $(top_srcdir)/test/c-mpi/ref_input
	mkdir -p ref_input
	for f in $</*; do cp $$f $@; done

ref_output: $(top_srcdir)/test/c-mpi/ref_output
	mkdir -p ref_output
	for f in $</*.out; do cp $$f $@; done

TESTS = ./run/init_finalize_1.sh \
	./run/init_finalize_2.sh \
	./run/init_finalize_3.sh \
	./run/init_finalize_4.sh \
	./run/send_recv.sh

clean-local:
	rm -rf ref_output
	rm -rf ref_input
	rm -rf run
