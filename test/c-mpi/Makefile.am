noinst_PROGRAMS = init_finalize \
	          send_recv
		
init_finalize_SOURCES = $(srcdir)/init_finalize.c
init_finalize_CFLAGS  = -O3 -finstrument-functions
init_finalize_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

send_recv_SOURCES = $(srcdir)/send_recv.c
send_recv_CFLAGS  = -O3 -finstrument-functions
send_recv_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if USES_NEC_MPI
  init_finalize_LDFLAGS += -mpiprof
  send_recv_LDFLAGS += -mpiprof
endif

dist_noinst_SCRIPTS = init_finalize_1.sh \
                      init_finalize_2.sh \
                      init_finalize_3.sh \
                      init_finalize_4.sh \
                      send_recv.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
