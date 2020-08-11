AM_CFLAGS = -O3 -finstrument-functions

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif


noinst_PROGRAMS = init_finalize \
		  types \
		  derived_types
		
init_finalize_SOURCES = $(srcdir)/init_finalize.c
init_finalize_CFLAGS  = $(AM_CFLAGS)
init_finalize_LDFLAGS = $(AM_LDFLAGS)

types_SOURCES = $(srcdir)/types.c
types_CFLAGS  = $(AM_CFLAGS)
types_LDFLAGS = $(AM_LDFLAGS)

derived_types_SOURCES = $(srcdir)/derived_types.c
derived_types_CFLAGS  = $(AM_CFLAGS)
derived_types_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = init_finalize-1.sh \
                      init_finalize-2.sh \
                      init_finalize-3.sh \
                      init_finalize-4.sh \
		      types-1.sh \
		      types-2.sh \
		      derived_types-1.sh \
		      derived_types-2.sh


noinst_PROGRAMS += send_recv \
		   isend_recv

send_recv_SOURCES = $(srcdir)/send_recv.c
send_recv_CFLAGS  = $(AM_CFLAGS)
send_recv_LDFLAGS = $(AM_LDFLAGS)

isend_recv_SOURCES = $(srcdir)/isend_recv.c
isend_recv_CFLAGS  = $(AM_CFLAGS)
isend_recv_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += send_recv-1.sh \
		       send_recv-2.sh \
		       isend_recv-1.sh \
		       isend_recv-2.sh

noinst_PROGRAMS += put \
		   get

put_SOURCES = $(srcdir)/put.c
put_CFLAGS = $(AM_CFLAGS)
put_LDFLAGS = $(AM_LDFLAGS)

get_SOURCES = $(srcdir)/get.c
get_CFLAGS = $(AM_CFLAGS)
get_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += put-1.sh \
		       put-2.sh \
		       get-1.sh \
		       get-2.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
