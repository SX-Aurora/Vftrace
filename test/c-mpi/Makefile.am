AM_CFLAGS = -O3 -finstrument-functions

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

################
# System Tests #
################
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

########################
# point to point tests #
########################
noinst_PROGRAMS += send_recv \
		   sendrecv \
		   sendrecv_replace \
		   isend_recv \
		   send_irecv

send_recv_SOURCES = $(srcdir)/send_recv.c
send_recv_CFLAGS  = $(AM_CFLAGS)
send_recv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_SOURCES = $(srcdir)/sendrecv.c
sendrecv_CFLAGS  = $(AM_CFLAGS)
sendrecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_replace_SOURCES = $(srcdir)/sendrecv_replace.c
sendrecv_replace_CFLAGS  = $(AM_CFLAGS)
sendrecv_replace_LDFLAGS = $(AM_LDFLAGS)

isend_recv_SOURCES = $(srcdir)/isend_recv.c
isend_recv_CFLAGS  = $(AM_CFLAGS)
isend_recv_LDFLAGS = $(AM_LDFLAGS)

send_irecv_SOURCES = $(srcdir)/send_irecv.c
send_irecv_CFLAGS  = $(AM_CFLAGS)
send_irecv_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += send_recv-1.sh \
		       send_recv-2.sh \
		       sendrecv-1.sh \
		       sendrecv-2.sh \
		       sendrecv_replace-1.sh \
		       sendrecv_replace-2.sh \
		       isend_recv-1.sh \
		       isend_recv-2.sh \
		       send_irecv-1.sh \
		       send_irecv-2.sh

####################
# Collective tests #
####################

noinst_PROGRAMS += bcast \
		   bcast_intercom

bcast_SOURCES = $(srcdir)/bcast.c
bcast_CFLAGS = $(AM_CFLAGS)
bcast_LDFLAGS = $(AM_LDFLAGS)

bcast_intercom_SOURCES = $(srcdir)/bcast_intercom.c
bcast_intercom_CFLAGS = $(AM_CFLAGS)
bcast_intercom_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += bcast-1.sh \
		       bcast-2.sh \
		       bcast_intercom-1.sh

###################
# One sided tests #
###################
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

##############
# Test tests #
##############
noinst_PROGRAMS += test \
		   testany \
		   testsome \
		   testall

test_SOURCES = $(srcdir)/test.c
test_CFLAGS = $(AM_CFLAGS)
test_LDFLAGS = $(AM_LDFLAGS)

testany_SOURCES = $(srcdir)/testany.c
testany_CFLAGS = $(AM_CFLAGS)
testany_LDFLAGS = $(AM_LDFLAGS)

testsome_SOURCES = $(srcdir)/testsome.c
testsome_CFLAGS = $(AM_CFLAGS)
testsome_LDFLAGS = $(AM_LDFLAGS)

testall_SOURCES = $(srcdir)/testall.c
testall_CFLAGS = $(AM_CFLAGS)
testall_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += test-1.sh \
		       test-2.sh \
		       testany-1.sh \
		       testany-2.sh \
		       testsome-1.sh \
		       testsome-2.sh \
		       testall-1.sh \
		       testall-2.sh

##############
# Wait tests #
##############

noinst_PROGRAMS += wait \
		   waitany \
		   waitsome \
		   waitall

wait_SOURCES = $(srcdir)/wait.c
wait_CFLAGS = $(AM_CFLAGS)
wait_LDFLAGS = $(AM_LDFLAGS)

waitany_SOURCES = $(srcdir)/waitany.c
waitany_CFLAGS = $(AM_CFLAGS)
waitany_LDFLAGS = $(AM_LDFLAGS)

waitsome_SOURCES = $(srcdir)/waitsome.c
waitsome_CFLAGS = $(AM_CFLAGS)
waitsome_LDFLAGS = $(AM_LDFLAGS)

waitall_SOURCES = $(srcdir)/waitall.c
waitall_CFLAGS = $(AM_CFLAGS)
waitall_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += wait-1.sh \
		       wait-2.sh \
		       waitany-1.sh \
		       waitany-2.sh \
		       waitsome-1.sh \
		       waitsome-2.sh \
		       waitall-1.sh \
		       waitall-2.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
