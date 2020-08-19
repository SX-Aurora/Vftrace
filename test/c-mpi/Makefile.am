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
		      types.sh \
		      derived_types.sh

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

dist_noinst_SCRIPTS += send_recv.sh \
		       sendrecv.sh \
		       sendrecv_replace.sh \
		       isend_recv.sh \
		       send_irecv.sh

####################
# Collective tests #
####################

noinst_PROGRAMS += bcast \
		   bcast_intercom \
		   gather \
		   gather_intercom \
		   gatherv \
		   gatherv_intercom \
		   scatter \
		   scatter_intercom \
		   scatterv \
		   scatterv_intercom \
		   allgather \
		   allgather_intercom \
		   allgatherv \
		   allgatherv_intercom \
		   alltoall \
		   alltoall_intercom \
		   alltoallv \
		   alltoallv_intercom \
		   alltoallw \
		   alltoallw_intercom

bcast_SOURCES = $(srcdir)/bcast.c
bcast_CFLAGS = $(AM_CFLAGS)
bcast_LDFLAGS = $(AM_LDFLAGS)

bcast_intercom_SOURCES = $(srcdir)/bcast_intercom.c
bcast_intercom_CFLAGS = $(AM_CFLAGS)
bcast_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_SOURCES = $(srcdir)/gather.c
gather_CFLAGS = $(AM_CFLAGS)
gather_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.c
gather_intercom_CFLAGS = $(AM_CFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.c
gatherv_CFLAGS = $(AM_CFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.c
gatherv_intercom_CFLAGS = $(AM_CFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.c
scatter_CFLAGS = $(AM_CFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.c
scatter_intercom_CFLAGS = $(AM_CFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.c
scatterv_CFLAGS = $(AM_CFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.c
scatterv_intercom_CFLAGS = $(AM_CFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.c
allgather_CFLAGS = $(AM_CFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.c
allgather_intercom_CFLAGS = $(AM_CFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.c
allgatherv_CFLAGS = $(AM_CFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.c
allgatherv_intercom_CFLAGS = $(AM_CFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.c
alltoall_CFLAGS = $(AM_CFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.c
alltoall_intercom_CFLAGS = $(AM_CFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.c
alltoallv_CFLAGS = $(AM_CFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.c
alltoallv_intercom_CFLAGS = $(AM_CFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.c
alltoallw_CFLAGS = $(AM_CFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.c
alltoallw_intercom_CFLAGS = $(AM_CFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += bcast.sh \
		       bcast_intercom.sh \
		       gather.sh \
		       gather_intercom.sh \
		       gatherv.sh \
		       gatherv_intercom.sh \
		       scatter.sh \
		       scatter_intercom.sh \
		       scatterv.sh \
		       scatterv_intercom.sh \
		       allgather.sh \
		       allgather_intercom.sh \
		       allgatherv.sh \
		       allgatherv_intercom.sh \
		       alltoall.sh \
		       alltoall_intercom.sh \
		       alltoallv.sh \
		       alltoallv_intercom.sh \
		       alltoallw.sh \
		       alltoallw_intercom.sh

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

dist_noinst_SCRIPTS += put.sh \
		       get.sh

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

dist_noinst_SCRIPTS += test.sh \
		       testany.sh \
		       testsome.sh \
		       testall.sh

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

dist_noinst_SCRIPTS += wait.sh \
		       waitany.sh \
		       waitsome.sh \
		       waitall.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
