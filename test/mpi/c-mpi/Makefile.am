AM_CFLAGS = -O3 -finstrument-functions

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

################
# System Tests #
################
check_PROGRAMS = init_finalize \
		 ctypes \
		 derived_ctypes
		
init_finalize_SOURCES = $(srcdir)/init_finalize.c
init_finalize_CFLAGS  = $(AM_CFLAGS)
init_finalize_LDFLAGS = $(AM_LDFLAGS)

ctypes_SOURCES = $(srcdir)/ctypes.c
ctypes_CFLAGS  = $(AM_CFLAGS)
ctypes_LDFLAGS = $(AM_LDFLAGS)

derived_ctypes_SOURCES = $(srcdir)/derived_ctypes.c
derived_ctypes_CFLAGS  = $(AM_CFLAGS)
derived_ctypes_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = init_finalize-1.sh \
                      init_finalize-2.sh \
                      init_finalize-3.sh \
                      init_finalize-4.sh \
		      ctypes.sh \
		      derived_ctypes.sh

########################
# point to point tests #
########################
check_PROGRAMS += send_recv \
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

check_PROGRAMS += bcast \
		  bcast_intercom \
		  ibcast \
		  ibcast_intercom \
		  gather \
		  gather_intercom \
		  igather \
		  igather_intercom \
		  gatherv \
		  gatherv_intercom \
		  igatherv \
		  igatherv_intercom \
		  scatter \
		  scatter_intercom \
		  iscatter \
		  iscatter_intercom \
		  scatterv \
		  scatterv_intercom \
		  iscatterv \
		  iscatterv_intercom \
		  allgather \
		  allgather_intercom \
		  iallgather \
		  iallgather_intercom \
		  allgatherv \
		  allgatherv_intercom \
		  iallgatherv \
		  iallgatherv_intercom \
		  alltoall \
		  alltoall_intercom \
		  ialltoall \
		  ialltoall_intercom \
		  alltoallv \
		  alltoallv_intercom \
		  ialltoallv \
		  ialltoallv_intercom \
		  alltoallw \
		  alltoallw_intercom

bcast_SOURCES = $(srcdir)/bcast.c
bcast_CFLAGS = $(AM_CFLAGS)
bcast_LDFLAGS = $(AM_LDFLAGS)

bcast_intercom_SOURCES = $(srcdir)/bcast_intercom.c
bcast_intercom_CFLAGS = $(AM_CFLAGS)
bcast_intercom_LDFLAGS = $(AM_LDFLAGS)

ibcast_SOURCES = $(srcdir)/ibcast.c
ibcast_CFLAGS = $(AM_CFLAGS)
ibcast_LDFLAGS = $(AM_LDFLAGS)

ibcast_intercom_SOURCES = $(srcdir)/ibcast_intercom.c
ibcast_intercom_CFLAGS = $(AM_CFLAGS)
ibcast_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_SOURCES = $(srcdir)/gather.c
gather_CFLAGS = $(AM_CFLAGS)
gather_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.c
gather_intercom_CFLAGS = $(AM_CFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

igather_SOURCES = $(srcdir)/igather.c
igather_CFLAGS = $(AM_CFLAGS)
igather_LDFLAGS = $(AM_LDFLAGS)

igather_intercom_SOURCES = $(srcdir)/igather_intercom.c
igather_intercom_CFLAGS = $(AM_CFLAGS)
igather_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.c
gatherv_CFLAGS = $(AM_CFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.c
gatherv_intercom_CFLAGS = $(AM_CFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

igatherv_SOURCES = $(srcdir)/igatherv.c
igatherv_CFLAGS = $(AM_CFLAGS)
igatherv_LDFLAGS = $(AM_LDFLAGS)

igatherv_intercom_SOURCES = $(srcdir)/igatherv_intercom.c
igatherv_intercom_CFLAGS = $(AM_CFLAGS)
igatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.c
scatter_CFLAGS = $(AM_CFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.c
scatter_intercom_CFLAGS = $(AM_CFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatter_SOURCES = $(srcdir)/iscatter.c
iscatter_CFLAGS = $(AM_CFLAGS)
iscatter_LDFLAGS = $(AM_LDFLAGS)

iscatter_intercom_SOURCES = $(srcdir)/iscatter_intercom.c
iscatter_intercom_CFLAGS = $(AM_CFLAGS)
iscatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.c
scatterv_CFLAGS = $(AM_CFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.c
scatterv_intercom_CFLAGS = $(AM_CFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatterv_SOURCES = $(srcdir)/iscatterv.c
iscatterv_CFLAGS = $(AM_CFLAGS)
iscatterv_LDFLAGS = $(AM_LDFLAGS)

iscatterv_intercom_SOURCES = $(srcdir)/iscatterv_intercom.c
iscatterv_intercom_CFLAGS = $(AM_CFLAGS)
iscatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.c
allgather_CFLAGS = $(AM_CFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.c
allgather_intercom_CFLAGS = $(AM_CFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgather_SOURCES = $(srcdir)/iallgather.c
iallgather_CFLAGS = $(AM_CFLAGS)
iallgather_LDFLAGS = $(AM_LDFLAGS)

iallgather_intercom_SOURCES = $(srcdir)/iallgather_intercom.c
iallgather_intercom_CFLAGS = $(AM_CFLAGS)
iallgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.c
allgatherv_CFLAGS = $(AM_CFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.c
allgatherv_intercom_CFLAGS = $(AM_CFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_SOURCES = $(srcdir)/iallgatherv.c
iallgatherv_CFLAGS = $(AM_CFLAGS)
iallgatherv_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_intercom_SOURCES = $(srcdir)/iallgatherv_intercom.c
iallgatherv_intercom_CFLAGS = $(AM_CFLAGS)
iallgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.c
alltoall_CFLAGS = $(AM_CFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.c
alltoall_intercom_CFLAGS = $(AM_CFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoall_SOURCES = $(srcdir)/ialltoall.c
ialltoall_CFLAGS = $(AM_CFLAGS)
ialltoall_LDFLAGS = $(AM_LDFLAGS)

ialltoall_intercom_SOURCES = $(srcdir)/ialltoall_intercom.c
ialltoall_intercom_CFLAGS = $(AM_CFLAGS)
ialltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.c
alltoallv_CFLAGS = $(AM_CFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.c
alltoallv_intercom_CFLAGS = $(AM_CFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_SOURCES = $(srcdir)/ialltoallv.c
ialltoallv_CFLAGS = $(AM_CFLAGS)
ialltoallv_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_intercom_SOURCES = $(srcdir)/ialltoallv_intercom.c
ialltoallv_intercom_CFLAGS = $(AM_CFLAGS)
ialltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.c
alltoallw_CFLAGS = $(AM_CFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.c
alltoallw_intercom_CFLAGS = $(AM_CFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += bcast.sh \
		       bcast_intercom.sh \
		       ibcast.sh \
		       ibcast_intercom.sh \
		       gather.sh \
		       gather_intercom.sh \
		       igather.sh \
		       igather_intercom.sh \
		       gatherv.sh \
		       gatherv_intercom.sh \
		       igatherv.sh \
		       igatherv_intercom.sh \
		       scatter.sh \
		       scatter_intercom.sh \
		       iscatter.sh \
		       iscatter_intercom.sh \
		       scatterv.sh \
		       scatterv_intercom.sh \
		       iscatterv.sh \
		       iscatterv_intercom.sh \
		       allgather.sh \
		       allgather_intercom.sh \
		       iallgather.sh \
		       iallgather_intercom.sh \
		       allgatherv.sh \
		       allgatherv_intercom.sh \
		       iallgatherv.sh \
		       iallgatherv_intercom.sh \
		       alltoall.sh \
		       alltoall_intercom.sh \
		       ialltoall.sh \
		       ialltoall_intercom.sh \
		       alltoallv.sh \
		       alltoallv_intercom.sh \
		       ialltoallv.sh \
		       ialltoallv_intercom.sh \
		       alltoallw.sh \
		       alltoallw_intercom.sh

#####################
# Global reductions #
#####################

check_PROGRAMS += reduce \
		  reduce_intercom \
		  allreduce \
		  allreduce_intercom

reduce_SOURCES = $(srcdir)/reduce.c
reduce_CFLAGS = $(AM_CFLAGS)
reduce_LDFLAGS = $(AM_LDFLAGS)

reduce_intercom_SOURCES = $(srcdir)/reduce_intercom.c
reduce_intercom_CFLAGS = $(AM_CFLAGS)
reduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_SOURCES = $(srcdir)/allreduce.c
allreduce_CFLAGS = $(AM_CFLAGS)
allreduce_LDFLAGS = $(AM_LDFLAGS)

allreduce_intercom_SOURCES = $(srcdir)/allreduce_intercom.c
allreduce_intercom_CFLAGS = $(AM_CFLAGS)
allreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += reduce.sh \
		       reduce_intercom.sh \
		       allreduce.sh \
		       allreduce_intercom.sh

###################
# One sided tests #
###################
check_PROGRAMS += put \
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
check_PROGRAMS += test \
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

check_PROGRAMS += wait \
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
