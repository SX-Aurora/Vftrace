AM_FCFLAGS = -O3 -finstrument-functions

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

################
# System Tests #
################
check_PROGRAMS = init_finalize \
		 ftypes \
		 derived_ftypes
		
init_finalize_SOURCES = $(srcdir)/init_finalize.F90
init_finalize_FCFLAGS = $(AM_FCFLAGS)
init_finalize_LDFLAGS = $(AM_LDFLAGS)

ftypes_SOURCES = $(srcdir)/ftypes.F90
ftypes_FCFLAGS = $(AM_FCFLAGS)
ftypes_LDFLAGS = $(AM_LDFLAGS)

derived_ftypes_SOURCES = $(srcdir)/derived_ftypes.F90
derived_ftypes_FCFLAGS = $(AM_FCFLAGS)
derived_ftypes_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = init_finalize-1.sh \
                      init_finalize-2.sh \
                      init_finalize-3.sh \
                      init_finalize-4.sh \
		      ftypes.sh \
		      derived_ftypes.sh

########################
# point to point tests #
########################
check_PROGRAMS += send_recv \
		  sendrecv \
		  sendrecv_replace \
		  isend_recv \
		  send_irecv

send_recv_SOURCES = $(srcdir)/send_recv.F90
send_recv_FCFLAGS = $(AM_FCFLAGS)
send_recv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_SOURCES = $(srcdir)/sendrecv.F90
sendrecv_FCFLAGS = $(AM_FCFLAGS)
sendrecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_replace_SOURCES = $(srcdir)/sendrecv_replace.F90
sendrecv_replace_FCFLAGS = $(AM_FCFLAGS)
sendrecv_replace_LDFLAGS = $(AM_LDFLAGS)

isend_recv_SOURCES = $(srcdir)/isend_recv.F90
isend_recv_FCFLAGS = $(AM_FCFLAGS)
isend_recv_LDFLAGS = $(AM_LDFLAGS)

send_irecv_SOURCES = $(srcdir)/send_irecv.F90
send_irecv_FCFLAGS = $(AM_FCFLAGS)
send_irecv_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += send_recv.sh \
		       sendrecv.sh \
		       sendrecv_replace.sh \
		       isend_recv.sh \
		       send_irecv.sh

#####################
## Collective tests #
#####################
#
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
		  alltoallw_intercom \
		  ialltoallw \
		  ialltoallw_intercom

bcast_SOURCES = $(srcdir)/bcast.F90
bcast_FCFLAGS = $(AM_FCFLAGS)
bcast_LDFLAGS = $(AM_LDFLAGS)

bcast_intercom_SOURCES = $(srcdir)/bcast_intercom.F90
bcast_intercom_FCFLAGS = $(AM_FCFLAGS)
bcast_intercom_LDFLAGS = $(AM_LDFLAGS)

ibcast_SOURCES = $(srcdir)/ibcast.F90
ibcast_FCFLAGS = $(AM_FCFLAGS)
ibcast_LDFLAGS = $(AM_LDFLAGS)

ibcast_intercom_SOURCES = $(srcdir)/ibcast_intercom.F90
ibcast_intercom_FCFLAGS = $(AM_FCFLAGS)
ibcast_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_SOURCES = $(srcdir)/gather.F90
gather_FCFLAGS = $(AM_FCFLAGS)
gather_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.F90
gather_intercom_FCFLAGS = $(AM_FCFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

igather_SOURCES = $(srcdir)/igather.F90
igather_FCFLAGS = $(AM_FCFLAGS)
igather_LDFLAGS = $(AM_LDFLAGS)

igather_intercom_SOURCES = $(srcdir)/igather_intercom.F90
igather_intercom_FCFLAGS = $(AM_FCFLAGS)
igather_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.F90
gatherv_FCFLAGS = $(AM_FCFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.F90
gatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

igatherv_SOURCES = $(srcdir)/igatherv.F90
igatherv_FCFLAGS = $(AM_FCFLAGS)
igatherv_LDFLAGS = $(AM_LDFLAGS)

igatherv_intercom_SOURCES = $(srcdir)/igatherv_intercom.F90
igatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
igatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.F90
scatter_FCFLAGS = $(AM_FCFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.F90
scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatter_SOURCES = $(srcdir)/iscatter.F90
iscatter_FCFLAGS = $(AM_FCFLAGS)
iscatter_LDFLAGS = $(AM_LDFLAGS)

iscatter_intercom_SOURCES = $(srcdir)/iscatter_intercom.F90
iscatter_intercom_FCFLAGS = $(AM_FCFLAGS)
iscatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.F90
scatterv_FCFLAGS = $(AM_FCFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.F90
scatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatterv_SOURCES = $(srcdir)/iscatterv.F90
iscatterv_FCFLAGS = $(AM_FCFLAGS)
iscatterv_LDFLAGS = $(AM_LDFLAGS)

iscatterv_intercom_SOURCES = $(srcdir)/iscatterv_intercom.F90
iscatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
iscatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.F90
allgather_FCFLAGS = $(AM_FCFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.F90
allgather_intercom_FCFLAGS = $(AM_FCFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgather_SOURCES = $(srcdir)/iallgather.F90
iallgather_FCFLAGS = $(AM_FCFLAGS)
iallgather_LDFLAGS = $(AM_LDFLAGS)

iallgather_intercom_SOURCES = $(srcdir)/iallgather_intercom.F90
iallgather_intercom_FCFLAGS = $(AM_FCFLAGS)
iallgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.F90
allgatherv_FCFLAGS = $(AM_FCFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.F90
allgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_SOURCES = $(srcdir)/iallgatherv.F90
iallgatherv_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_intercom_SOURCES = $(srcdir)/iallgatherv_intercom.F90
iallgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.F90
alltoall_FCFLAGS = $(AM_FCFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.F90
alltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoall_SOURCES = $(srcdir)/ialltoall.F90
ialltoall_FCFLAGS = $(AM_FCFLAGS)
ialltoall_LDFLAGS = $(AM_LDFLAGS)

ialltoall_intercom_SOURCES = $(srcdir)/ialltoall_intercom.F90
ialltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.F90
alltoallv_FCFLAGS = $(AM_FCFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.F90
alltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_SOURCES = $(srcdir)/ialltoallv.F90
ialltoallv_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_intercom_SOURCES = $(srcdir)/ialltoallv_intercom.F90
ialltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.F90
alltoallw_FCFLAGS = $(AM_FCFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.F90
alltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_SOURCES = $(srcdir)/ialltoallw.F90
ialltoallw_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_intercom_SOURCES = $(srcdir)/ialltoallw_intercom.F90
ialltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

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
		       alltoallw_intercom.sh \
		       ialltoallw.sh \
		       ialltoallw_intercom.sh

#####################
# Global reductions #
#####################

check_PROGRAMS += reduce \
		  reduce_intercom \
		  ireduce \
		  ireduce_intercom \
		  allreduce \
		  allreduce_intercom \
		  iallreduce \
		  iallreduce_intercom \
		  scan \
		  iscan \
		  exscan \
		  iexscan

reduce_SOURCES = $(srcdir)/reduce.F90
reduce_FCFLAGS = $(AM_FCFLAGS)
reduce_LDFLAGS = $(AM_LDFLAGS)

reduce_intercom_SOURCES = $(srcdir)/reduce_intercom.F90
reduce_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_SOURCES = $(srcdir)/ireduce.F90
ireduce_FCFLAGS = $(AM_FCFLAGS)
ireduce_LDFLAGS = $(AM_LDFLAGS)

ireduce_intercom_SOURCES = $(srcdir)/ireduce_intercom.F90
ireduce_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_SOURCES = $(srcdir)/allreduce.F90
allreduce_FCFLAGS = $(AM_FCFLAGS)
allreduce_LDFLAGS = $(AM_LDFLAGS)

allreduce_intercom_SOURCES = $(srcdir)/allreduce_intercom.F90
allreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
allreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

iallreduce_SOURCES = $(srcdir)/iallreduce.F90
iallreduce_FCFLAGS = $(AM_FCFLAGS)
iallreduce_LDFLAGS = $(AM_LDFLAGS)

iallreduce_intercom_SOURCES = $(srcdir)/iallreduce_intercom.F90
iallreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
iallreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

scan_SOURCES = $(srcdir)/scan.F90
scan_FCFLAGS = $(AM_FCFLAGS)
scan_LDFLAGS = $(AM_LDFLAGS)

iscan_SOURCES = $(srcdir)/iscan.F90
iscan_FCFLAGS = $(AM_FCFLAGS)
iscan_LDFLAGS = $(AM_LDFLAGS)

exscan_SOURCES = $(srcdir)/exscan.F90
exscan_FCFLAGS = $(AM_FCFLAGS)
exscan_LDFLAGS = $(AM_LDFLAGS)

iexscan_SOURCES = $(srcdir)/iexscan.F90
iexscan_FCFLAGS = $(AM_FCFLAGS)
iexscan_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += reduce.sh \
		       reduce_intercom.sh \
		       ireduce.sh \
		       ireduce_intercom.sh \
		       allreduce.sh \
		       allreduce_intercom.sh \
		       iallreduce.sh \
		       iallreduce_intercom.sh \
		       scan.sh \
		       iscan.sh \
		       exscan.sh \
		       iexscan.sh

###################
# One sided tests #
###################
check_PROGRAMS += put \
		  rput \
		  get \
		  rget \
		  accumulate \
		  raccumulate

put_SOURCES = $(srcdir)/put.F90
put_FCFLAGS = $(AM_FCFLAGS)
put_LDFLAGS = $(AM_LDFLAGS)

rput_SOURCES = $(srcdir)/rput.F90
rput_FCFLAGS = $(AM_FCFLAGS)
rput_LDFLAGS = $(AM_LDFLAGS)

get_SOURCES = $(srcdir)/get.F90
get_FCFLAGS = $(AM_FCFLAGS)
get_LDFLAGS = $(AM_LDFLAGS)

rget_SOURCES = $(srcdir)/rget.F90
rget_FCFLAGS = $(AM_FCFLAGS)
rget_LDFLAGS = $(AM_LDFLAGS)

accumulate_SOURCES = $(srcdir)/accumulate.F90
accumulate_FCFLAGS = $(AM_FCFLAGS)
accumulate_LDFLAGS = $(AM_LDFLAGS)

raccumulate_SOURCES = $(srcdir)/raccumulate.F90
raccumulate_FCFLAGS = $(AM_FCFLAGS)
raccumulate_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += put.sh \
		       rput.sh \
		       get.sh \
		       rget.sh \
		       accumulate.sh \
		       raccumulate.sh

##############
# Test tests #
##############
check_PROGRAMS += test \
		  testany \
		  testsome \
		  testall

test_SOURCES = $(srcdir)/test.F90
test_FCFLAGS = $(AM_FCFLAGS)
test_LDFLAGS = $(AM_LDFLAGS)

testany_SOURCES = $(srcdir)/testany.F90
testany_FCFLAGS = $(AM_FCFLAGS)
testany_LDFLAGS = $(AM_LDFLAGS)

testsome_SOURCES = $(srcdir)/testsome.F90
testsome_FCFLAGS = $(AM_FCFLAGS)
testsome_LDFLAGS = $(AM_LDFLAGS)

testall_SOURCES = $(srcdir)/testall.F90
testall_FCFLAGS = $(AM_FCFLAGS)
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

wait_SOURCES = $(srcdir)/wait.F90
wait_FCFLAGS = $(AM_FCFLAGS)
wait_LDFLAGS = $(AM_LDFLAGS)

waitany_SOURCES = $(srcdir)/waitany.F90
waitany_FCFLAGS = $(AM_FCFLAGS)
waitany_LDFLAGS = $(AM_LDFLAGS)

waitsome_SOURCES = $(srcdir)/waitsome.F90
waitsome_FCFLAGS = $(AM_FCFLAGS)
waitsome_LDFLAGS = $(AM_LDFLAGS)

waitall_SOURCES = $(srcdir)/waitall.F90
waitall_FCFLAGS = $(AM_FCFLAGS)
waitall_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += wait.sh \
		       waitany.sh \
		       waitsome.sh \
		       waitall.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
