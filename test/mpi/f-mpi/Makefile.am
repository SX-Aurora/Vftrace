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
		  isend_recv \
		  bsend_recv \
		  ibsend_recv \
		  ssend_recv \
		  issend_recv \
		  rsend_recv \
		  irsend_recv \
		  send_irecv \
		  sendrecv \
		  sendrecv_replace 

send_recv_SOURCES = $(srcdir)/send_recv.F90
send_recv_FCFLAGS = $(AM_FCFLAGS)
send_recv_LDFLAGS = $(AM_LDFLAGS)

isend_recv_SOURCES = $(srcdir)/isend_recv.F90
isend_recv_FCFLAGS = $(AM_FCFLAGS)
isend_recv_LDFLAGS = $(AM_LDFLAGS)

bsend_recv_SOURCES = $(srcdir)/bsend_recv.F90
bsend_recv_FCFLAGS = $(AM_FCFLAGS)
bsend_recv_LDFLAGS = $(AM_LDFLAGS)

ibsend_recv_SOURCES = $(srcdir)/ibsend_recv.F90
ibsend_recv_FCFLAGS = $(AM_FCFLAGS)
ibsend_recv_LDFLAGS = $(AM_LDFLAGS)

ssend_recv_SOURCES = $(srcdir)/ssend_recv.F90
ssend_recv_FCFLAGS = $(AM_FCFLAGS)
ssend_recv_LDFLAGS = $(AM_LDFLAGS)

issend_recv_SOURCES = $(srcdir)/issend_recv.F90
issend_recv_FCFLAGS = $(AM_FCFLAGS)
issend_recv_LDFLAGS = $(AM_LDFLAGS)

send_irecv_SOURCES = $(srcdir)/send_irecv.F90
send_irecv_FCFLAGS = $(AM_FCFLAGS)
send_irecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_SOURCES = $(srcdir)/sendrecv.F90
sendrecv_FCFLAGS = $(AM_FCFLAGS)
sendrecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_replace_SOURCES = $(srcdir)/sendrecv_replace.F90
sendrecv_replace_FCFLAGS = $(AM_FCFLAGS)
sendrecv_replace_LDFLAGS = $(AM_LDFLAGS)

rsend_recv_SOURCES = $(srcdir)/rsend_recv.F90
rsend_recv_FCFLAGS = $(AM_FCFLAGS)
rsend_recv_LDFLAGS = $(AM_LDFLAGS)

irsend_recv_SOURCES = $(srcdir)/irsend_recv.F90
irsend_recv_FCFLAGS = $(AM_FCFLAGS)
irsend_recv_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += send_recv.sh \
		       isend_recv.sh \
		       bsend_recv.sh \
		       ibsend_recv.sh \
		       ssend_recv.sh \
		       issend_recv.sh \
		       rsend_recv.sh \
		       irsend_recv.sh \
		       send_irecv.sh \
		       sendrecv.sh \
		       sendrecv_replace.sh 

#####################
## Collective tests #
#####################
#
check_PROGRAMS += barrier \
		  bcast \
		  bcast_intercom \
		  ibcast \
		  ibcast_intercom \
		  gather \
		  gather_inplace \
		  gather_intercom \
		  igather \
		  igather_inplace \
		  igather_intercom \
		  gatherv \
		  gatherv_inplace \
		  gatherv_intercom \
		  igatherv \
		  igatherv_inplace \
		  igatherv_intercom \
		  scatter \
		  scatter_inplace \
		  scatter_intercom \
		  iscatter \
		  iscatter_inplace \
		  iscatter_intercom \
		  scatterv \
		  scatterv_inplace \
		  scatterv_intercom \
		  iscatterv \
		  iscatterv_inplace \
		  iscatterv_intercom \
		  allgather \
		  allgather_inplace \
		  allgather_intercom \
		  iallgather \
		  iallgather_inplace \
		  iallgather_intercom \
		  allgatherv \
		  allgatherv_inplace \
		  allgatherv_intercom \
		  iallgatherv \
		  iallgatherv_inplace \
		  iallgatherv_intercom \
		  alltoall \
		  alltoall_inplace \
		  alltoall_intercom \
		  ialltoall \
		  ialltoall_inplace \
		  ialltoall_intercom \
		  alltoallv \
		  alltoallv_inplace \
		  alltoallv_intercom \
		  ialltoallv \
		  ialltoallv_inplace \
		  ialltoallv_intercom \
		  alltoallw \
		  alltoallw_inplace \
		  alltoallw_intercom \
		  ialltoallw \
		  ialltoallw_inplace \
		  ialltoallw_intercom

barrier_SOURCES = $(srcdir)/barrier.F90
barrier_FCFLAGS = $(AM_FCFLAGS)
barrier_LDFLAGS = $(AM_LDFLAGS)

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

gather_inplace_SOURCES = $(srcdir)/gather_inplace.F90
gather_inplace_FCFLAGS = $(AM_FCFLAGS)
gather_inplace_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.F90
gather_intercom_FCFLAGS = $(AM_FCFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

igather_SOURCES = $(srcdir)/igather.F90
igather_FCFLAGS = $(AM_FCFLAGS)
igather_LDFLAGS = $(AM_LDFLAGS)

igather_inplace_SOURCES = $(srcdir)/igather_inplace.F90
igather_inplace_FCFLAGS = $(AM_FCFLAGS)
igather_inplace_LDFLAGS = $(AM_LDFLAGS)

igather_intercom_SOURCES = $(srcdir)/igather_intercom.F90
igather_intercom_FCFLAGS = $(AM_FCFLAGS)
igather_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.F90
gatherv_FCFLAGS = $(AM_FCFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_inplace_SOURCES = $(srcdir)/gatherv_inplace.F90
gatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
gatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.F90
gatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

igatherv_SOURCES = $(srcdir)/igatherv.F90
igatherv_FCFLAGS = $(AM_FCFLAGS)
igatherv_LDFLAGS = $(AM_LDFLAGS)

igatherv_inplace_SOURCES = $(srcdir)/igatherv_inplace.F90
igatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
igatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

igatherv_intercom_SOURCES = $(srcdir)/igatherv_intercom.F90
igatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
igatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.F90
scatter_FCFLAGS = $(AM_FCFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_inplace_SOURCES = $(srcdir)/scatter_inplace.F90
scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.F90
scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatter_SOURCES = $(srcdir)/iscatter.F90
iscatter_FCFLAGS = $(AM_FCFLAGS)
iscatter_LDFLAGS = $(AM_LDFLAGS)

iscatter_inplace_SOURCES = $(srcdir)/iscatter_inplace.F90
iscatter_inplace_FCFLAGS = $(AM_FCFLAGS)
iscatter_inplace_LDFLAGS = $(AM_LDFLAGS)

iscatter_intercom_SOURCES = $(srcdir)/iscatter_intercom.F90
iscatter_intercom_FCFLAGS = $(AM_FCFLAGS)
iscatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.F90
scatterv_FCFLAGS = $(AM_FCFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_inplace_SOURCES = $(srcdir)/scatterv_inplace.F90
scatterv_inplace_FCFLAGS = $(AM_FCFLAGS)
scatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.F90
scatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatterv_SOURCES = $(srcdir)/iscatterv.F90
iscatterv_FCFLAGS = $(AM_FCFLAGS)
iscatterv_LDFLAGS = $(AM_LDFLAGS)

iscatterv_inplace_SOURCES = $(srcdir)/iscatterv_inplace.F90
iscatterv_inplace_FCFLAGS = $(AM_FCFLAGS)
iscatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

iscatterv_intercom_SOURCES = $(srcdir)/iscatterv_intercom.F90
iscatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
iscatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.F90
allgather_FCFLAGS = $(AM_FCFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_inplace_SOURCES = $(srcdir)/allgather_inplace.F90
allgather_inplace_FCFLAGS = $(AM_FCFLAGS)
allgather_inplace_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.F90
allgather_intercom_FCFLAGS = $(AM_FCFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgather_SOURCES = $(srcdir)/iallgather.F90
iallgather_FCFLAGS = $(AM_FCFLAGS)
iallgather_LDFLAGS = $(AM_LDFLAGS)

iallgather_inplace_SOURCES = $(srcdir)/iallgather_inplace.F90
iallgather_inplace_FCFLAGS = $(AM_FCFLAGS)
iallgather_inplace_LDFLAGS = $(AM_LDFLAGS)

iallgather_intercom_SOURCES = $(srcdir)/iallgather_intercom.F90
iallgather_intercom_FCFLAGS = $(AM_FCFLAGS)
iallgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.F90
allgatherv_FCFLAGS = $(AM_FCFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_inplace_SOURCES = $(srcdir)/allgatherv_inplace.F90
allgatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
allgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.F90
allgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_SOURCES = $(srcdir)/iallgatherv.F90
iallgatherv_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_inplace_SOURCES = $(srcdir)/iallgatherv_inplace.F90
iallgatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_intercom_SOURCES = $(srcdir)/iallgatherv_intercom.F90
iallgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.F90
alltoall_FCFLAGS = $(AM_FCFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_inplace_SOURCES = $(srcdir)/alltoall_inplace.F90
alltoall_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.F90
alltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoall_SOURCES = $(srcdir)/ialltoall.F90
ialltoall_FCFLAGS = $(AM_FCFLAGS)
ialltoall_LDFLAGS = $(AM_LDFLAGS)

ialltoall_inplace_SOURCES = $(srcdir)/ialltoall_inplace.F90
ialltoall_inplace_FCFLAGS = $(AM_FCFLAGS)
ialltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoall_intercom_SOURCES = $(srcdir)/ialltoall_intercom.F90
ialltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.F90
alltoallv_FCFLAGS = $(AM_FCFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_inplace_SOURCES = $(srcdir)/alltoallv_inplace.F90
alltoallv_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.F90
alltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_SOURCES = $(srcdir)/ialltoallv.F90
ialltoallv_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_inplace_SOURCES = $(srcdir)/ialltoallv_inplace.F90
ialltoallv_inplace_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_intercom_SOURCES = $(srcdir)/ialltoallv_intercom.F90
ialltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.F90
alltoallw_FCFLAGS = $(AM_FCFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_inplace_SOURCES = $(srcdir)/alltoallw_inplace.F90
alltoallw_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.F90
alltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_SOURCES = $(srcdir)/ialltoallw.F90
ialltoallw_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_inplace_SOURCES = $(srcdir)/ialltoallw_inplace.F90
ialltoallw_inplace_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_intercom_SOURCES = $(srcdir)/ialltoallw_intercom.F90
ialltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += barrier.sh \
		       bcast.sh \
		       bcast_intercom.sh \
		       ibcast.sh \
		       ibcast_intercom.sh \
		       gather.sh \
		       gather_inplace.sh \
		       gather_intercom.sh \
		       igather.sh \
		       igather_inplace.sh \
		       igather_intercom.sh \
		       gatherv.sh \
		       gatherv_inplace.sh \
		       gatherv_intercom.sh \
		       igatherv.sh \
		       igatherv_inplace.sh \
		       igatherv_intercom.sh \
		       scatter.sh \
		       scatter_inplace.sh \
		       scatter_intercom.sh \
		       iscatter.sh \
		       iscatter_inplace.sh \
		       iscatter_intercom.sh \
		       scatterv.sh \
		       scatterv_inplace.sh \
		       scatterv_intercom.sh \
		       iscatterv.sh \
		       iscatterv_inplace.sh \
		       iscatterv_intercom.sh \
		       allgather.sh \
		       allgather_inplace.sh \
		       allgather_intercom.sh \
		       iallgather.sh \
		       iallgather_inplace.sh \
		       iallgather_intercom.sh \
		       allgatherv.sh \
		       allgatherv_inplace.sh \
		       allgatherv_intercom.sh \
		       iallgatherv.sh \
		       iallgatherv_inplace.sh \
		       iallgatherv_intercom.sh \
		       alltoall.sh \
		       alltoall_inplace.sh \
		       alltoall_intercom.sh \
		       ialltoall.sh \
		       ialltoall_inplace.sh \
		       ialltoall_intercom.sh \
		       alltoallv.sh \
		       alltoallv_inplace.sh \
		       alltoallv_intercom.sh \
		       ialltoallv.sh \
		       ialltoallv_inplace.sh \
		       ialltoallv_intercom.sh \
		       alltoallw.sh \
		       alltoallw_inplace.sh \
		       alltoallw_intercom.sh \
		       ialltoallw.sh \
		       ialltoallw_inplace.sh \
		       ialltoallw_intercom.sh

#####################
# Global reductions #
#####################

check_PROGRAMS += reduce \
		  reduce_inplace \
		  reduce_intercom \
		  ireduce \
		  ireduce_inplace \
		  ireduce_intercom \
		  allreduce \
		  allreduce_inplace \
		  allreduce_intercom \
		  iallreduce \
		  iallreduce_inplace \
		  iallreduce_intercom \
		  reduce_scatter_block \
		  reduce_scatter_block_inplace \
		  reduce_scatter_block_intercom \
		  ireduce_scatter_block \
		  ireduce_scatter_block_inplace \
		  ireduce_scatter_block_intercom \
		  reduce_scatter \
		  reduce_scatter_inplace \
		  reduce_scatter_intercom \
		  ireduce_scatter \
		  ireduce_scatter_inplace \
		  ireduce_scatter_intercom \
		  scan \
		  scan_inplace \
		  iscan \
		  iscan_inplace \
		  exscan \
		  exscan_inplace \
		  iexscan \
		  iexscan_inplace

reduce_SOURCES = $(srcdir)/reduce.F90
reduce_FCFLAGS = $(AM_FCFLAGS)
reduce_LDFLAGS = $(AM_LDFLAGS)

reduce_inplace_SOURCES = $(srcdir)/reduce_inplace.F90
reduce_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_intercom_SOURCES = $(srcdir)/reduce_intercom.F90
reduce_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_SOURCES = $(srcdir)/ireduce.F90
ireduce_FCFLAGS = $(AM_FCFLAGS)
ireduce_LDFLAGS = $(AM_LDFLAGS)

ireduce_inplace_SOURCES = $(srcdir)/ireduce_inplace.F90
ireduce_inplace_FCFLAGS = $(AM_FCFLAGS)
ireduce_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_intercom_SOURCES = $(srcdir)/ireduce_intercom.F90
ireduce_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_SOURCES = $(srcdir)/allreduce.F90
allreduce_FCFLAGS = $(AM_FCFLAGS)
allreduce_LDFLAGS = $(AM_LDFLAGS)

allreduce_inplace_SOURCES = $(srcdir)/allreduce_inplace.F90
allreduce_inplace_FCFLAGS = $(AM_FCFLAGS)
allreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

allreduce_intercom_SOURCES = $(srcdir)/allreduce_intercom.F90
allreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
allreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

iallreduce_SOURCES = $(srcdir)/iallreduce.F90
iallreduce_FCFLAGS = $(AM_FCFLAGS)
iallreduce_LDFLAGS = $(AM_LDFLAGS)

iallreduce_inplace_SOURCES = $(srcdir)/iallreduce_inplace.F90
iallreduce_inplace_FCFLAGS = $(AM_FCFLAGS)
iallreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

iallreduce_intercom_SOURCES = $(srcdir)/iallreduce_intercom.F90
iallreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
iallreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_SOURCES = $(srcdir)/reduce_scatter_block.F90
reduce_scatter_block_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_inplace_SOURCES = $(srcdir)/reduce_scatter_block_inplace.F90
reduce_scatter_block_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_intercom_SOURCES = $(srcdir)/reduce_scatter_block_intercom.F90
reduce_scatter_block_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_SOURCES = $(srcdir)/ireduce_scatter_block.F90
ireduce_scatter_block_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_inplace_SOURCES = $(srcdir)/ireduce_scatter_block_inplace.F90
ireduce_scatter_block_inplace_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_intercom_SOURCES = $(srcdir)/ireduce_scatter_block_intercom.F90
ireduce_scatter_block_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_SOURCES = $(srcdir)/reduce_scatter.F90
reduce_scatter_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_inplace_SOURCES = $(srcdir)/reduce_scatter_inplace.F90
reduce_scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_intercom_SOURCES = $(srcdir)/reduce_scatter_intercom.F90
reduce_scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_SOURCES = $(srcdir)/ireduce_scatter.F90
ireduce_scatter_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_inplace_SOURCES = $(srcdir)/ireduce_scatter_inplace.F90
ireduce_scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_intercom_SOURCES = $(srcdir)/ireduce_scatter_intercom.F90
ireduce_scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scan_SOURCES = $(srcdir)/scan.F90
scan_FCFLAGS = $(AM_FCFLAGS)
scan_LDFLAGS = $(AM_LDFLAGS)

scan_inplace_SOURCES = $(srcdir)/scan_inplace.F90
scan_inplace_FCFLAGS = $(AM_FCFLAGS)
scan_inplace_LDFLAGS = $(AM_LDFLAGS)

iscan_SOURCES = $(srcdir)/iscan.F90
iscan_FCFLAGS = $(AM_FCFLAGS)
iscan_LDFLAGS = $(AM_LDFLAGS)

iscan_inplace_SOURCES = $(srcdir)/iscan_inplace.F90
iscan_inplace_FCFLAGS = $(AM_FCFLAGS)
iscan_inplace_LDFLAGS = $(AM_LDFLAGS)

exscan_SOURCES = $(srcdir)/exscan.F90
exscan_FCFLAGS = $(AM_FCFLAGS)
exscan_LDFLAGS = $(AM_LDFLAGS)

exscan_inplace_SOURCES = $(srcdir)/exscan_inplace.F90
exscan_inplace_FCFLAGS = $(AM_FCFLAGS)
exscan_inplace_LDFLAGS = $(AM_LDFLAGS)

iexscan_SOURCES = $(srcdir)/iexscan.F90
iexscan_FCFLAGS = $(AM_FCFLAGS)
iexscan_LDFLAGS = $(AM_LDFLAGS)

iexscan_inplace_SOURCES = $(srcdir)/iexscan_inplace.F90
iexscan_inplace_FCFLAGS = $(AM_FCFLAGS)
iexscan_inplace_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += reduce.sh \
		       reduce_inplace.sh \
		       reduce_intercom.sh \
		       ireduce.sh \
		       ireduce_inplace.sh \
		       ireduce_intercom.sh \
		       allreduce.sh \
		       allreduce_inplace.sh \
		       allreduce_intercom.sh \
		       iallreduce.sh \
		       iallreduce_inplace.sh \
		       iallreduce_intercom.sh \
		       reduce_scatter_block.sh \
		       reduce_scatter_block_inplace.sh \
		       reduce_scatter_block_intercom.sh \
		       ireduce_scatter_block.sh \
		       ireduce_scatter_block_inplace.sh \
		       ireduce_scatter_block_intercom.sh \
		       reduce_scatter.sh \
		       reduce_scatter_inplace.sh \
		       reduce_scatter_intercom.sh \
		       ireduce_scatter.sh \
		       ireduce_scatter_inplace.sh \
		       ireduce_scatter_intercom.sh \
		       scan.sh \
		       scan_inplace.sh \
		       iscan.sh \
		       iscan_inplace.sh \
		       exscan.sh \
		       exscan_inplace.sh \
		       iexscan.sh \
		       iexscan_inplace.sh

###################
# One sided tests #
###################
check_PROGRAMS += put \
		  rput \
		  get \
		  rget \
		  accumulate \
		  raccumulate \
		  get_accumulate \
		  rget_accumulate \
		  fetch_and_op \
		  compare_and_swap

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

get_accumulate_SOURCES = $(srcdir)/get_accumulate.F90
get_accumulate_FCFLAGS = $(AM_FCFLAGS)
get_accumulate_LDFLAGS = $(AM_LDFLAGS)

rget_accumulate_SOURCES = $(srcdir)/rget_accumulate.F90
rget_accumulate_FCFLAGS = $(AM_FCFLAGS)
rget_accumulate_LDFLAGS = $(AM_LDFLAGS)

fetch_and_op_SOURCES = $(srcdir)/fetch_and_op.F90
fetch_and_op_FCFLAGS = $(AM_FCFLAGS)
fetch_and_op_LDFLAGS = $(AM_LDFLAGS)

compare_and_swap_SOURCES = $(srcdir)/compare_and_swap.F90
compare_and_swap_FCFLAGS = $(AM_FCFLAGS)
compare_and_swap_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += put.sh \
		       rput.sh \
		       get.sh \
		       rget.sh \
		       accumulate.sh \
		       raccumulate.sh \
		       get_accumulate.sh \
		       rget_accumulate.sh \
		       fetch_and_op.sh \
		       compare_and_swap.sh

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
