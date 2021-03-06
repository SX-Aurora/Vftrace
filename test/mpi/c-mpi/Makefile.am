if WITH_MPI
if USES_OPEN_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_openmpi.sh;
else
if USES_NEC_MPI
if ON_VECTOR_ENGINE
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necvhmpi.sh;
endif
else
if USES_INTEL_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_intelmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_genericmpi.sh;
endif
endif
endif
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_nompi.sh;
endif

AM_CFLAGS = -O3 -finstrument-functions

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

if ENABLE_OPENMP
if USES_INTEL_COMPILER
  AM_CFLAGS += -qopenmp
endif
if USES_GNU_COMPILER
  AM_CFLAGS += -fopenmp
endif
if USES_NEC_COMPILER
  AM_CFLAGS += -fopenmp
endif
endif


################
# System Tests #
################
check_PROGRAMS = init_finalize_1 \
		 init_finalize_2 \
		 init_finalize_3 \
		 init_finalize_4 \
		 ctypes \
		 derived_ctypes \
		 pcontrol
		
init_finalize_1_SOURCES = $(srcdir)/init_finalize.c
init_finalize_1_CFLAGS  = $(AM_CFLAGS)
init_finalize_1_LDFLAGS = $(AM_LDFLAGS)

init_finalize_2_SOURCES = $(srcdir)/init_finalize.c
init_finalize_2_CFLAGS  = $(AM_CFLAGS)
init_finalize_2_LDFLAGS = $(AM_LDFLAGS)

init_finalize_3_SOURCES = $(srcdir)/init_finalize.c
init_finalize_3_CFLAGS  = $(AM_CFLAGS)
init_finalize_3_LDFLAGS = $(AM_LDFLAGS)

init_finalize_4_SOURCES = $(srcdir)/init_finalize.c
init_finalize_4_CFLAGS  = $(AM_CFLAGS)
init_finalize_4_LDFLAGS = $(AM_LDFLAGS)

ctypes_SOURCES = $(srcdir)/ctypes.c
ctypes_CFLAGS  = $(AM_CFLAGS)
ctypes_LDFLAGS = $(AM_LDFLAGS)

derived_ctypes_SOURCES = $(srcdir)/derived_ctypes.c
derived_ctypes_CFLAGS  = $(AM_CFLAGS)
derived_ctypes_LDFLAGS = $(AM_LDFLAGS)

pcontrol_SOURCES = $(srcdir)/pcontrol.c
pcontrol_CFLAGS  = $(AM_CFLAGS)
pcontrol_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = init_finalize-1.sh \
                      init_finalize-2.sh \
                      init_finalize-3.sh \
                      init_finalize-4.sh \
		      ctypes.sh \
		      derived_ctypes.sh \
		      pcontrol.sh

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
		  sendrecv_replace \
		  probe \
		  iprobe

send_recv_SOURCES = $(srcdir)/send_recv.c
send_recv_CFLAGS  = $(AM_CFLAGS)
send_recv_LDFLAGS = $(AM_LDFLAGS)

isend_recv_SOURCES = $(srcdir)/isend_recv.c
isend_recv_CFLAGS  = $(AM_CFLAGS)
isend_recv_LDFLAGS = $(AM_LDFLAGS)

bsend_recv_SOURCES = $(srcdir)/bsend_recv.c
bsend_recv_CFLAGS  = $(AM_CFLAGS)
bsend_recv_LDFLAGS = $(AM_LDFLAGS)

ibsend_recv_SOURCES = $(srcdir)/ibsend_recv.c
ibsend_recv_CFLAGS  = $(AM_CFLAGS)
ibsend_recv_LDFLAGS = $(AM_LDFLAGS)

ssend_recv_SOURCES = $(srcdir)/ssend_recv.c
ssend_recv_CFLAGS  = $(AM_CFLAGS)
ssend_recv_LDFLAGS = $(AM_LDFLAGS)

issend_recv_SOURCES = $(srcdir)/issend_recv.c
issend_recv_CFLAGS  = $(AM_CFLAGS)
issend_recv_LDFLAGS = $(AM_LDFLAGS)

rsend_recv_SOURCES = $(srcdir)/rsend_recv.c
rsend_recv_CFLAGS  = $(AM_CFLAGS)
rsend_recv_LDFLAGS = $(AM_LDFLAGS)

irsend_recv_SOURCES = $(srcdir)/irsend_recv.c
irsend_recv_CFLAGS  = $(AM_CFLAGS)
irsend_recv_LDFLAGS = $(AM_LDFLAGS)

send_irecv_SOURCES = $(srcdir)/send_irecv.c
send_irecv_CFLAGS  = $(AM_CFLAGS)
send_irecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_SOURCES = $(srcdir)/sendrecv.c
sendrecv_CFLAGS  = $(AM_CFLAGS)
sendrecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_replace_SOURCES = $(srcdir)/sendrecv_replace.c
sendrecv_replace_CFLAGS  = $(AM_CFLAGS)
sendrecv_replace_LDFLAGS = $(AM_LDFLAGS)

probe_SOURCES = $(srcdir)/probe.c
probe_CFLAGS  = $(AM_CFLAGS)
probe_LDFLAGS = $(AM_LDFLAGS)

iprobe_SOURCES = $(srcdir)/iprobe.c
iprobe_CFLAGS  = $(AM_CFLAGS)
iprobe_LDFLAGS = $(AM_LDFLAGS)

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
		       sendrecv_replace.sh \
		       probe.sh \
		       iprobe.sh

########################
# Persistent P2P tests #
########################

check_PROGRAMS += start \
		  startall \
		  send_init \
		  bsend_init \
		  ssend_init \
		  rsend_init \
		  recv_init

start_SOURCES = $(srcdir)/start.c
start_CFLAGS  = $(AM_CFLAGS)
start_LDFLAGS = $(AM_LDFLAGS)

startall_SOURCES = $(srcdir)/startall.c
startall_CFLAGS  = $(AM_CFLAGS)
startall_LDFLAGS = $(AM_LDFLAGS)

send_init_SOURCES = $(srcdir)/send_init.c
send_init_CFLAGS  = $(AM_CFLAGS)
send_init_LDFLAGS = $(AM_LDFLAGS)

bsend_init_SOURCES = $(srcdir)/bsend_init.c
bsend_init_CFLAGS  = $(AM_CFLAGS)
bsend_init_LDFLAGS = $(AM_LDFLAGS)

ssend_init_SOURCES = $(srcdir)/ssend_init.c
ssend_init_CFLAGS  = $(AM_CFLAGS)
ssend_init_LDFLAGS = $(AM_LDFLAGS)

rsend_init_SOURCES = $(srcdir)/rsend_init.c
rsend_init_CFLAGS  = $(AM_CFLAGS)
rsend_init_LDFLAGS = $(AM_LDFLAGS)

recv_init_SOURCES = $(srcdir)/recv_init.c
recv_init_CFLAGS  = $(AM_CFLAGS)
recv_init_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += start.sh \
		       startall.sh \
		       send_init.sh \
		       bsend_init.sh \
		       ssend_init.sh \
		       rsend_init.sh \
		       recv_init.sh

####################
# Collective tests #
####################

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

barrier_SOURCES = $(srcdir)/barrier.c
barrier_CFLAGS = $(AM_CFLAGS)
barrier_LDFLAGS = $(AM_LDFLAGS)

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

gather_inplace_SOURCES = $(srcdir)/gather_inplace.c
gather_inplace_CFLAGS = $(AM_CFLAGS)
gather_inplace_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.c
gather_intercom_CFLAGS = $(AM_CFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

igather_SOURCES = $(srcdir)/igather.c
igather_CFLAGS = $(AM_CFLAGS)
igather_LDFLAGS = $(AM_LDFLAGS)

igather_inplace_SOURCES = $(srcdir)/igather_inplace.c
igather_inplace_CFLAGS = $(AM_CFLAGS)
igather_inplace_LDFLAGS = $(AM_LDFLAGS)

igather_intercom_SOURCES = $(srcdir)/igather_intercom.c
igather_intercom_CFLAGS = $(AM_CFLAGS)
igather_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.c
gatherv_CFLAGS = $(AM_CFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_inplace_SOURCES = $(srcdir)/gatherv_inplace.c
gatherv_inplace_CFLAGS = $(AM_CFLAGS)
gatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.c
gatherv_intercom_CFLAGS = $(AM_CFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

igatherv_SOURCES = $(srcdir)/igatherv.c
igatherv_CFLAGS = $(AM_CFLAGS)
igatherv_LDFLAGS = $(AM_LDFLAGS)

igatherv_inplace_SOURCES = $(srcdir)/igatherv_inplace.c
igatherv_inplace_CFLAGS = $(AM_CFLAGS)
igatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

igatherv_intercom_SOURCES = $(srcdir)/igatherv_intercom.c
igatherv_intercom_CFLAGS = $(AM_CFLAGS)
igatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.c
scatter_CFLAGS = $(AM_CFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_inplace_SOURCES = $(srcdir)/scatter_inplace.c
scatter_inplace_CFLAGS = $(AM_CFLAGS)
scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.c
scatter_intercom_CFLAGS = $(AM_CFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatter_SOURCES = $(srcdir)/iscatter.c
iscatter_CFLAGS = $(AM_CFLAGS)
iscatter_LDFLAGS = $(AM_LDFLAGS)

iscatter_inplace_SOURCES = $(srcdir)/iscatter_inplace.c
iscatter_inplace_CFLAGS = $(AM_CFLAGS)
iscatter_inplace_LDFLAGS = $(AM_LDFLAGS)

iscatter_intercom_SOURCES = $(srcdir)/iscatter_intercom.c
iscatter_intercom_CFLAGS = $(AM_CFLAGS)
iscatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.c
scatterv_CFLAGS = $(AM_CFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_inplace_SOURCES = $(srcdir)/scatterv_inplace.c
scatterv_inplace_CFLAGS = $(AM_CFLAGS)
scatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.c
scatterv_intercom_CFLAGS = $(AM_CFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

iscatterv_SOURCES = $(srcdir)/iscatterv.c
iscatterv_CFLAGS = $(AM_CFLAGS)
iscatterv_LDFLAGS = $(AM_LDFLAGS)

iscatterv_inplace_SOURCES = $(srcdir)/iscatterv_inplace.c
iscatterv_inplace_CFLAGS = $(AM_CFLAGS)
iscatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

iscatterv_intercom_SOURCES = $(srcdir)/iscatterv_intercom.c
iscatterv_intercom_CFLAGS = $(AM_CFLAGS)
iscatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.c
allgather_CFLAGS = $(AM_CFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_inplace_SOURCES = $(srcdir)/allgather_inplace.c
allgather_inplace_CFLAGS = $(AM_CFLAGS)
allgather_inplace_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.c
allgather_intercom_CFLAGS = $(AM_CFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgather_SOURCES = $(srcdir)/iallgather.c
iallgather_CFLAGS = $(AM_CFLAGS)
iallgather_LDFLAGS = $(AM_LDFLAGS)

iallgather_inplace_SOURCES = $(srcdir)/iallgather_inplace.c
iallgather_inplace_CFLAGS = $(AM_CFLAGS)
iallgather_inplace_LDFLAGS = $(AM_LDFLAGS)

iallgather_intercom_SOURCES = $(srcdir)/iallgather_intercom.c
iallgather_intercom_CFLAGS = $(AM_CFLAGS)
iallgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.c
allgatherv_CFLAGS = $(AM_CFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_inplace_SOURCES = $(srcdir)/allgatherv_inplace.c
allgatherv_inplace_CFLAGS = $(AM_CFLAGS)
allgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.c
allgatherv_intercom_CFLAGS = $(AM_CFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_SOURCES = $(srcdir)/iallgatherv.c
iallgatherv_CFLAGS = $(AM_CFLAGS)
iallgatherv_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_inplace_SOURCES = $(srcdir)/iallgatherv_inplace.c
iallgatherv_inplace_CFLAGS = $(AM_CFLAGS)
iallgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_intercom_SOURCES = $(srcdir)/iallgatherv_intercom.c
iallgatherv_intercom_CFLAGS = $(AM_CFLAGS)
iallgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.c
alltoall_CFLAGS = $(AM_CFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_inplace_SOURCES = $(srcdir)/alltoall_inplace.c
alltoall_inplace_CFLAGS = $(AM_CFLAGS)
alltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.c
alltoall_intercom_CFLAGS = $(AM_CFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoall_SOURCES = $(srcdir)/ialltoall.c
ialltoall_CFLAGS = $(AM_CFLAGS)
ialltoall_LDFLAGS = $(AM_LDFLAGS)

ialltoall_inplace_SOURCES = $(srcdir)/ialltoall_inplace.c
ialltoall_inplace_CFLAGS = $(AM_CFLAGS)
ialltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoall_intercom_SOURCES = $(srcdir)/ialltoall_intercom.c
ialltoall_intercom_CFLAGS = $(AM_CFLAGS)
ialltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.c
alltoallv_CFLAGS = $(AM_CFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_inplace_SOURCES = $(srcdir)/alltoallv_inplace.c
alltoallv_inplace_CFLAGS = $(AM_CFLAGS)
alltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.c
alltoallv_intercom_CFLAGS = $(AM_CFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_SOURCES = $(srcdir)/ialltoallv.c
ialltoallv_CFLAGS = $(AM_CFLAGS)
ialltoallv_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_inplace_SOURCES = $(srcdir)/ialltoallv_inplace.c
ialltoallv_inplace_CFLAGS = $(AM_CFLAGS)
ialltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_intercom_SOURCES = $(srcdir)/ialltoallv_intercom.c
ialltoallv_intercom_CFLAGS = $(AM_CFLAGS)
ialltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.c
alltoallw_CFLAGS = $(AM_CFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_inplace_SOURCES = $(srcdir)/alltoallw_inplace.c
alltoallw_inplace_CFLAGS = $(AM_CFLAGS)
alltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.c
alltoallw_intercom_CFLAGS = $(AM_CFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_SOURCES = $(srcdir)/ialltoallw.c
ialltoallw_CFLAGS = $(AM_CFLAGS)
ialltoallw_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_inplace_SOURCES = $(srcdir)/ialltoallw_inplace.c
ialltoallw_inplace_CFLAGS = $(AM_CFLAGS)
ialltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_intercom_SOURCES = $(srcdir)/ialltoallw_intercom.c
ialltoallw_intercom_CFLAGS = $(AM_CFLAGS)
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

reduce_SOURCES = $(srcdir)/reduce.c
reduce_CFLAGS = $(AM_CFLAGS)
reduce_LDFLAGS = $(AM_LDFLAGS)

reduce_inplace_SOURCES = $(srcdir)/reduce_inplace.c
reduce_inplace_CFLAGS = $(AM_CFLAGS)
reduce_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_intercom_SOURCES = $(srcdir)/reduce_intercom.c
reduce_intercom_CFLAGS = $(AM_CFLAGS)
reduce_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_SOURCES = $(srcdir)/ireduce.c
ireduce_CFLAGS = $(AM_CFLAGS)
ireduce_LDFLAGS = $(AM_LDFLAGS)

ireduce_inplace_SOURCES = $(srcdir)/ireduce_inplace.c
ireduce_inplace_CFLAGS = $(AM_CFLAGS)
ireduce_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_intercom_SOURCES = $(srcdir)/ireduce_intercom.c
ireduce_intercom_CFLAGS = $(AM_CFLAGS)
ireduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_SOURCES = $(srcdir)/allreduce.c
allreduce_CFLAGS = $(AM_CFLAGS)
allreduce_LDFLAGS = $(AM_LDFLAGS)

allreduce_inplace_SOURCES = $(srcdir)/allreduce_inplace.c
allreduce_inplace_CFLAGS = $(AM_CFLAGS)
allreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

allreduce_intercom_SOURCES = $(srcdir)/allreduce_intercom.c
allreduce_intercom_CFLAGS = $(AM_CFLAGS)
allreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

iallreduce_SOURCES = $(srcdir)/iallreduce.c
iallreduce_CFLAGS = $(AM_CFLAGS)
iallreduce_LDFLAGS = $(AM_LDFLAGS)

iallreduce_inplace_SOURCES = $(srcdir)/iallreduce_inplace.c
iallreduce_inplace_CFLAGS = $(AM_CFLAGS)
iallreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

iallreduce_intercom_SOURCES = $(srcdir)/iallreduce_intercom.c
iallreduce_intercom_CFLAGS = $(AM_CFLAGS)
iallreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_SOURCES = $(srcdir)/reduce_scatter_block.c
reduce_scatter_block_CFLAGS = $(AM_CFLAGS)
reduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_inplace_SOURCES = $(srcdir)/reduce_scatter_block_inplace.c
reduce_scatter_block_inplace_CFLAGS = $(AM_CFLAGS)
reduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_intercom_SOURCES = $(srcdir)/reduce_scatter_block_intercom.c
reduce_scatter_block_intercom_CFLAGS = $(AM_CFLAGS)
reduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_SOURCES = $(srcdir)/ireduce_scatter_block.c
ireduce_scatter_block_CFLAGS = $(AM_CFLAGS)
ireduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_inplace_SOURCES = $(srcdir)/ireduce_scatter_block_inplace.c
ireduce_scatter_block_inplace_CFLAGS = $(AM_CFLAGS)
ireduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_intercom_SOURCES = $(srcdir)/ireduce_scatter_block_intercom.c
ireduce_scatter_block_intercom_CFLAGS = $(AM_CFLAGS)
ireduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_SOURCES = $(srcdir)/reduce_scatter.c
reduce_scatter_CFLAGS = $(AM_CFLAGS)
reduce_scatter_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_inplace_SOURCES = $(srcdir)/reduce_scatter_inplace.c
reduce_scatter_inplace_CFLAGS = $(AM_CFLAGS)
reduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_intercom_SOURCES = $(srcdir)/reduce_scatter_intercom.c
reduce_scatter_intercom_CFLAGS = $(AM_CFLAGS)
reduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_SOURCES = $(srcdir)/ireduce_scatter.c
ireduce_scatter_CFLAGS = $(AM_CFLAGS)
ireduce_scatter_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_inplace_SOURCES = $(srcdir)/ireduce_scatter_inplace.c
ireduce_scatter_inplace_CFLAGS = $(AM_CFLAGS)
ireduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_intercom_SOURCES = $(srcdir)/ireduce_scatter_intercom.c
ireduce_scatter_intercom_CFLAGS = $(AM_CFLAGS)
ireduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scan_SOURCES = $(srcdir)/scan.c
scan_CFLAGS = $(AM_CFLAGS)
scan_LDFLAGS = $(AM_LDFLAGS)

scan_inplace_SOURCES = $(srcdir)/scan_inplace.c
scan_inplace_CFLAGS = $(AM_CFLAGS)
scan_inplace_LDFLAGS = $(AM_LDFLAGS)

iscan_SOURCES = $(srcdir)/iscan.c
iscan_CFLAGS = $(AM_CFLAGS)
iscan_LDFLAGS = $(AM_LDFLAGS)

iscan_inplace_SOURCES = $(srcdir)/iscan_inplace.c
iscan_inplace_CFLAGS = $(AM_CFLAGS)
iscan_inplace_LDFLAGS = $(AM_LDFLAGS)

exscan_SOURCES = $(srcdir)/exscan.c
exscan_CFLAGS = $(AM_CFLAGS)
exscan_LDFLAGS = $(AM_LDFLAGS)

exscan_inplace_SOURCES = $(srcdir)/exscan_inplace.c
exscan_inplace_CFLAGS = $(AM_CFLAGS)
exscan_inplace_LDFLAGS = $(AM_LDFLAGS)

iexscan_SOURCES = $(srcdir)/iexscan.c
iexscan_CFLAGS = $(AM_CFLAGS)
iexscan_LDFLAGS = $(AM_LDFLAGS)

iexscan_inplace_SOURCES = $(srcdir)/iexscan_inplace.c
iexscan_inplace_CFLAGS = $(AM_CFLAGS)
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

put_SOURCES = $(srcdir)/put.c
put_CFLAGS = $(AM_CFLAGS)
put_LDFLAGS = $(AM_LDFLAGS)

rput_SOURCES = $(srcdir)/rput.c
rput_CFLAGS = $(AM_CFLAGS)
rput_LDFLAGS = $(AM_LDFLAGS)

get_SOURCES = $(srcdir)/get.c
get_CFLAGS = $(AM_CFLAGS)
get_LDFLAGS = $(AM_LDFLAGS)

rget_SOURCES = $(srcdir)/rget.c
rget_CFLAGS = $(AM_CFLAGS)
rget_LDFLAGS = $(AM_LDFLAGS)

accumulate_SOURCES = $(srcdir)/accumulate.c
accumulate_CFLAGS = $(AM_CFLAGS)
accumulate_LDFLAGS = $(AM_LDFLAGS)

raccumulate_SOURCES = $(srcdir)/raccumulate.c
raccumulate_CFLAGS = $(AM_CFLAGS)
raccumulate_LDFLAGS = $(AM_LDFLAGS)

get_accumulate_SOURCES = $(srcdir)/get_accumulate.c
get_accumulate_CFLAGS = $(AM_CFLAGS)
get_accumulate_LDFLAGS = $(AM_LDFLAGS)

rget_accumulate_SOURCES = $(srcdir)/rget_accumulate.c
rget_accumulate_CFLAGS = $(AM_CFLAGS)
rget_accumulate_LDFLAGS = $(AM_LDFLAGS)

fetch_and_op_SOURCES = $(srcdir)/fetch_and_op.c
fetch_and_op_CFLAGS = $(AM_CFLAGS)
fetch_and_op_LDFLAGS = $(AM_LDFLAGS)

compare_and_swap_SOURCES = $(srcdir)/compare_and_swap.c
compare_and_swap_CFLAGS = $(AM_CFLAGS)
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
