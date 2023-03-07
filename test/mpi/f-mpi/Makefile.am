if ENABLE_MPI
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
AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/unset_vftr_env.sh;

AM_FCFLAGS = -O3 $(FINSTRUMENT_FLAG)

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

################
# System Tests #
################
check_PROGRAMS = no_init_finalize \
		 init_finalize_1 \
		 init_finalize_2 \
		 init_finalize_3 \
		 init_finalize_4 \
		 init_thread_finalize_1 \
		 init_thread_finalize_2 \
		 init_thread_finalize_3 \
		 init_thread_finalize_4 \
		 ftypes \
		 derived_ftypes \
		 pcontrol
		
no_init_finalize_SOURCES = $(srcdir)/no_init_finalize.F90
no_init_finalize_FCFLAGS = $(AM_FCFLAGS)
no_init_finalize_LDFLAGS = $(AM_LDFLAGS)

init_finalize_1_SOURCES = $(srcdir)/init_finalize.F90
init_finalize_1_FCFLAGS = $(AM_FCFLAGS)
init_finalize_1_LDFLAGS = $(AM_LDFLAGS)

init_finalize_2_SOURCES = $(srcdir)/init_finalize.F90
init_finalize_2_FCFLAGS = $(AM_FCFLAGS)
init_finalize_2_LDFLAGS = $(AM_LDFLAGS)

init_finalize_3_SOURCES = $(srcdir)/init_finalize.F90
init_finalize_3_FCFLAGS = $(AM_FCFLAGS)
init_finalize_3_LDFLAGS = $(AM_LDFLAGS)

init_finalize_4_SOURCES = $(srcdir)/init_finalize.F90
init_finalize_4_FCFLAGS = $(AM_FCFLAGS)
init_finalize_4_LDFLAGS = $(AM_LDFLAGS)

init_thread_finalize_1_SOURCES = $(srcdir)/init_thread_finalize.F90
init_thread_finalize_1_FCFLAGS = $(AM_FCFLAGS)
init_thread_finalize_1_LDFLAGS = $(AM_LDFLAGS)

init_thread_finalize_2_SOURCES = $(srcdir)/init_thread_finalize.F90
init_thread_finalize_2_FCFLAGS = $(AM_FCFLAGS)
init_thread_finalize_2_LDFLAGS = $(AM_LDFLAGS)

init_thread_finalize_3_SOURCES = $(srcdir)/init_thread_finalize.F90
init_thread_finalize_3_FCFLAGS = $(AM_FCFLAGS)
init_thread_finalize_3_LDFLAGS = $(AM_LDFLAGS)

init_thread_finalize_4_SOURCES = $(srcdir)/init_thread_finalize.F90
init_thread_finalize_4_FCFLAGS = $(AM_FCFLAGS)
init_thread_finalize_4_LDFLAGS = $(AM_LDFLAGS)

ftypes_SOURCES = $(srcdir)/ftypes.F90
ftypes_FCFLAGS = $(AM_FCFLAGS)
ftypes_LDFLAGS = $(AM_LDFLAGS)

derived_ftypes_SOURCES = $(srcdir)/derived_ftypes.F90
derived_ftypes_FCFLAGS = $(AM_FCFLAGS)
derived_ftypes_LDFLAGS = $(AM_LDFLAGS)

pcontrol_SOURCES = $(srcdir)/pcontrol.F90
pcontrol_FCFLAGS = $(AM_FCFLAGS)
pcontrol_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = no_init_finalize.sh \
		      init_finalize-1.sh \
                      init_finalize-2.sh \
                      init_finalize-3.sh \
                      init_finalize-4.sh \
		      init_thread_finalize-1.sh \
                      init_thread_finalize-2.sh \
                      init_thread_finalize-3.sh \
                      init_thread_finalize-4.sh \
		      ftypes.sh \
		      derived_ftypes.sh \
		      pcontrol.sh

########################
# point to point tests #
########################
check_PROGRAMS += send_recv \
		  send_recv_nolog \
		  isend_recv \
		  isend_recv_nolog \
		  bsend_recv \
		  bsend_recv_nolog \
		  ibsend_recv \
		  ibsend_recv_nolog \
		  ssend_recv \
		  ssend_recv_nolog \
		  issend_recv \
		  issend_recv_nolog \
		  rsend_recv \
		  rsend_recv_nolog \
		  irsend_recv \
		  irsend_recv_nolog \
		  send_irecv \
		  send_irecv_nolog \
		  sendrecv \
		  sendrecv_nolog \
		  sendrecv_replace \
		  sendrecv_replace_nolog \
		  probe \
		  iprobe

send_recv_SOURCES = $(srcdir)/send_recv.F90
send_recv_FCFLAGS = $(AM_FCFLAGS)
send_recv_LDFLAGS = $(AM_LDFLAGS)

send_recv_nolog_SOURCES = $(srcdir)/send_recv.F90
send_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
send_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

isend_recv_SOURCES = $(srcdir)/isend_recv.F90
isend_recv_FCFLAGS  = $(AM_FCFLAGS)
isend_recv_LDFLAGS = $(AM_LDFLAGS)

isend_recv_nolog_SOURCES = $(srcdir)/isend_recv.F90
isend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
isend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

bsend_recv_SOURCES = $(srcdir)/bsend_recv.F90
bsend_recv_FCFLAGS = $(AM_FCFLAGS)
bsend_recv_LDFLAGS = $(AM_LDFLAGS)

bsend_recv_nolog_SOURCES = $(srcdir)/bsend_recv.F90
bsend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
bsend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

ibsend_recv_SOURCES = $(srcdir)/ibsend_recv.F90
ibsend_recv_FCFLAGS = $(AM_FCFLAGS)
ibsend_recv_LDFLAGS = $(AM_LDFLAGS)

ibsend_recv_nolog_SOURCES = $(srcdir)/ibsend_recv.F90
ibsend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
ibsend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

ssend_recv_SOURCES = $(srcdir)/ssend_recv.F90
ssend_recv_FCFLAGS = $(AM_FCFLAGS)
ssend_recv_LDFLAGS = $(AM_LDFLAGS)

ssend_recv_nolog_SOURCES = $(srcdir)/ssend_recv.F90
ssend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
ssend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

issend_recv_SOURCES = $(srcdir)/issend_recv.F90
issend_recv_FCFLAGS = $(AM_FCFLAGS)
issend_recv_LDFLAGS = $(AM_LDFLAGS)

issend_recv_nolog_SOURCES = $(srcdir)/issend_recv.F90
issend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
issend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

rsend_recv_SOURCES = $(srcdir)/rsend_recv.F90
rsend_recv_FCFLAGS = $(AM_FCFLAGS)
rsend_recv_LDFLAGS = $(AM_LDFLAGS)

rsend_recv_nolog_SOURCES = $(srcdir)/rsend_recv.F90
rsend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
rsend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

irsend_recv_SOURCES = $(srcdir)/irsend_recv.F90
irsend_recv_FCFLAGS = $(AM_FCFLAGS)
irsend_recv_LDFLAGS = $(AM_LDFLAGS)

irsend_recv_nolog_SOURCES = $(srcdir)/irsend_recv.F90
irsend_recv_nolog_FCFLAGS = $(AM_FCFLAGS)
irsend_recv_nolog_LDFLAGS = $(AM_LDFLAGS)

send_irecv_SOURCES = $(srcdir)/send_irecv.F90
send_irecv_FCFLAGS = $(AM_FCFLAGS)
send_irecv_LDFLAGS = $(AM_LDFLAGS)

send_irecv_nolog_SOURCES = $(srcdir)/send_irecv.F90
send_irecv_nolog_FCFLAGS = $(AM_FCFLAGS)
send_irecv_nolog_LDFLAGS = $(AM_LDFLAGS)

sendrecv_SOURCES = $(srcdir)/sendrecv.F90
sendrecv_FCFLAGS = $(AM_FCFLAGS)
sendrecv_LDFLAGS = $(AM_LDFLAGS)

sendrecv_nolog_SOURCES = $(srcdir)/sendrecv.F90
sendrecv_nolog_FCFLAGS = $(AM_FCFLAGS)
sendrecv_nolog_LDFLAGS = $(AM_LDFLAGS)

sendrecv_replace_SOURCES = $(srcdir)/sendrecv_replace.F90
sendrecv_replace_FCFLAGS = $(AM_FCFLAGS)
sendrecv_replace_LDFLAGS = $(AM_LDFLAGS)

sendrecv_replace_nolog_SOURCES = $(srcdir)/sendrecv_replace.F90
sendrecv_replace_nolog_FCFLAGS = $(AM_FCFLAGS)
sendrecv_replace_nolog_LDFLAGS = $(AM_LDFLAGS)

probe_SOURCES = $(srcdir)/probe.F90
probe_FCFLAGS = $(AM_FCFLAGS)
probe_LDFLAGS = $(AM_LDFLAGS)

iprobe_SOURCES = $(srcdir)/iprobe.F90
iprobe_FCFLAGS = $(AM_FCFLAGS)
iprobe_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += send_recv.sh \
		       send_recv_nolog.sh \
		       isend_recv.sh \
		       isend_recv_nolog.sh \
		       bsend_recv.sh \
		       bsend_recv_nolog.sh \
		       ibsend_recv.sh \
		       ibsend_recv_nolog.sh \
		       ssend_recv.sh \
		       ssend_recv_nolog.sh \
		       issend_recv.sh \
		       issend_recv_nolog.sh \
		       rsend_recv.sh \
		       rsend_recv_nolog.sh \
		       irsend_recv.sh \
		       irsend_recv_nolog.sh \
		       send_irecv.sh \
		       send_irecv_nolog.sh \
		       sendrecv.sh \
		       sendrecv_nolog.sh \
		       sendrecv_replace.sh \
		       sendrecv_replace_nolog.sh \
		       probe.sh \
		       iprobe.sh

########################
# Persistent P2P tests #
########################

check_PROGRAMS += start \
		  startall \
		  send_init \
		  send_init_nolog \
		  bsend_init \
		  bsend_init_nolog \
		  ssend_init \
		  ssend_init_nolog \
		  rsend_init \
		  rsend_init_nolog \
		  recv_init \
		  recv_init_nolog

start_SOURCES = $(srcdir)/start.F90
start_FCFLAGS = $(AM_FCFLAGS)
start_LDFLAGS = $(AM_LDFLAGS)

startall_SOURCES = $(srcdir)/startall.F90
startall_FCFLAGS = $(AM_FCFLAGS)
startall_LDFLAGS = $(AM_LDFLAGS)

send_init_SOURCES = $(srcdir)/send_init.F90
send_init_FCFLAGS = $(AM_FCFLAGS)
send_init_LDFLAGS = $(AM_LDFLAGS)

send_init_nolog_SOURCES = $(srcdir)/send_init.F90
send_init_nolog_FCFLAGS = $(AM_FCFLAGS)
send_init_nolog_LDFLAGS = $(AM_LDFLAGS)

bsend_init_SOURCES = $(srcdir)/bsend_init.F90
bsend_init_FCFLAGS = $(AM_FCFLAGS)
bsend_init_LDFLAGS = $(AM_LDFLAGS)

bsend_init_nolog_SOURCES = $(srcdir)/bsend_init.F90
bsend_init_nolog_FCFLAGS = $(AM_FCFLAGS)
bsend_init_nolog_LDFLAGS = $(AM_LDFLAGS)

ssend_init_SOURCES = $(srcdir)/ssend_init.F90
ssend_init_FCFLAGS = $(AM_FCFLAGS)
ssend_init_LDFLAGS = $(AM_LDFLAGS)

ssend_init_nolog_SOURCES = $(srcdir)/ssend_init.F90
ssend_init_nolog_FCFLAGS = $(AM_FCFLAGS)
ssend_init_nolog_LDFLAGS = $(AM_LDFLAGS)

rsend_init_SOURCES = $(srcdir)/rsend_init.F90
rsend_init_FCFLAGS = $(AM_FCFLAGS)
rsend_init_LDFLAGS = $(AM_LDFLAGS)

rsend_init_nolog_SOURCES = $(srcdir)/rsend_init.F90
rsend_init_nolog_FCFLAGS = $(AM_FCFLAGS)
rsend_init_nolog_LDFLAGS = $(AM_LDFLAGS)

recv_init_SOURCES = $(srcdir)/recv_init.F90
recv_init_FCFLAGS = $(AM_FCFLAGS)
recv_init_LDFLAGS = $(AM_LDFLAGS)

recv_init_nolog_SOURCES = $(srcdir)/recv_init.F90
recv_init_nolog_FCFLAGS = $(AM_FCFLAGS)
recv_init_nolog_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += start.sh \
		       startall.sh \
		       send_init.sh \
		       send_init_nolog.sh \
		       bsend_init.sh \
		       bsend_init_nolog.sh \
		       ssend_init.sh \
		       ssend_init_nolog.sh \
		       rsend_init.sh \
		       rsend_init_nolog.sh \
		       recv_init.sh \
		       recv_init_nolog.sh

####################
# Collective tests #
####################

check_PROGRAMS += barrier \
		  ibarrier \
		  bcast \
		  bcast_nolog \
		  bcast_intercom \
		  bcast_sync_time \
		  ibcast \
		  ibcast_nolog \
		  ibcast_intercom \
		  gather \
		  gather_nolog \
		  gather_inplace \
		  gather_intercom \
		  gather_sync_time \
		  igather \
		  igather_nolog \
		  igather_inplace \
		  igather_intercom \
		  gatherv \
		  gatherv_nolog \
		  gatherv_inplace \
		  gatherv_intercom \
		  gatherv_sync_time \
		  igatherv \
		  igatherv_nolog \
		  igatherv_inplace \
		  igatherv_intercom \
		  scatter \
		  scatter_nolog \
		  scatter_inplace \
		  scatter_intercom \
		  scatter_sync_time \
		  iscatter \
		  iscatter_nolog \
		  iscatter_inplace \
		  iscatter_intercom \
		  scatterv \
		  scatterv_nolog \
		  scatterv_inplace \
		  scatterv_intercom \
		  scatterv_sync_time \
		  iscatterv \
		  iscatterv_nolog \
		  iscatterv_inplace \
		  iscatterv_intercom \
		  allgather \
		  allgather_nolog \
		  allgather_inplace \
		  allgather_intercom \
		  allgather_sync_time \
		  iallgather \
		  iallgather_nolog \
		  iallgather_inplace \
		  iallgather_intercom \
		  allgatherv \
		  allgatherv_nolog \
		  allgatherv_inplace \
		  allgatherv_intercom \
		  allgatherv_sync_time \
		  iallgatherv \
		  iallgatherv_nolog \
		  iallgatherv_inplace \
		  iallgatherv_intercom \
		  alltoall \
		  alltoall_nolog \
		  alltoall_inplace \
		  alltoall_intercom \
		  alltoall_sync_time \
		  ialltoall \
		  ialltoall_nolog \
		  ialltoall_inplace \
		  ialltoall_intercom \
		  alltoallv \
		  alltoallv_nolog \
		  alltoallv_inplace \
		  alltoallv_intercom \
		  alltoallv_sync_time \
		  ialltoallv \
		  ialltoallv_nolog \
		  ialltoallv_inplace \
		  ialltoallv_intercom \
		  alltoallw \
		  alltoallw_nolog \
		  alltoallw_inplace \
		  alltoallw_intercom \
		  alltoallw_sync_time \
		  ialltoallw \
		  ialltoallw_nolog \
		  ialltoallw_inplace \
		  ialltoallw_intercom

barrier_SOURCES = $(srcdir)/barrier.F90
barrier_FCFLAGS = $(AM_FCFLAGS)
barrier_LDFLAGS = $(AM_LDFLAGS)

ibarrier_SOURCES = $(srcdir)/ibarrier.F90
ibarrier_FCFLAGS = $(AM_FCFLAGS)
ibarrier_LDFLAGS = $(AM_LDFLAGS)

bcast_SOURCES = $(srcdir)/bcast.F90
bcast_FCFLAGS = $(AM_FCFLAGS)
bcast_LDFLAGS = $(AM_LDFLAGS)

bcast_nolog_SOURCES = $(srcdir)/bcast.F90
bcast_nolog_FCFLAGS = $(AM_FCFLAGS)
bcast_nolog_LDFLAGS = $(AM_LDFLAGS)

bcast_intercom_SOURCES = $(srcdir)/bcast_intercom.F90
bcast_intercom_FCFLAGS = $(AM_FCFLAGS)
bcast_intercom_LDFLAGS = $(AM_LDFLAGS)

bcast_sync_time_SOURCES = $(srcdir)/bcast.F90
bcast_sync_time_FCFLAGS = $(AM_FCFLAGS)
bcast_sync_time_LDFLAGS = $(AM_LDFLAGS)

ibcast_SOURCES = $(srcdir)/ibcast.F90
ibcast_FCFLAGS = $(AM_FCFLAGS)
ibcast_LDFLAGS = $(AM_LDFLAGS)

ibcast_nolog_SOURCES = $(srcdir)/ibcast.F90
ibcast_nolog_FCFLAGS = $(AM_FCFLAGS)
ibcast_nolog_LDFLAGS = $(AM_LDFLAGS)

ibcast_intercom_SOURCES = $(srcdir)/ibcast_intercom.F90
ibcast_intercom_FCFLAGS = $(AM_FCFLAGS)
ibcast_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_SOURCES = $(srcdir)/gather.F90
gather_FCFLAGS = $(AM_FCFLAGS)
gather_LDFLAGS = $(AM_LDFLAGS)

gather_nolog_SOURCES = $(srcdir)/gather.F90
gather_nolog_FCFLAGS = $(AM_FCFLAGS)
gather_nolog_LDFLAGS = $(AM_LDFLAGS)

gather_inplace_SOURCES = $(srcdir)/gather_inplace.F90
gather_inplace_FCFLAGS = $(AM_FCFLAGS)
gather_inplace_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.F90
gather_intercom_FCFLAGS = $(AM_FCFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_sync_time_SOURCES = $(srcdir)/gather.F90
gather_sync_time_FCFLAGS = $(AM_FCFLAGS)
gather_sync_time_LDFLAGS = $(AM_LDFLAGS)

igather_SOURCES = $(srcdir)/igather.F90
igather_FCFLAGS = $(AM_FCFLAGS)
igather_LDFLAGS = $(AM_LDFLAGS)

igather_nolog_SOURCES = $(srcdir)/igather.F90
igather_nolog_FCFLAGS = $(AM_FCFLAGS)
igather_nolog_LDFLAGS = $(AM_LDFLAGS)

igather_inplace_SOURCES = $(srcdir)/igather_inplace.F90
igather_inplace_FCFLAGS = $(AM_FCFLAGS)
igather_inplace_LDFLAGS = $(AM_LDFLAGS)

igather_intercom_SOURCES = $(srcdir)/igather_intercom.F90
igather_intercom_FCFLAGS = $(AM_FCFLAGS)
igather_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.F90
gatherv_FCFLAGS = $(AM_FCFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_nolog_SOURCES = $(srcdir)/gatherv.F90
gatherv_nolog_FCFLAGS = $(AM_FCFLAGS)
gatherv_nolog_LDFLAGS = $(AM_LDFLAGS)

gatherv_inplace_SOURCES = $(srcdir)/gatherv_inplace.F90
gatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
gatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.F90
gatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_sync_time_SOURCES = $(srcdir)/gatherv.F90
gatherv_sync_time_FCFLAGS = $(AM_FCFLAGS)
gatherv_sync_time_LDFLAGS = $(AM_LDFLAGS)

igatherv_SOURCES = $(srcdir)/igatherv.F90
igatherv_FCFLAGS = $(AM_FCFLAGS)
igatherv_LDFLAGS = $(AM_LDFLAGS)

igatherv_nolog_SOURCES = $(srcdir)/igatherv.F90
igatherv_nolog_FCFLAGS = $(AM_FCFLAGS)
igatherv_nolog_LDFLAGS = $(AM_LDFLAGS)

igatherv_inplace_SOURCES = $(srcdir)/igatherv_inplace.F90
igatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
igatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

igatherv_intercom_SOURCES = $(srcdir)/igatherv_intercom.F90
igatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
igatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.F90
scatter_FCFLAGS = $(AM_FCFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_nolog_SOURCES = $(srcdir)/scatter.F90
scatter_nolog_FCFLAGS = $(AM_FCFLAGS)
scatter_nolog_LDFLAGS = $(AM_LDFLAGS)

scatter_inplace_SOURCES = $(srcdir)/scatter_inplace.F90
scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.F90
scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_sync_time_SOURCES = $(srcdir)/scatter.F90
scatter_sync_time_FCFLAGS = $(AM_FCFLAGS)
scatter_sync_time_LDFLAGS = $(AM_LDFLAGS)

iscatter_SOURCES = $(srcdir)/iscatter.F90
iscatter_FCFLAGS = $(AM_FCFLAGS)
iscatter_LDFLAGS = $(AM_LDFLAGS)

iscatter_nolog_SOURCES = $(srcdir)/iscatter.F90
iscatter_nolog_FCFLAGS = $(AM_FCFLAGS)
iscatter_nolog_LDFLAGS = $(AM_LDFLAGS)

iscatter_inplace_SOURCES = $(srcdir)/iscatter_inplace.F90
iscatter_inplace_FCFLAGS = $(AM_FCFLAGS)
iscatter_inplace_LDFLAGS = $(AM_LDFLAGS)

iscatter_intercom_SOURCES = $(srcdir)/iscatter_intercom.F90
iscatter_intercom_FCFLAGS = $(AM_FCFLAGS)
iscatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.F90
scatterv_FCFLAGS = $(AM_FCFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_nolog_SOURCES = $(srcdir)/scatterv.F90
scatterv_nolog_FCFLAGS = $(AM_FCFLAGS)
scatterv_nolog_LDFLAGS = $(AM_LDFLAGS)

scatterv_inplace_SOURCES = $(srcdir)/scatterv_inplace.F90
scatterv_inplace_FCFLAGS = $(AM_FCFLAGS)
scatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.F90
scatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_sync_time_SOURCES = $(srcdir)/scatterv.F90
scatterv_sync_time_FCFLAGS = $(AM_FCFLAGS)
scatterv_sync_time_LDFLAGS = $(AM_LDFLAGS)

iscatterv_SOURCES = $(srcdir)/iscatterv.F90
iscatterv_FCFLAGS = $(AM_FCFLAGS)
iscatterv_LDFLAGS = $(AM_LDFLAGS)

iscatterv_nolog_SOURCES = $(srcdir)/iscatterv.F90
iscatterv_nolog_FCFLAGS = $(AM_FCFLAGS)
iscatterv_nolog_LDFLAGS = $(AM_LDFLAGS)

iscatterv_inplace_SOURCES = $(srcdir)/iscatterv_inplace.F90
iscatterv_inplace_FCFLAGS = $(AM_FCFLAGS)
iscatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

iscatterv_intercom_SOURCES = $(srcdir)/iscatterv_intercom.F90
iscatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
iscatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.F90
allgather_FCFLAGS = $(AM_FCFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_nolog_SOURCES = $(srcdir)/allgather.F90
allgather_nolog_FCFLAGS = $(AM_FCFLAGS)
allgather_nolog_LDFLAGS = $(AM_LDFLAGS)

allgather_inplace_SOURCES = $(srcdir)/allgather_inplace.F90
allgather_inplace_FCFLAGS = $(AM_FCFLAGS)
allgather_inplace_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.F90
allgather_intercom_FCFLAGS = $(AM_FCFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_sync_time_SOURCES = $(srcdir)/allgather.F90
allgather_sync_time_FCFLAGS = $(AM_FCFLAGS)
allgather_sync_time_LDFLAGS = $(AM_LDFLAGS)

iallgather_SOURCES = $(srcdir)/iallgather.F90
iallgather_FCFLAGS = $(AM_FCFLAGS)
iallgather_LDFLAGS = $(AM_LDFLAGS)

iallgather_nolog_SOURCES = $(srcdir)/iallgather.F90
iallgather_nolog_FCFLAGS = $(AM_FCFLAGS)
iallgather_nolog_LDFLAGS = $(AM_LDFLAGS)

iallgather_inplace_SOURCES = $(srcdir)/iallgather_inplace.F90
iallgather_inplace_FCFLAGS = $(AM_FCFLAGS)
iallgather_inplace_LDFLAGS = $(AM_LDFLAGS)

iallgather_intercom_SOURCES = $(srcdir)/iallgather_intercom.F90
iallgather_intercom_FCFLAGS = $(AM_FCFLAGS)
iallgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.F90
allgatherv_FCFLAGS = $(AM_FCFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_nolog_SOURCES = $(srcdir)/allgatherv.F90
allgatherv_nolog_FCFLAGS = $(AM_FCFLAGS)
allgatherv_nolog_LDFLAGS = $(AM_LDFLAGS)

allgatherv_inplace_SOURCES = $(srcdir)/allgatherv_inplace.F90
allgatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
allgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.F90
allgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_sync_time_SOURCES = $(srcdir)/allgatherv.F90
allgatherv_sync_time_FCFLAGS = $(AM_FCFLAGS)
allgatherv_sync_time_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_SOURCES = $(srcdir)/iallgatherv.F90
iallgatherv_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_nolog_SOURCES = $(srcdir)/iallgatherv.F90
iallgatherv_nolog_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_nolog_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_inplace_SOURCES = $(srcdir)/iallgatherv_inplace.F90
iallgatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

iallgatherv_intercom_SOURCES = $(srcdir)/iallgatherv_intercom.F90
iallgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
iallgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.F90
alltoall_FCFLAGS = $(AM_FCFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_nolog_SOURCES = $(srcdir)/alltoall.F90
alltoall_nolog_FCFLAGS = $(AM_FCFLAGS)
alltoall_nolog_LDFLAGS = $(AM_LDFLAGS)

alltoall_inplace_SOURCES = $(srcdir)/alltoall_inplace.F90
alltoall_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.F90
alltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_sync_time_SOURCES = $(srcdir)/alltoall.F90
alltoall_sync_time_FCFLAGS = $(AM_FCFLAGS)
alltoall_sync_time_LDFLAGS = $(AM_LDFLAGS)

ialltoall_SOURCES = $(srcdir)/ialltoall.F90
ialltoall_FCFLAGS = $(AM_FCFLAGS)
ialltoall_LDFLAGS = $(AM_LDFLAGS)

ialltoall_nolog_SOURCES = $(srcdir)/ialltoall.F90
ialltoall_nolog_FCFLAGS = $(AM_FCFLAGS)
ialltoall_nolog_LDFLAGS = $(AM_LDFLAGS)

ialltoall_inplace_SOURCES = $(srcdir)/ialltoall_inplace.F90
ialltoall_inplace_FCFLAGS = $(AM_FCFLAGS)
ialltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoall_intercom_SOURCES = $(srcdir)/ialltoall_intercom.F90
ialltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.F90
alltoallv_FCFLAGS = $(AM_FCFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_nolog_SOURCES = $(srcdir)/alltoallv.F90
alltoallv_nolog_FCFLAGS = $(AM_FCFLAGS)
alltoallv_nolog_LDFLAGS = $(AM_LDFLAGS)

alltoallv_inplace_SOURCES = $(srcdir)/alltoallv_inplace.F90
alltoallv_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.F90
alltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_sync_time_SOURCES = $(srcdir)/alltoallv.F90
alltoallv_sync_time_FCFLAGS = $(AM_FCFLAGS)
alltoallv_sync_time_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_SOURCES = $(srcdir)/ialltoallv.F90
ialltoallv_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_nolog_SOURCES = $(srcdir)/ialltoallv.F90
ialltoallv_nolog_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_nolog_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_inplace_SOURCES = $(srcdir)/ialltoallv_inplace.F90
ialltoallv_inplace_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoallv_intercom_SOURCES = $(srcdir)/ialltoallv_intercom.F90
ialltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.F90
alltoallw_FCFLAGS = $(AM_FCFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_nolog_SOURCES = $(srcdir)/alltoallw.F90
alltoallw_nolog_FCFLAGS = $(AM_FCFLAGS)
alltoallw_nolog_LDFLAGS = $(AM_LDFLAGS)

alltoallw_inplace_SOURCES = $(srcdir)/alltoallw_inplace.F90
alltoallw_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.F90
alltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_sync_time_SOURCES = $(srcdir)/alltoallw.F90
alltoallw_sync_time_FCFLAGS = $(AM_FCFLAGS)
alltoallw_sync_time_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_SOURCES = $(srcdir)/ialltoallw.F90
ialltoallw_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_nolog_SOURCES = $(srcdir)/ialltoallw.F90
ialltoallw_nolog_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_nolog_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_inplace_SOURCES = $(srcdir)/ialltoallw_inplace.F90
ialltoallw_inplace_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

ialltoallw_intercom_SOURCES = $(srcdir)/ialltoallw_intercom.F90
ialltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
ialltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += barrier.sh \
		       ibarrier.sh \
		       bcast.sh \
		       bcast_nolog.sh \
		       bcast_intercom.sh \
		       bcast_sync_time.sh \
		       ibcast.sh \
		       ibcast_nolog.sh \
		       ibcast_intercom.sh \
		       gather.sh \
		       gather_nolog.sh \
		       gather_inplace.sh \
		       gather_intercom.sh \
		       gather_sync_time.sh \
		       igather.sh \
		       igather_nolog.sh \
		       igather_inplace.sh \
		       igather_intercom.sh \
		       gatherv.sh \
		       gatherv_nolog.sh \
		       gatherv_inplace.sh \
		       gatherv_intercom.sh \
		       gatherv_sync_time.sh \
		       igatherv.sh \
		       igatherv_nolog.sh \
		       igatherv_inplace.sh \
		       igatherv_intercom.sh \
		       scatter.sh \
		       scatter_nolog.sh \
		       scatter_inplace.sh \
		       scatter_intercom.sh \
		       scatter_sync_time.sh \
		       iscatter.sh \
		       iscatter_nolog.sh \
		       iscatter_inplace.sh \
		       iscatter_intercom.sh \
		       scatterv.sh \
		       scatterv_nolog.sh \
		       scatterv_inplace.sh \
		       scatterv_intercom.sh \
		       scatterv_sync_time.sh \
		       iscatterv.sh \
		       iscatterv_nolog.sh \
		       iscatterv_inplace.sh \
		       iscatterv_intercom.sh \
		       allgather.sh \
		       allgather_nolog.sh \
		       allgather_inplace.sh \
		       allgather_intercom.sh \
		       allgather_sync_time.sh \
		       iallgather.sh \
		       iallgather_nolog.sh \
		       iallgather_inplace.sh \
		       iallgather_intercom.sh \
		       allgatherv.sh \
		       allgatherv_nolog.sh \
		       allgatherv_inplace.sh \
		       allgatherv_intercom.sh \
		       allgatherv_sync_time.sh \
		       iallgatherv.sh \
		       iallgatherv_nolog.sh \
		       iallgatherv_inplace.sh \
		       iallgatherv_intercom.sh \
		       alltoall.sh \
		       alltoall_nolog.sh \
		       alltoall_inplace.sh \
		       alltoall_intercom.sh \
		       alltoall_sync_time.sh \
		       ialltoall.sh \
		       ialltoall_nolog.sh \
		       ialltoall_inplace.sh \
		       ialltoall_intercom.sh \
		       alltoallv.sh \
		       alltoallv_nolog.sh \
		       alltoallv_inplace.sh \
		       alltoallv_intercom.sh \
		       alltoallv_sync_time.sh \
		       ialltoallv.sh \
		       ialltoallv_nolog.sh \
		       ialltoallv_inplace.sh \
		       ialltoallv_intercom.sh \
		       alltoallw.sh \
		       alltoallw_nolog.sh \
		       alltoallw_inplace.sh \
		       alltoallw_intercom.sh \
		       alltoallw_sync_time.sh \
		       ialltoallw.sh \
		       ialltoallw_nolog.sh \
		       ialltoallw_inplace.sh \
		       ialltoallw_intercom.sh

########################
# Neighbor Collectives #
########################
check_PROGRAMS += neighbor_allgather_nolog \
		  neighbor_allgather_sync_time \
		  neighbor_allgather_cart \
		  ineighbor_allgather_nolog \
		  ineighbor_allgather_cart \
		  neighbor_allgather_cart_periodic \
		  ineighbor_allgather_cart_periodic \
		  neighbor_allgather_graph \
		  ineighbor_allgather_graph \
		  neighbor_allgather_dist_graph \
		  ineighbor_allgather_dist_graph \
		  neighbor_allgatherv_nolog \
		  neighbor_allgatherv_sync_time \
		  neighbor_allgatherv_cart \
		  ineighbor_allgatherv_nolog \
		  ineighbor_allgatherv_cart \
		  neighbor_allgatherv_cart_periodic \
		  ineighbor_allgatherv_cart_periodic \
		  neighbor_allgatherv_graph \
		  ineighbor_allgatherv_graph \
		  neighbor_allgatherv_dist_graph \
		  ineighbor_allgatherv_dist_graph \
		  neighbor_alltoall_nolog \
		  neighbor_alltoall_sync_time \
		  neighbor_alltoall_cart \
		  ineighbor_alltoall_nolog \
		  ineighbor_alltoall_cart \
		  neighbor_alltoall_cart_periodic \
		  ineighbor_alltoall_cart_periodic \
		  neighbor_alltoall_graph \
		  ineighbor_alltoall_graph \
		  neighbor_alltoall_dist_graph \
		  ineighbor_alltoall_dist_graph \
		  neighbor_alltoallv_nolog \
		  neighbor_alltoallv_sync_time \
		  neighbor_alltoallv_cart \
		  ineighbor_alltoallv_nolog \
		  ineighbor_alltoallv_cart \
		  neighbor_alltoallv_cart_periodic \
		  ineighbor_alltoallv_cart_periodic \
		  neighbor_alltoallv_graph \
		  ineighbor_alltoallv_graph \
		  neighbor_alltoallv_dist_graph \
		  ineighbor_alltoallv_dist_graph \
		  neighbor_alltoallw_nolog \
		  neighbor_alltoallw_sync_time \
		  neighbor_alltoallw_cart \
		  ineighbor_alltoallw_nolog \
		  ineighbor_alltoallw_cart \
		  neighbor_alltoallw_cart_periodic \
		  ineighbor_alltoallw_cart_periodic \
		  neighbor_alltoallw_graph \
		  ineighbor_alltoallw_graph \
		  neighbor_alltoallw_dist_graph \
		  ineighbor_alltoallw_dist_graph 

neighbor_allgather_nolog_SOURCES = $(srcdir)/neighbor_allgather_cart.F90
neighbor_allgather_nolog_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgather_nolog_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgather_sync_time_SOURCES = $(srcdir)/neighbor_allgather_cart.F90
neighbor_allgather_sync_time_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgather_sync_time_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgather_cart_SOURCES = $(srcdir)/neighbor_allgather_cart.F90
neighbor_allgather_cart_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgather_cart_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgather_nolog_SOURCES = $(srcdir)/ineighbor_allgather_cart.F90
ineighbor_allgather_nolog_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgather_nolog_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgather_cart_SOURCES = $(srcdir)/ineighbor_allgather_cart.F90
ineighbor_allgather_cart_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgather_cart_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgather_cart_periodic_SOURCES = $(srcdir)/neighbor_allgather_cart_periodic.F90
neighbor_allgather_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgather_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgather_cart_periodic_SOURCES = $(srcdir)/ineighbor_allgather_cart_periodic.F90
ineighbor_allgather_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgather_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgather_graph_SOURCES = $(srcdir)/neighbor_allgather_graph.F90
neighbor_allgather_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgather_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgather_graph_SOURCES = $(srcdir)/ineighbor_allgather_graph.F90
ineighbor_allgather_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgather_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgather_dist_graph_SOURCES = $(srcdir)/neighbor_allgather_dist_graph.F90
neighbor_allgather_dist_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgather_dist_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgather_dist_graph_SOURCES = $(srcdir)/ineighbor_allgather_dist_graph.F90
ineighbor_allgather_dist_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgather_dist_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgatherv_nolog_SOURCES = $(srcdir)/neighbor_allgatherv_cart.F90
neighbor_allgatherv_nolog_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgatherv_nolog_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgatherv_sync_time_SOURCES = $(srcdir)/neighbor_allgatherv_cart.F90
neighbor_allgatherv_sync_time_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgatherv_sync_time_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgatherv_cart_SOURCES = $(srcdir)/neighbor_allgatherv_cart.F90
neighbor_allgatherv_cart_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgatherv_cart_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgatherv_nolog_SOURCES = $(srcdir)/ineighbor_allgatherv_cart.F90
ineighbor_allgatherv_nolog_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgatherv_nolog_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgatherv_cart_SOURCES = $(srcdir)/ineighbor_allgatherv_cart.F90
ineighbor_allgatherv_cart_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgatherv_cart_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgatherv_cart_periodic_SOURCES = $(srcdir)/neighbor_allgatherv_cart_periodic.F90
neighbor_allgatherv_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgatherv_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgatherv_cart_periodic_SOURCES = $(srcdir)/ineighbor_allgatherv_cart_periodic.F90
ineighbor_allgatherv_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgatherv_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgatherv_graph_SOURCES = $(srcdir)/neighbor_allgatherv_graph.F90
neighbor_allgatherv_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgatherv_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgatherv_graph_SOURCES = $(srcdir)/ineighbor_allgatherv_graph.F90
ineighbor_allgatherv_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgatherv_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_allgatherv_dist_graph_SOURCES = $(srcdir)/neighbor_allgatherv_dist_graph.F90
neighbor_allgatherv_dist_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_allgatherv_dist_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_allgatherv_dist_graph_SOURCES = $(srcdir)/ineighbor_allgatherv_dist_graph.F90
ineighbor_allgatherv_dist_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_allgatherv_dist_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoall_nolog_SOURCES = $(srcdir)/neighbor_alltoall_cart.F90
neighbor_alltoall_nolog_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoall_nolog_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoall_sync_time_SOURCES = $(srcdir)/neighbor_alltoall_cart.F90
neighbor_alltoall_sync_time_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoall_sync_time_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoall_cart_SOURCES = $(srcdir)/neighbor_alltoall_cart.F90
neighbor_alltoall_cart_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoall_cart_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoall_nolog_SOURCES = $(srcdir)/ineighbor_alltoall_cart.F90
ineighbor_alltoall_nolog_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoall_nolog_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoall_cart_SOURCES = $(srcdir)/ineighbor_alltoall_cart.F90
ineighbor_alltoall_cart_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoall_cart_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoall_cart_periodic_SOURCES = $(srcdir)/neighbor_alltoall_cart_periodic.F90
neighbor_alltoall_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoall_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoall_cart_periodic_SOURCES = $(srcdir)/ineighbor_alltoall_cart_periodic.F90
ineighbor_alltoall_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoall_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoall_graph_SOURCES = $(srcdir)/neighbor_alltoall_graph.F90
neighbor_alltoall_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoall_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoall_graph_SOURCES = $(srcdir)/ineighbor_alltoall_graph.F90
ineighbor_alltoall_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoall_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoall_dist_graph_SOURCES = $(srcdir)/neighbor_alltoall_dist_graph.F90
neighbor_alltoall_dist_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoall_dist_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoall_dist_graph_SOURCES = $(srcdir)/ineighbor_alltoall_dist_graph.F90
ineighbor_alltoall_dist_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoall_dist_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallv_nolog_SOURCES = $(srcdir)/neighbor_alltoallv_cart.F90
neighbor_alltoallv_nolog_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallv_nolog_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallv_sync_time_SOURCES = $(srcdir)/neighbor_alltoallv_cart.F90
neighbor_alltoallv_sync_time_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallv_sync_time_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallv_cart_SOURCES = $(srcdir)/neighbor_alltoallv_cart.F90
neighbor_alltoallv_cart_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallv_cart_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallv_nolog_SOURCES = $(srcdir)/ineighbor_alltoallv_cart.F90
ineighbor_alltoallv_nolog_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallv_nolog_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallv_cart_SOURCES = $(srcdir)/ineighbor_alltoallv_cart.F90
ineighbor_alltoallv_cart_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallv_cart_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallv_cart_periodic_SOURCES = $(srcdir)/neighbor_alltoallv_cart_periodic.F90
neighbor_alltoallv_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallv_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallv_cart_periodic_SOURCES = $(srcdir)/ineighbor_alltoallv_cart_periodic.F90
ineighbor_alltoallv_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallv_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallv_graph_SOURCES = $(srcdir)/neighbor_alltoallv_graph.F90
neighbor_alltoallv_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallv_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallv_graph_SOURCES = $(srcdir)/ineighbor_alltoallv_graph.F90
ineighbor_alltoallv_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallv_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallv_dist_graph_SOURCES = $(srcdir)/neighbor_alltoallv_dist_graph.F90
neighbor_alltoallv_dist_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallv_dist_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallv_dist_graph_SOURCES = $(srcdir)/ineighbor_alltoallv_dist_graph.F90
ineighbor_alltoallv_dist_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallv_dist_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallw_nolog_SOURCES = $(srcdir)/neighbor_alltoallw_cart.F90
neighbor_alltoallw_nolog_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallw_nolog_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallw_sync_time_SOURCES = $(srcdir)/neighbor_alltoallw_cart.F90
neighbor_alltoallw_sync_time_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallw_sync_time_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallw_cart_SOURCES = $(srcdir)/neighbor_alltoallw_cart.F90
neighbor_alltoallw_cart_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallw_cart_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallw_nolog_SOURCES = $(srcdir)/ineighbor_alltoallw_cart.F90
ineighbor_alltoallw_nolog_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallw_nolog_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallw_cart_SOURCES = $(srcdir)/ineighbor_alltoallw_cart.F90
ineighbor_alltoallw_cart_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallw_cart_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallw_cart_periodic_SOURCES = $(srcdir)/neighbor_alltoallw_cart_periodic.F90
neighbor_alltoallw_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallw_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallw_cart_periodic_SOURCES = $(srcdir)/ineighbor_alltoallw_cart_periodic.F90
ineighbor_alltoallw_cart_periodic_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallw_cart_periodic_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallw_graph_SOURCES = $(srcdir)/neighbor_alltoallw_graph.F90
neighbor_alltoallw_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallw_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallw_graph_SOURCES = $(srcdir)/ineighbor_alltoallw_graph.F90
ineighbor_alltoallw_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallw_graph_LDFLAGS = $(AM_LDFLAGS)

neighbor_alltoallw_dist_graph_SOURCES = $(srcdir)/neighbor_alltoallw_dist_graph.F90
neighbor_alltoallw_dist_graph_FCFLAGS = $(AM_FCFLAGS)
neighbor_alltoallw_dist_graph_LDFLAGS = $(AM_LDFLAGS)

ineighbor_alltoallw_dist_graph_SOURCES = $(srcdir)/ineighbor_alltoallw_dist_graph.F90
ineighbor_alltoallw_dist_graph_FCFLAGS = $(AM_FCFLAGS)
ineighbor_alltoallw_dist_graph_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += neighbor_allgather_nolog.sh \
		       neighbor_allgather_sync_time.sh \
		       neighbor_allgather_cart.sh \
		       ineighbor_allgather_nolog.sh \
		       ineighbor_allgather_cart.sh \
		       neighbor_allgather_cart_periodic.sh \
		       ineighbor_allgather_cart_periodic.sh \
		       neighbor_allgather_graph.sh \
		       ineighbor_allgather_graph.sh \
		       neighbor_allgather_dist_graph.sh \
		       ineighbor_allgather_dist_graph.sh \
		       neighbor_allgatherv_nolog.sh \
		       neighbor_allgatherv_sync_time.sh \
		       neighbor_allgatherv_cart.sh \
		       ineighbor_allgatherv_nolog.sh \
		       ineighbor_allgatherv_cart.sh \
		       neighbor_allgatherv_cart_periodic.sh \
		       ineighbor_allgatherv_cart_periodic.sh \
		       neighbor_allgatherv_graph.sh \
		       ineighbor_allgatherv_graph.sh \
		       neighbor_allgatherv_dist_graph.sh \
		       ineighbor_allgatherv_dist_graph.sh \
		       neighbor_alltoall_nolog.sh \
		       neighbor_alltoall_sync_time.sh \
		       neighbor_alltoall_cart.sh \
		       ineighbor_alltoall_nolog.sh \
		       ineighbor_alltoall_cart.sh \
		       neighbor_alltoall_cart_periodic.sh \
		       ineighbor_alltoall_cart_periodic.sh \
		       neighbor_alltoall_graph.sh \
		       ineighbor_alltoall_graph.sh \
		       neighbor_alltoall_dist_graph.sh \
		       ineighbor_alltoall_dist_graph.sh \
		       neighbor_alltoallv_sync_time.sh \
		       neighbor_alltoallv_nolog.sh \
		       neighbor_alltoallv_cart.sh \
		       ineighbor_alltoallv_nolog.sh \
		       ineighbor_alltoallv_cart.sh \
		       neighbor_alltoallv_cart_periodic.sh \
		       ineighbor_alltoallv_cart_periodic.sh \
		       neighbor_alltoallv_graph.sh \
		       ineighbor_alltoallv_graph.sh \
		       neighbor_alltoallv_dist_graph.sh \
		       ineighbor_alltoallv_dist_graph.sh \
		       neighbor_alltoallw_nolog.sh \
		       neighbor_alltoallw_sync_time.sh \
		       neighbor_alltoallw_cart.sh \
		       ineighbor_alltoallw_nolog.sh \
		       ineighbor_alltoallw_cart.sh \
		       neighbor_alltoallw_cart_periodic.sh \
		       ineighbor_alltoallw_cart_periodic.sh \
		       neighbor_alltoallw_graph.sh \
		       ineighbor_alltoallw_graph.sh \
		       neighbor_alltoallw_dist_graph.sh \
		       ineighbor_alltoallw_dist_graph.sh

#####################
# Global reductions #
#####################

check_PROGRAMS += reduce \
		  reduce_nolog \
		  reduce_inplace \
		  reduce_intercom \
		  reduce_sync_time \
		  ireduce \
		  ireduce_nolog \
		  ireduce_inplace \
		  ireduce_intercom \
		  allreduce \
		  allreduce_nolog \
		  allreduce_inplace \
		  allreduce_intercom \
		  allreduce_sync_time \
		  iallreduce \
		  iallreduce_nolog \
		  iallreduce_inplace \
		  iallreduce_intercom \
		  reduce_scatter_block \
		  reduce_scatter_block_nolog \
		  reduce_scatter_block_inplace \
		  reduce_scatter_block_intercom \
		  reduce_scatter_block_sync_time \
		  ireduce_scatter_block \
		  ireduce_scatter_block_nolog \
		  ireduce_scatter_block_inplace \
		  ireduce_scatter_block_intercom \
		  reduce_scatter \
		  reduce_scatter_nolog \
		  reduce_scatter_inplace \
		  reduce_scatter_intercom \
		  reduce_scatter_sync_time \
		  ireduce_scatter \
		  ireduce_scatter_nolog \
		  ireduce_scatter_inplace \
		  ireduce_scatter_intercom \
		  scan \
		  scan_nolog \
		  scan_inplace \
		  scan_sync_time \
		  iscan \
		  iscan_nolog \
		  iscan_inplace \
		  exscan \
		  exscan_nolog \
		  exscan_inplace \
		  exscan_sync_time \
		  iexscan \
		  iexscan_nolog \
		  iexscan_inplace

reduce_SOURCES = $(srcdir)/reduce.F90
reduce_FCFLAGS = $(AM_FCFLAGS)
reduce_LDFLAGS = $(AM_LDFLAGS)

reduce_nolog_SOURCES = $(srcdir)/reduce.F90
reduce_nolog_FCFLAGS = $(AM_FCFLAGS)
reduce_nolog_LDFLAGS = $(AM_LDFLAGS)

reduce_inplace_SOURCES = $(srcdir)/reduce_inplace.F90
reduce_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_intercom_SOURCES = $(srcdir)/reduce_intercom.F90
reduce_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_sync_time_SOURCES = $(srcdir)/reduce.F90
reduce_sync_time_FCFLAGS = $(AM_FCFLAGS)
reduce_sync_time_LDFLAGS = $(AM_LDFLAGS)

ireduce_SOURCES = $(srcdir)/ireduce.F90
ireduce_FCFLAGS = $(AM_FCFLAGS)
ireduce_LDFLAGS = $(AM_LDFLAGS)

ireduce_nolog_SOURCES = $(srcdir)/ireduce.F90
ireduce_nolog_FCFLAGS = $(AM_FCFLAGS)
ireduce_nolog_LDFLAGS = $(AM_LDFLAGS)

ireduce_inplace_SOURCES = $(srcdir)/ireduce_inplace.F90
ireduce_inplace_FCFLAGS = $(AM_FCFLAGS)
ireduce_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_intercom_SOURCES = $(srcdir)/ireduce_intercom.F90
ireduce_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_SOURCES = $(srcdir)/allreduce.F90
allreduce_FCFLAGS = $(AM_FCFLAGS)
allreduce_LDFLAGS = $(AM_LDFLAGS)

allreduce_nolog_SOURCES = $(srcdir)/allreduce.F90
allreduce_nolog_FCFLAGS = $(AM_FCFLAGS)
allreduce_nolog_LDFLAGS = $(AM_LDFLAGS)

allreduce_inplace_SOURCES = $(srcdir)/allreduce_inplace.F90
allreduce_inplace_FCFLAGS = $(AM_FCFLAGS)
allreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

allreduce_intercom_SOURCES = $(srcdir)/allreduce_intercom.F90
allreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
allreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_sync_time_SOURCES = $(srcdir)/allreduce.F90
allreduce_sync_time_FCFLAGS = $(AM_FCFLAGS)
allreduce_sync_time_LDFLAGS = $(AM_LDFLAGS)

iallreduce_SOURCES = $(srcdir)/iallreduce.F90
iallreduce_FCFLAGS = $(AM_FCFLAGS)
iallreduce_LDFLAGS = $(AM_LDFLAGS)

iallreduce_nolog_SOURCES = $(srcdir)/iallreduce.F90
iallreduce_nolog_FCFLAGS = $(AM_FCFLAGS)
iallreduce_nolog_LDFLAGS = $(AM_LDFLAGS)

iallreduce_inplace_SOURCES = $(srcdir)/iallreduce_inplace.F90
iallreduce_inplace_FCFLAGS = $(AM_FCFLAGS)
iallreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

iallreduce_intercom_SOURCES = $(srcdir)/iallreduce_intercom.F90
iallreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
iallreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_SOURCES = $(srcdir)/reduce_scatter_block.F90
reduce_scatter_block_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_nolog_SOURCES = $(srcdir)/reduce_scatter_block.F90
reduce_scatter_block_nolog_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_nolog_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_inplace_SOURCES = $(srcdir)/reduce_scatter_block_inplace.F90
reduce_scatter_block_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_intercom_SOURCES = $(srcdir)/reduce_scatter_block_intercom.F90
reduce_scatter_block_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_sync_time_SOURCES = $(srcdir)/reduce_scatter_block.F90
reduce_scatter_block_sync_time_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_sync_time_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_SOURCES = $(srcdir)/ireduce_scatter_block.F90
ireduce_scatter_block_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_nolog_SOURCES = $(srcdir)/ireduce_scatter_block.F90
ireduce_scatter_block_nolog_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_nolog_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_inplace_SOURCES = $(srcdir)/ireduce_scatter_block_inplace.F90
ireduce_scatter_block_inplace_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_block_intercom_SOURCES = $(srcdir)/ireduce_scatter_block_intercom.F90
ireduce_scatter_block_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_SOURCES = $(srcdir)/reduce_scatter.F90
reduce_scatter_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_nolog_SOURCES = $(srcdir)/reduce_scatter.F90
reduce_scatter_nolog_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_nolog_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_inplace_SOURCES = $(srcdir)/reduce_scatter_inplace.F90
reduce_scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_intercom_SOURCES = $(srcdir)/reduce_scatter_intercom.F90
reduce_scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_sync_time_SOURCES = $(srcdir)/reduce_scatter.F90
reduce_scatter_sync_time_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_sync_time_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_SOURCES = $(srcdir)/ireduce_scatter.F90
ireduce_scatter_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_nolog_SOURCES = $(srcdir)/ireduce_scatter.F90
ireduce_scatter_nolog_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_nolog_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_inplace_SOURCES = $(srcdir)/ireduce_scatter_inplace.F90
ireduce_scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

ireduce_scatter_intercom_SOURCES = $(srcdir)/ireduce_scatter_intercom.F90
ireduce_scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
ireduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scan_SOURCES = $(srcdir)/scan.F90
scan_FCFLAGS = $(AM_FCFLAGS)
scan_LDFLAGS = $(AM_LDFLAGS)

scan_nolog_SOURCES = $(srcdir)/scan.F90
scan_nolog_FCFLAGS = $(AM_FCFLAGS)
scan_nolog_LDFLAGS = $(AM_LDFLAGS)

scan_inplace_SOURCES = $(srcdir)/scan_inplace.F90
scan_inplace_FCFLAGS = $(AM_FCFLAGS)
scan_inplace_LDFLAGS = $(AM_LDFLAGS)

scan_sync_time_SOURCES = $(srcdir)/scan.F90
scan_sync_time_FCFLAGS = $(AM_FCFLAGS)
scan_sync_time_LDFLAGS = $(AM_LDFLAGS)

iscan_SOURCES = $(srcdir)/iscan.F90
iscan_FCFLAGS = $(AM_FCFLAGS)
iscan_LDFLAGS = $(AM_LDFLAGS)

iscan_nolog_SOURCES = $(srcdir)/iscan.F90
iscan_nolog_FCFLAGS = $(AM_FCFLAGS)
iscan_nolog_LDFLAGS = $(AM_LDFLAGS)

iscan_inplace_SOURCES = $(srcdir)/iscan_inplace.F90
iscan_inplace_FCFLAGS = $(AM_FCFLAGS)
iscan_inplace_LDFLAGS = $(AM_LDFLAGS)

exscan_SOURCES = $(srcdir)/exscan.F90
exscan_FCFLAGS = $(AM_FCFLAGS)
exscan_LDFLAGS = $(AM_LDFLAGS)

exscan_nolog_SOURCES = $(srcdir)/exscan.F90
exscan_nolog_FCFLAGS = $(AM_FCFLAGS)
exscan_nolog_LDFLAGS = $(AM_LDFLAGS)

exscan_inplace_SOURCES = $(srcdir)/exscan_inplace.F90
exscan_inplace_FCFLAGS = $(AM_FCFLAGS)
exscan_inplace_LDFLAGS = $(AM_LDFLAGS)

exscan_sync_time_SOURCES = $(srcdir)/exscan.F90
exscan_sync_time_FCFLAGS = $(AM_FCFLAGS)
exscan_sync_time_LDFLAGS = $(AM_LDFLAGS)

iexscan_SOURCES = $(srcdir)/iexscan.F90
iexscan_FCFLAGS = $(AM_FCFLAGS)
iexscan_LDFLAGS = $(AM_LDFLAGS)

iexscan_nolog_SOURCES = $(srcdir)/iexscan.F90
iexscan_nolog_FCFLAGS = $(AM_FCFLAGS)
iexscan_nolog_LDFLAGS = $(AM_LDFLAGS)

iexscan_inplace_SOURCES = $(srcdir)/iexscan_inplace.F90
iexscan_inplace_FCFLAGS = $(AM_FCFLAGS)
iexscan_inplace_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += reduce.sh \
		       reduce_nolog.sh \
		       reduce_inplace.sh \
		       reduce_intercom.sh \
		       reduce_sync_time.sh \
		       ireduce.sh \
		       ireduce_nolog.sh \
		       ireduce_inplace.sh \
		       ireduce_intercom.sh \
		       allreduce.sh \
		       allreduce_nolog.sh \
		       allreduce_inplace.sh \
		       allreduce_intercom.sh \
		       allreduce_sync_time.sh \
		       iallreduce.sh \
		       iallreduce_nolog.sh \
		       iallreduce_inplace.sh \
		       iallreduce_intercom.sh \
		       reduce_scatter_block.sh \
		       reduce_scatter_block_nolog.sh \
		       reduce_scatter_block_inplace.sh \
		       reduce_scatter_block_intercom.sh \
		       reduce_scatter_block_sync_time.sh \
		       ireduce_scatter_block.sh \
		       ireduce_scatter_block_nolog.sh \
		       ireduce_scatter_block_inplace.sh \
		       ireduce_scatter_block_intercom.sh \
		       reduce_scatter.sh \
		       reduce_scatter_nolog.sh \
		       reduce_scatter_inplace.sh \
		       reduce_scatter_intercom.sh \
		       reduce_scatter_sync_time.sh \
		       ireduce_scatter.sh \
		       ireduce_scatter_nolog.sh \
		       ireduce_scatter_inplace.sh \
		       ireduce_scatter_intercom.sh \
		       scan.sh \
		       scan_nolog.sh \
		       scan_inplace.sh \
		       scan_sync_time.sh \
		       iscan.sh \
		       iscan_nolog.sh \
		       iscan_inplace.sh \
		       exscan.sh \
		       exscan_nolog.sh \
		       exscan_inplace.sh \
		       exscan_sync_time.sh \
		       iexscan.sh \
		       iexscan_nolog.sh \
		       iexscan_inplace.sh

###################
# One sided tests #
###################
check_PROGRAMS += put \
		  put_nolog \
		  rput \
		  rput_nolog \
		  get \
		  get_nolog \
		  rget \
		  rget_nolog \
		  accumulate \
		  accumulate_nolog \
		  raccumulate \
		  raccumulate_nolog \
		  get_accumulate \
		  get_accumulate_nolog \
		  rget_accumulate \
		  rget_accumulate_nolog \
		  fetch_and_op \
		  fetch_and_op_nolog \
		  compare_and_swap \
		  compare_and_swap_nolog

put_SOURCES = $(srcdir)/put.F90
put_FCFLAGS = $(AM_FCFLAGS)
put_LDFLAGS = $(AM_LDFLAGS)

put_nolog_SOURCES = $(srcdir)/put.F90
put_nolog_FCFLAGS = $(AM_FCFLAGS)
put_nolog_LDFLAGS = $(AM_LDFLAGS)

rput_SOURCES = $(srcdir)/rput.F90
rput_FCFLAGS = $(AM_FCFLAGS)
rput_LDFLAGS = $(AM_LDFLAGS)

rput_nolog_SOURCES = $(srcdir)/rput.F90
rput_nolog_FCFLAGS = $(AM_FCFLAGS)
rput_nolog_LDFLAGS = $(AM_LDFLAGS)

get_SOURCES = $(srcdir)/get.F90
get_FCFLAGS = $(AM_FCFLAGS)
get_LDFLAGS = $(AM_LDFLAGS)

get_nolog_SOURCES = $(srcdir)/get.F90
get_nolog_FCFLAGS = $(AM_FCFLAGS)
get_nolog_LDFLAGS = $(AM_LDFLAGS)

rget_SOURCES = $(srcdir)/rget.F90
rget_FCFLAGS = $(AM_FCFLAGS)
rget_LDFLAGS = $(AM_LDFLAGS)

rget_nolog_SOURCES = $(srcdir)/rget.F90
rget_nolog_FCFLAGS = $(AM_FCFLAGS)
rget_nolog_LDFLAGS = $(AM_LDFLAGS)

accumulate_SOURCES = $(srcdir)/accumulate.F90
accumulate_FCFLAGS = $(AM_FCFLAGS)
accumulate_LDFLAGS = $(AM_LDFLAGS)

accumulate_nolog_SOURCES = $(srcdir)/accumulate.F90
accumulate_nolog_FCFLAGS = $(AM_FCFLAGS)
accumulate_nolog_LDFLAGS = $(AM_LDFLAGS)

raccumulate_SOURCES = $(srcdir)/raccumulate.F90
raccumulate_FCFLAGS = $(AM_FCFLAGS)
raccumulate_LDFLAGS = $(AM_LDFLAGS)

raccumulate_nolog_SOURCES = $(srcdir)/raccumulate.F90
raccumulate_nolog_FCFLAGS = $(AM_FCFLAGS)
raccumulate_nolog_LDFLAGS = $(AM_LDFLAGS)

get_accumulate_SOURCES = $(srcdir)/get_accumulate.F90
get_accumulate_FCFLAGS = $(AM_FCFLAGS)
get_accumulate_LDFLAGS = $(AM_LDFLAGS)

get_accumulate_nolog_SOURCES = $(srcdir)/get_accumulate.F90
get_accumulate_nolog_FCFLAGS = $(AM_FCFLAGS)
get_accumulate_nolog_LDFLAGS = $(AM_LDFLAGS)

rget_accumulate_SOURCES = $(srcdir)/rget_accumulate.F90
rget_accumulate_FCFLAGS = $(AM_FCFLAGS)
rget_accumulate_LDFLAGS = $(AM_LDFLAGS)

rget_accumulate_nolog_SOURCES = $(srcdir)/rget_accumulate.F90
rget_accumulate_nolog_FCFLAGS = $(AM_FCFLAGS)
rget_accumulate_nolog_LDFLAGS = $(AM_LDFLAGS)

fetch_and_op_SOURCES = $(srcdir)/fetch_and_op.F90
fetch_and_op_FCFLAGS = $(AM_FCFLAGS)
fetch_and_op_LDFLAGS = $(AM_LDFLAGS)

fetch_and_op_nolog_SOURCES = $(srcdir)/fetch_and_op.F90
fetch_and_op_nolog_FCFLAGS = $(AM_FCFLAGS)
fetch_and_op_nolog_LDFLAGS = $(AM_LDFLAGS)

compare_and_swap_SOURCES = $(srcdir)/compare_and_swap.F90
compare_and_swap_FCFLAGS = $(AM_FCFLAGS)
compare_and_swap_LDFLAGS = $(AM_LDFLAGS)

compare_and_swap_nolog_SOURCES = $(srcdir)/compare_and_swap.F90
compare_and_swap_nolog_FCFLAGS = $(AM_FCFLAGS)
compare_and_swap_nolog_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += put.sh \
		       put_nolog.sh \
		       rput.sh \
		       rput_nolog.sh \
		       get.sh \
		       get_nolog.sh \
		       rget.sh \
		       rget_nolog.sh \
		       accumulate.sh \
		       accumulate_nolog.sh \
		       raccumulate.sh \
		       raccumulate_nolog.sh \
		       get_accumulate.sh \
		       get_accumulate_nolog.sh \
		       rget_accumulate.sh \
		       rget_accumulate_nolog.sh \
		       fetch_and_op.sh \
		       fetch_and_op_nolog.sh \
		       compare_and_swap.sh \
		       compare_and_swap_nolog.sh

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
XFAIL_TESTS = neighbor_alltoallw_nolog.sh \
              ineighbor_alltoallw_nolog.sh

CLEANFILES = vftrace_self_profile-*
