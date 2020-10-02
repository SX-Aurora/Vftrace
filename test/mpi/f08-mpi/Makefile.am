AM_FCFLAGS = -O3 -finstrument-functions

AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif

################
# System Tests #
################
check_PROGRAMS = init_finalize 
#		  ftypes \
#		  derived_ftypes
		
init_finalize_SOURCES = $(srcdir)/init_finalize.F90
init_finalize_FCFLAGS  = $(AM_FCFLAGS)
init_finalize_LDFLAGS = $(AM_LDFLAGS)

#ftypes_SOURCES = $(srcdir)/ftypes.F90
#ftypes_FCFLAGS  = $(AM_FCFLAGS)
#ftypes_LDFLAGS = $(AM_LDFLAGS)
#
#derived_ftypes_SOURCES = $(srcdir)/derived_ftypes.F90
#derived_ftypes_FCFLAGS  = $(AM_FCFLAGS)
#derived_ftypes_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS = init_finalize-1.sh \
                      init_finalize-2.sh \
                      init_finalize-3.sh \
                      init_finalize-4.sh 

if SUPPORTS_MPIF08_TS
else
#dist_noinst_SCRIPTS = ftypes.sh \
#		      derived_ftypes.sh
#
#########################
## point to point tests #
#########################
#check_PROGRAMS += send_recv \
#		   sendrecv \
#		   sendrecv_replace \
#		   isend_recv \
#		   send_irecv
#
#send_recv_SOURCES = $(srcdir)/send_recv.F90
#send_recv_FCFLAGS  = $(AM_FCFLAGS)
#send_recv_LDFLAGS = $(AM_LDFLAGS)
#
#sendrecv_SOURCES = $(srcdir)/sendrecv.F90
#sendrecv_FCFLAGS  = $(AM_FCFLAGS)
#sendrecv_LDFLAGS = $(AM_LDFLAGS)
#
#sendrecv_replace_SOURCES = $(srcdir)/sendrecv_replace.F90
#sendrecv_replace_FCFLAGS  = $(AM_FCFLAGS)
#sendrecv_replace_LDFLAGS = $(AM_LDFLAGS)
#
#isend_recv_SOURCES = $(srcdir)/isend_recv.F90
#isend_recv_FCFLAGS  = $(AM_FCFLAGS)
#isend_recv_LDFLAGS = $(AM_LDFLAGS)
#
#send_irecv_SOURCES = $(srcdir)/send_irecv.F90
#send_irecv_FCFLAGS  = $(AM_FCFLAGS)
#send_irecv_LDFLAGS = $(AM_LDFLAGS)
#
#dist_noinst_SCRIPTS += send_recv.sh \
#		       sendrecv.sh \
#		       sendrecv_replace.sh \
#		       isend_recv.sh \
#		       send_irecv.sh

#####################
## Collective tests #
#####################
#
check_PROGRAMS += bcast \
		  bcast_intercom \
		  gather \
		  gather_intercom \
		  gather_inplace \
		  gatherv \
		  gatherv_intercom \
		  gatherv_inplace \
		  scatter \
		  scatter_intercom \
		  scatter_inplace \
		  scatterv \
		  scatterv_intercom \
		  scatterv_inplace \
		  allgather \
		  allgather_intercom \
		  allgather_inplace \
		  allgatherv \
		  allgatherv_intercom \
		  allgatherv_inplace \
		  alltoall \
		  alltoall_intercom \
		  alltoall_inplace \
		  alltoallv \
		  alltoallv_intercom \
		  alltoallv_inplace \
		  alltoallw \
		  alltoallw_intercom \
		  alltoallw_inplace

bcast_SOURCES = $(srcdir)/bcast.F90
bcast_FCFLAGS = $(AM_FCFLAGS)
bcast_LDFLAGS = $(AM_LDFLAGS)

bcast_intercom_SOURCES = $(srcdir)/bcast_intercom.F90
bcast_intercom_FCFLAGS = $(AM_FCFLAGS)
bcast_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_SOURCES = $(srcdir)/gather.F90
gather_FCFLAGS = $(AM_FCFLAGS)
gather_LDFLAGS = $(AM_LDFLAGS)

gather_intercom_SOURCES = $(srcdir)/gather_intercom.F90
gather_intercom_FCFLAGS = $(AM_FCFLAGS)
gather_intercom_LDFLAGS = $(AM_LDFLAGS)

gather_inplace_SOURCES = $(srcdir)/gather_inplace.F90
gather_inplace_FCFLAGS = $(AM_FCFLAGS)
gather_inplace_LDFLAGS = $(AM_LDFLAGS)

gatherv_SOURCES = $(srcdir)/gatherv.F90
gatherv_FCFLAGS = $(AM_FCFLAGS)
gatherv_LDFLAGS = $(AM_LDFLAGS)

gatherv_intercom_SOURCES = $(srcdir)/gatherv_intercom.F90
gatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
gatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

gatherv_inplace_SOURCES = $(srcdir)/gatherv_inplace.F90
gatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
gatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

scatter_SOURCES = $(srcdir)/scatter.F90
scatter_FCFLAGS = $(AM_FCFLAGS)
scatter_LDFLAGS = $(AM_LDFLAGS)

scatter_intercom_SOURCES = $(srcdir)/scatter_intercom.F90
scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

scatter_inplace_SOURCES = $(srcdir)/scatter_inplace.F90
scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

scatterv_SOURCES = $(srcdir)/scatterv.F90
scatterv_FCFLAGS = $(AM_FCFLAGS)
scatterv_LDFLAGS = $(AM_LDFLAGS)

scatterv_intercom_SOURCES = $(srcdir)/scatterv_intercom.F90
scatterv_intercom_FCFLAGS = $(AM_FCFLAGS)
scatterv_intercom_LDFLAGS = $(AM_LDFLAGS)

scatterv_inplace_SOURCES = $(srcdir)/scatterv_inplace.F90
scatterv_inplace_FCFLAGS = $(AM_FCFLAGS)
scatterv_inplace_LDFLAGS = $(AM_LDFLAGS)

allgather_SOURCES = $(srcdir)/allgather.F90
allgather_FCFLAGS = $(AM_FCFLAGS)
allgather_LDFLAGS = $(AM_LDFLAGS)

allgather_intercom_SOURCES = $(srcdir)/allgather_intercom.F90
allgather_intercom_FCFLAGS = $(AM_FCFLAGS)
allgather_intercom_LDFLAGS = $(AM_LDFLAGS)

allgather_inplace_SOURCES = $(srcdir)/allgather_inplace.F90
allgather_inplace_FCFLAGS = $(AM_FCFLAGS)
allgather_inplace_LDFLAGS = $(AM_LDFLAGS)

allgatherv_SOURCES = $(srcdir)/allgatherv.F90
allgatherv_FCFLAGS = $(AM_FCFLAGS)
allgatherv_LDFLAGS = $(AM_LDFLAGS)

allgatherv_intercom_SOURCES = $(srcdir)/allgatherv_intercom.F90
allgatherv_intercom_FCFLAGS = $(AM_FCFLAGS)
allgatherv_intercom_LDFLAGS = $(AM_LDFLAGS)

allgatherv_inplace_SOURCES = $(srcdir)/allgatherv_inplace.F90
allgatherv_inplace_FCFLAGS = $(AM_FCFLAGS)
allgatherv_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoall_SOURCES = $(srcdir)/alltoall.F90
alltoall_FCFLAGS = $(AM_FCFLAGS)
alltoall_LDFLAGS = $(AM_LDFLAGS)

alltoall_intercom_SOURCES = $(srcdir)/alltoall_intercom.F90
alltoall_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoall_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoall_inplace_SOURCES = $(srcdir)/alltoall_inplace.F90
alltoall_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoall_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallv_SOURCES = $(srcdir)/alltoallv.F90
alltoallv_FCFLAGS = $(AM_FCFLAGS)
alltoallv_LDFLAGS = $(AM_LDFLAGS)

alltoallv_intercom_SOURCES = $(srcdir)/alltoallv_intercom.F90
alltoallv_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallv_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallv_inplace_SOURCES = $(srcdir)/alltoallv_inplace.F90
alltoallv_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoallv_inplace_LDFLAGS = $(AM_LDFLAGS)

alltoallw_SOURCES = $(srcdir)/alltoallw.F90
alltoallw_FCFLAGS = $(AM_FCFLAGS)
alltoallw_LDFLAGS = $(AM_LDFLAGS)

alltoallw_intercom_SOURCES = $(srcdir)/alltoallw_intercom.F90
alltoallw_intercom_FCFLAGS = $(AM_FCFLAGS)
alltoallw_intercom_LDFLAGS = $(AM_LDFLAGS)

alltoallw_inplace_SOURCES = $(srcdir)/alltoallw_inplace.F90
alltoallw_inplace_FCFLAGS = $(AM_FCFLAGS)
alltoallw_inplace_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += bcast.sh \
		       bcast_intercom.sh \
		       gather.sh \
		       gather_intercom.sh \
		       gather_inplace.sh \
		       gatherv.sh \
		       gatherv_intercom.sh \
		       gatherv_inplace.sh \
		       scatter.sh \
		       scatter_intercom.sh \
		       scatter_inplace.sh \
		       scatterv.sh \
		       scatterv_intercom.sh \
		       scatterv_inplace.sh \
		       allgather.sh \
		       allgather_intercom.sh \
		       allgather_inplace.sh \
		       allgatherv.sh \
		       allgatherv_intercom.sh \
		       allgatherv_inplace.sh \
		       alltoall.sh \
		       alltoall_intercom.sh \
		       alltoall_inplace.sh \
		       alltoallv.sh \
		       alltoallv_intercom.sh \
		       alltoallv_inplace.sh \
		       alltoallw.sh \
		       alltoallw_intercom.sh \
		       alltoallw_inplace.sh

#####################
# Global reductions #
#####################

check_PROGRAMS += reduce \
		  reduce_intercom \
		  reduce_inplace \
		  allreduce \
		  allreduce_intercom \
		  allreduce_inplace \
		  reduce_scatter_block \
		  reduce_scatter_block_intercom \
		  reduce_scatter_block_inplace \
		  reduce_scatter \
		  reduce_scatter_intercom \
		  reduce_scatter_inplace \
		  scan \
		  scan_inplace \
		  exscan \
		  exscan_inplace 

reduce_SOURCES = $(srcdir)/reduce.F90
reduce_FCFLAGS = $(AM_FCFLAGS)
reduce_LDFLAGS = $(AM_LDFLAGS)

reduce_intercom_SOURCES = $(srcdir)/reduce_intercom.F90
reduce_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_inplace_SOURCES = $(srcdir)/reduce_inplace.F90
reduce_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_inplace_LDFLAGS = $(AM_LDFLAGS)

allreduce_SOURCES = $(srcdir)/allreduce.F90
allreduce_FCFLAGS = $(AM_FCFLAGS)
allreduce_LDFLAGS = $(AM_LDFLAGS)

allreduce_intercom_SOURCES = $(srcdir)/allreduce_intercom.F90
allreduce_intercom_FCFLAGS = $(AM_FCFLAGS)
allreduce_intercom_LDFLAGS = $(AM_LDFLAGS)

allreduce_inplace_SOURCES = $(srcdir)/allreduce_inplace.F90
allreduce_inplace_FCFLAGS = $(AM_FCFLAGS)
allreduce_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_SOURCES = $(srcdir)/reduce_scatter_block.F90
reduce_scatter_block_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_intercom_SOURCES = $(srcdir)/reduce_scatter_block_intercom.F90
reduce_scatter_block_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_block_inplace_SOURCES = $(srcdir)/reduce_scatter_block_inplace.F90
reduce_scatter_block_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_block_inplace_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_SOURCES = $(srcdir)/reduce_scatter.F90
reduce_scatter_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_intercom_SOURCES = $(srcdir)/reduce_scatter_intercom.F90
reduce_scatter_intercom_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_intercom_LDFLAGS = $(AM_LDFLAGS)

reduce_scatter_inplace_SOURCES = $(srcdir)/reduce_scatter_inplace.F90
reduce_scatter_inplace_FCFLAGS = $(AM_FCFLAGS)
reduce_scatter_inplace_LDFLAGS = $(AM_LDFLAGS)

scan_SOURCES = $(srcdir)/scan.F90
scan_FCFLAGS = $(AM_FCFLAGS)
scan_LDFLAGS = $(AM_LDFLAGS)

scan_inplace_SOURCES = $(srcdir)/scan_inplace.F90
scan_inplace_FCFLAGS = $(AM_FCFLAGS)
scan_inplace_LDFLAGS = $(AM_LDFLAGS)

exscan_SOURCES = $(srcdir)/exscan.F90
exscan_FCFLAGS = $(AM_FCFLAGS)
exscan_LDFLAGS = $(AM_LDFLAGS)

exscan_inplace_SOURCES = $(srcdir)/exscan_inplace.F90
exscan_inplace_FCFLAGS = $(AM_FCFLAGS)
exscan_inplace_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += reduce.sh \
		       reduce_intercom.sh \
		       reduce_inplace.sh \
		       allreduce.sh \
		       allreduce_intercom.sh \
		       allreduce_inplace.sh \
		       reduce_scatter_block.sh \
		       reduce_scatter_block_intercom.sh \
		       reduce_scatter_block_inplace.sh \
		       reduce_scatter.sh \
		       reduce_scatter_intercom.sh \
		       reduce_scatter_inplace.sh \
		       scan.sh \
		       scan_inplace.sh \
		       exscan.sh \
		       exscan_inplace.sh 

###################
# One sided tests #
###################
check_PROGRAMS += put \
		  get \
		  accumulate \
		  get_accumulate \
		  fetch_and_op

put_SOURCES = $(srcdir)/put.F90
put_FCFLAGS = $(AM_FCFLAGS)
put_LDFLAGS = $(AM_LDFLAGS)

get_SOURCES = $(srcdir)/get.F90
get_FCFLAGS = $(AM_FCFLAGS)
get_LDFLAGS = $(AM_LDFLAGS)

accumulate_SOURCES = $(srcdir)/accumulate.F90
accumulate_FCFLAGS = $(AM_FCFLAGS)
accumulate_LDFLAGS = $(AM_LDFLAGS)

get_accumulate_SOURCES = $(srcdir)/get_accumulate.F90
get_accumulate_FCFLAGS = $(AM_FCFLAGS)
get_accumulate_LDFLAGS = $(AM_LDFLAGS)

fetch_and_op_SOURCES = $(srcdir)/fetch_and_op.F90
fetch_and_op_FCFLAGS = $(AM_FCFLAGS)
fetch_and_op_LDFLAGS = $(AM_LDFLAGS)

dist_noinst_SCRIPTS += put.sh \
		       get.sh \
		       accumulate.sh \
		       get_accumulate.sh \
		       fetch_and_op.sh

###############
## Test tests #
###############
#check_PROGRAMS += test \
#		   testany \
#		   testsome \
#		   testall
#
#test_SOURCES = $(srcdir)/test.F90
#test_FCFLAGS = $(AM_FCFLAGS)
#test_LDFLAGS = $(AM_LDFLAGS)
#
#testany_SOURCES = $(srcdir)/testany.F90
#testany_FCFLAGS = $(AM_FCFLAGS)
#testany_LDFLAGS = $(AM_LDFLAGS)
#
#testsome_SOURCES = $(srcdir)/testsome.F90
#testsome_FCFLAGS = $(AM_FCFLAGS)
#testsome_LDFLAGS = $(AM_LDFLAGS)
#
#testall_SOURCES = $(srcdir)/testall.F90
#testall_FCFLAGS = $(AM_FCFLAGS)
#testall_LDFLAGS = $(AM_LDFLAGS)
#
#dist_noinst_SCRIPTS += test.sh \
#		       testany.sh \
#		       testsome.sh \
#		       testall.sh
#
###############
## Wait tests #
###############
#
#check_PROGRAMS += wait \
#		   waitany \
#		   waitsome \
#		   waitall
#
#wait_SOURCES = $(srcdir)/wait.F90
#wait_FCFLAGS = $(AM_FCFLAGS)
#wait_LDFLAGS = $(AM_LDFLAGS)
#
#waitany_SOURCES = $(srcdir)/waitany.F90
#waitany_FCFLAGS = $(AM_FCFLAGS)
#waitany_LDFLAGS = $(AM_LDFLAGS)
#
#waitsome_SOURCES = $(srcdir)/waitsome.F90
#waitsome_FCFLAGS = $(AM_FCFLAGS)
#waitsome_LDFLAGS = $(AM_LDFLAGS)
#
#waitall_SOURCES = $(srcdir)/waitall.F90
#waitall_FCFLAGS = $(AM_FCFLAGS)
#waitall_LDFLAGS = $(AM_LDFLAGS)
#
#dist_noinst_SCRIPTS += wait.sh \
#		       waitany.sh \
#		       waitsome.sh \
#		       waitall.sh
endif

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_noinst_SCRIPTS)
