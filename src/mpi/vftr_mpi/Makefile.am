noinst_LTLIBRARIES = libvftr_mpi.la


libvftr_mpi_la_SOURCES = vftr_mpi_utils.c
if WITH_MPI
libvftr_mpi_la_SOURCES += vftr_requests.c \
			  vftr_p2p_requests.c \
			  vftr_persistent_requests.c \
			  vftr_collective_requests.c \
			  vftr_onesided_requests.c \
			  vftr_clear_requests.c \
			  vftr_sync_messages.c \
                          vftr_mpi_buf_addr_const.c

##################
# point to point #
##################
libvftr_mpi_la_SOURCES += vftr_mpi_send.c \
			  vftr_mpi_isend.c \
			  vftr_mpi_bsend.c \
			  vftr_mpi_ibsend.c \
			  vftr_mpi_ssend.c \
			  vftr_mpi_issend.c \
			  vftr_mpi_rsend.c \
			  vftr_mpi_irsend.c \
			  vftr_mpi_recv.c \
			  vftr_mpi_irecv.c \
			  vftr_mpi_sendrecv.c \
			  vftr_mpi_sendrecv_replace.c \
			  vftr_mpi_probe.c \
			  vftr_mpi_iprobe.c 

##################
# persistent p2p #
##################
libvftr_mpi_la_SOURCES += vftr_mpi_start.c \
                          vftr_mpi_startall.c \
			  vftr_mpi_send_init.c \
			  vftr_mpi_bsend_init.c \
			  vftr_mpi_ssend_init.c \
			  vftr_mpi_rsend_init.c \
			  vftr_mpi_recv_init.c

#########
# tests #
#########
libvftr_mpi_la_SOURCES += vftr_mpi_test.c \
			  vftr_mpi_testany.c \
			  vftr_mpi_testsome.c \
			  vftr_mpi_testall.c

########
# wait #
########
libvftr_mpi_la_SOURCES += vftr_mpi_wait.c \
			  vftr_mpi_waitany.c \
			  vftr_mpi_waitsome.c \
			  vftr_mpi_waitall.c

########################
# request deallocation #
########################
libvftr_mpi_la_SOURCES += vftr_mpi_request_free.c

###############
# collectives #
###############
libvftr_mpi_la_SOURCES += vftr_mpi_barrier.c \
			  vftr_mpi_bcast.c \
			  vftr_mpi_ibcast.c \
			  vftr_mpi_gather.c \
			  vftr_mpi_igather.c \
			  vftr_mpi_gatherv.c \
			  vftr_mpi_igatherv.c \
			  vftr_mpi_scatter.c \
			  vftr_mpi_iscatter.c \
			  vftr_mpi_scatterv.c \
			  vftr_mpi_iscatterv.c \
			  vftr_mpi_allgather.c \
			  vftr_mpi_iallgather.c \
			  vftr_mpi_allgatherv.c \
			  vftr_mpi_iallgatherv.c \
			  vftr_mpi_alltoall.c \
			  vftr_mpi_ialltoall.c \
			  vftr_mpi_alltoallv.c \
			  vftr_mpi_ialltoallv.c \
			  vftr_mpi_alltoallw.c \
			  vftr_mpi_ialltoallw.c

####################
# global reduction #
####################
libvftr_mpi_la_SOURCES += vftr_mpi_reduce.c \
			  vftr_mpi_ireduce.c \
			  vftr_mpi_allreduce.c \
			  vftr_mpi_iallreduce.c \
			  vftr_mpi_reduce_scatter_block.c \
			  vftr_mpi_ireduce_scatter_block.c \
			  vftr_mpi_reduce_scatter.c \
			  vftr_mpi_reduce_scatter.c \
			  vftr_mpi_ireduce_scatter.c \
			  vftr_mpi_scan.c \
			  vftr_mpi_iscan.c \
			  vftr_mpi_exscan.c \
			  vftr_mpi_iexscan.c 

#############
# one sided #
#############
libvftr_mpi_la_SOURCES += vftr_mpi_put.c \
			  vftr_mpi_rput.c \
			  vftr_mpi_get.c \
			  vftr_mpi_rget.c \
			  vftr_mpi_accumulate.c \
			  vftr_mpi_raccumulate.c \
			  vftr_mpi_get_accumulate.c \
			  vftr_mpi_rget_accumulate.c \
			  vftr_mpi_fetch_and_op.c \
			  vftr_mpi_compare_and_swap.c

############
# pcontrol #
############
libvftr_mpi_la_SOURCES += vftr_mpi_pcontrol.c
endif

libvftr_mpi_la_CFLAGS =
libvftr_mpi_la_FCFLAGS =

if WITH_MPI
libvftr_mpi_la_CFLAGS += -D_MPI
libvftr_mpi_la_FCFLAGS += -D_MPI
endif

libvftr_mpi_la_CFLAGS += -I${srcdir}/../../
