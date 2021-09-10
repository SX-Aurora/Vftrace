noinst_LTLIBRARIES = libvftr_mpi_c2vftr.la


libvftr_mpi_c2vftr_la_SOURCES = init_c2vftr.c \
				finalize_c2vftr.c

##################
# point to point #
##################
libvftr_mpi_c2vftr_la_SOURCES += send_c2vftr.c \
				 isend_c2vftr.c \
				 bsend_c2vftr.c \
				 ibsend_c2vftr.c \
				 ssend_c2vftr.c \
				 issend_c2vftr.c \
				 rsend_c2vftr.c \
				 irsend_c2vftr.c \
				 recv_c2vftr.c \
				 irecv_c2vftr.c \
				 sendrecv_c2vftr.c \
				 sendrecv_replace_c2vftr.c \
				 probe_c2vftr.c \
				 iprobe_c2vftr.c 

###################
## persistent p2p #
###################
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_start_c2vftr.c \
#                              vftr_mpi_startall_c2vftr.c \
#                              vftr_mpi_send_init_c2vftr.c \
#                              vftr_mpi_bsend_init_c2vftr.c \
#                              vftr_mpi_ssend_init_c2vftr.c \
#                              vftr_mpi_rsend_init_c2vftr.c \
#                              vftr_mpi_recv_init_c2vftr.c
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_start_c2vftr_finterface.F90 \
#                              vftr_mpi_startall_c2vftr_finterface.F90 \
#                              vftr_mpi_send_init_c2vftr_finterface.F90 \
#                              vftr_mpi_bsend_init_c2vftr_finterface.F90 \
#                              vftr_mpi_ssend_init_c2vftr_finterface.F90 \
#                              vftr_mpi_rsend_init_c2vftr_finterface.F90 \
#                              vftr_mpi_recv_init_c2vftr_finterface.F90

#########
# tests #
#########
libvftr_mpi_c2vftr_la_SOURCES += test_c2vftr.c \
				 testany_c2vftr.c \
				 testsome_c2vftr.c \
				 testall_c2vftr.c

########
# wait #
########
libvftr_mpi_c2vftr_la_SOURCES += wait_c2vftr.c \
				 waitany_c2vftr.c \
				 waitsome_c2vftr.c \
				 waitall_c2vftr.c

#########################
## request deallocation #
#########################
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_request_free_c2vftr.c
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_request_free_c2vftr_finterface.F90
#
################
## collectives #
################
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_barrier_c2vftr.c \
#                              vftr_mpi_bcast_c2vftr.c \
#                              vftr_mpi_ibcast_c2vftr.c \
#                              vftr_mpi_gather_c2vftr.c \
#                              vftr_mpi_igather_c2vftr.c \
#                              vftr_mpi_gatherv_c2vftr.c \
#                              vftr_mpi_igatherv_c2vftr.c \
#                              vftr_mpi_scatter_c2vftr.c \
#                              vftr_mpi_iscatter_c2vftr.c \
#                              vftr_mpi_scatterv_c2vftr.c \
#                              vftr_mpi_iscatterv_c2vftr.c \
#                              vftr_mpi_allgather_c2vftr.c \
#                              vftr_mpi_iallgather_c2vftr.c \
#                              vftr_mpi_allgatherv_c2vftr.c \
#                              vftr_mpi_iallgatherv_c2vftr.c \
#                              vftr_mpi_alltoall_c2vftr.c \
#                              vftr_mpi_ialltoall_c2vftr.c \
#                              vftr_mpi_alltoallv_c2vftr.c \
#                              vftr_mpi_ialltoallv_c2vftr.c \
#                              vftr_mpi_alltoallw_c2vftr.c \
#                              vftr_mpi_ialltoallw_c2vftr.c
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_barrier_c2vftr_finterface.F90 \
#                              vftr_mpi_bcast_c2vftr_finterface.F90 \
#                              vftr_mpi_ibcast_c2vftr_finterface.F90 \
#                              vftr_mpi_gather_c2vftr_finterface.F90 \
#                              vftr_mpi_igather_c2vftr_finterface.F90 \
#                              vftr_mpi_gatherv_c2vftr_finterface.F90 \
#                              vftr_mpi_igatherv_c2vftr_finterface.F90 \
#                              vftr_mpi_scatter_c2vftr_finterface.F90 \
#                              vftr_mpi_iscatter_c2vftr_finterface.F90 \
#                              vftr_mpi_scatterv_c2vftr_finterface.F90 \
#                              vftr_mpi_iscatterv_c2vftr_finterface.F90 \
#                              vftr_mpi_allgather_c2vftr_finterface.F90 \
#                              vftr_mpi_iallgather_c2vftr_finterface.F90 \
#                              vftr_mpi_allgatherv_c2vftr_finterface.F90 \
#                              vftr_mpi_iallgatherv_c2vftr_finterface.F90 \
#                              vftr_mpi_alltoall_c2vftr_finterface.F90 \
#                              vftr_mpi_ialltoall_c2vftr_finterface.F90 \
#                              vftr_mpi_alltoallv_c2vftr_finterface.F90 \
#                              vftr_mpi_ialltoallv_c2vftr_finterface.F90 \
#                              vftr_mpi_alltoallw_c2vftr_finterface.F90 \
#                              vftr_mpi_ialltoallw_c2vftr_finterface.F90
#
#####################
## global reduction #
#####################
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_reduce_c2vftr.c \
#                              vftr_mpi_ireduce_c2vftr.c \
#                              vftr_mpi_allreduce_c2vftr.c \
#                              vftr_mpi_iallreduce_c2vftr.c \
#                              vftr_mpi_reduce_scatter_block_c2vftr.c \
#                              vftr_mpi_ireduce_scatter_block_c2vftr.c \
#                              vftr_mpi_reduce_scatter_c2vftr.c \
#                              vftr_mpi_ireduce_scatter_c2vftr.c \
#                              vftr_mpi_scan_c2vftr.c \
#                              vftr_mpi_iscan_c2vftr.c \
#                              vftr_mpi_exscan_c2vftr.c \
#                              vftr_mpi_iexscan_c2vftr.c 
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_reduce_c2vftr_finterface.F90 \
#                              vftr_mpi_ireduce_c2vftr_finterface.F90 \
#                              vftr_mpi_allreduce_c2vftr_finterface.F90 \
#                              vftr_mpi_iallreduce_c2vftr_finterface.F90 \
#                              vftr_mpi_reduce_scatter_block_c2vftr_finterface.F90 \
#                              vftr_mpi_ireduce_scatter_block_c2vftr_finterface.F90 \
#                              vftr_mpi_reduce_scatter_c2vftr_finterface.F90 \
#                              vftr_mpi_ireduce_scatter_c2vftr_finterface.F90 \
#                              vftr_mpi_scan_c2vftr_finterface.F90 \
#                              vftr_mpi_iscan_c2vftr_finterface.F90 \
#                              vftr_mpi_exscan_c2vftr_finterface.F90 \
#                              vftr_mpi_iexscan_c2vftr_finterface.F90
#
##############
## one sided #
##############
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_put_c2vftr.c \
#                              vftr_mpi_rput_c2vftr.c \
#                              vftr_mpi_get_c2vftr.c \
#                              vftr_mpi_rget_c2vftr.c \
#                              vftr_mpi_accumulate_c2vftr.c \
#                              vftr_mpi_raccumulate_c2vftr.c \
#                              vftr_mpi_get_accumulate_c2vftr.c \
#                              vftr_mpi_rget_accumulate_c2vftr.c \
#                              vftr_mpi_fetch_and_op_c2vftr.c \
#                              vftr_mpi_compare_and_swap_c2vftr.c
#libvftr_mpi_c2vftr_la_SOURCES += vftr_mpi_put_c2vftr_finterface.F90 \
#                              vftr_mpi_rput_c2vftr_finterface.F90 \
#                              vftr_mpi_get_c2vftr_finterface.F90 \
#                              vftr_mpi_rget_c2vftr_finterface.F90 \
#                              vftr_mpi_accumulate_c2vftr_finterface.F90 \
#                              vftr_mpi_raccumulate_c2vftr_finterface.F90 \
#                              vftr_mpi_get_accumulate_c2vftr_finterface.F90 \
#                              vftr_mpi_rget_accumulate_c2vftr_finterface.F90 \
#                              vftr_mpi_fetch_and_op_c2vftr_finterface.F90 \
#                              vftr_mpi_compare_and_swap_c2vftr_finterface.F90

############
# pcontrol #
############
libvftr_mpi_c2vftr_la_SOURCES += pcontrol_c2vftr.c

libvftr_mpi_c2vftr_la_CFLAGS =
libvftr_mpi_c2vftr_la_FCFLAGS =

libvftr_mpi_c2vftr_la_CFLAGS += -D_MPI
libvftr_mpi_c2vftr_la_CFLAGS += -I${srcdir}/../../ \
				-I${srcdir}/../vftr_mpi/
