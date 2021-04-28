noinst_LTLIBRARIES = libvftr_mpi_f2c.la


libvftr_mpi_f2c_la_SOURCES = vftr_after_mpi_init_f2c_finterface.F90 \
			     vftr_finalize_f2c_finterface.F90 \
			     vftr_mpi_buf_addr_const_F.F90

##################
# point to point #
##################
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_send_f2c.c \
                              vftr_mpi_isend_f2c.c \
                              vftr_mpi_bsend_f2c.c \
                              vftr_mpi_ibsend_f2c.c \
                              vftr_mpi_ssend_f2c.c \
                              vftr_mpi_issend_f2c.c \
                              vftr_mpi_rsend_f2c.c \
                              vftr_mpi_irsend_f2c.c \
                              vftr_mpi_recv_f2c.c \
                              vftr_mpi_irecv_f2c.c \
                              vftr_mpi_sendrecv_f2c.c \
                              vftr_mpi_sendrecv_replace_f2c.c \
                              vftr_mpi_probe_f2c.c \
                              vftr_mpi_iprobe_f2c.c 
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_send_f2c_finterface.F90 \
                              vftr_mpi_isend_f2c_finterface.F90 \
                              vftr_mpi_bsend_f2c_finterface.F90 \
                              vftr_mpi_ibsend_f2c_finterface.F90 \
                              vftr_mpi_ssend_f2c_finterface.F90 \
                              vftr_mpi_issend_f2c_finterface.F90 \
                              vftr_mpi_rsend_f2c_finterface.F90 \
                              vftr_mpi_irsend_f2c_finterface.F90 \
                              vftr_mpi_recv_f2c_finterface.F90 \
                              vftr_mpi_irecv_f2c_finterface.F90 \
                              vftr_mpi_sendrecv_f2c_finterface.F90 \
                              vftr_mpi_sendrecv_replace_f2c_finterface.F90 \
                              vftr_mpi_probe_f2c_finterface.F90 \
                              vftr_mpi_iprobe_f2c_finterface.F90

##################
# persistent p2p #
##################
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_start_f2c.c \
                              vftr_mpi_startall_f2c.c \
                              vftr_mpi_send_init_f2c.c \
                              vftr_mpi_bsend_init_f2c.c \
                              vftr_mpi_ssend_init_f2c.c \
                              vftr_mpi_rsend_init_f2c.c \
                              vftr_mpi_recv_init_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_start_f2c_finterface.F90 \
                              vftr_mpi_startall_f2c_finterface.F90 \
                              vftr_mpi_send_init_f2c_finterface.F90 \
                              vftr_mpi_bsend_init_f2c_finterface.F90 \
                              vftr_mpi_ssend_init_f2c_finterface.F90 \
                              vftr_mpi_rsend_init_f2c_finterface.F90 \
                              vftr_mpi_recv_init_f2c_finterface.F90

#########
# tests #
#########
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_test_f2c.c \
                              vftr_mpi_testany_f2c.c \
                              vftr_mpi_testsome_f2c.c \
                              vftr_mpi_testall_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_test_f2c_finterface.F90 \
                              vftr_mpi_testany_f2c_finterface.F90 \
                              vftr_mpi_testsome_f2c_finterface.F90 \
                              vftr_mpi_testall_f2c_finterface.F90

########
# wait #
########
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_wait_f2c.c \
                              vftr_mpi_waitany_f2c.c \
                              vftr_mpi_waitsome_f2c.c \
                              vftr_mpi_waitall_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_wait_f2c_finterface.F90 \
                              vftr_mpi_waitany_f2c_finterface.F90 \
                              vftr_mpi_waitsome_f2c_finterface.F90 \
                              vftr_mpi_waitall_f2c_finterface.F90

########################
# request deallocation #
########################
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_request_free_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_request_free_f2c_finterface.F90

###############
# collectives #
###############
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_barrier_f2c.c \
                              vftr_mpi_bcast_f2c.c \
                              vftr_mpi_ibcast_f2c.c \
                              vftr_mpi_gather_f2c.c \
                              vftr_mpi_igather_f2c.c \
                              vftr_mpi_gatherv_f2c.c \
                              vftr_mpi_igatherv_f2c.c \
                              vftr_mpi_scatter_f2c.c \
                              vftr_mpi_iscatter_f2c.c \
                              vftr_mpi_scatterv_f2c.c \
                              vftr_mpi_iscatterv_f2c.c \
                              vftr_mpi_allgather_f2c.c \
                              vftr_mpi_iallgather_f2c.c \
                              vftr_mpi_allgatherv_f2c.c \
                              vftr_mpi_iallgatherv_f2c.c \
                              vftr_mpi_alltoall_f2c.c \
                              vftr_mpi_ialltoall_f2c.c \
                              vftr_mpi_alltoallv_f2c.c \
                              vftr_mpi_ialltoallv_f2c.c \
                              vftr_mpi_alltoallw_f2c.c \
                              vftr_mpi_ialltoallw_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_barrier_f2c_finterface.F90 \
                              vftr_mpi_bcast_f2c_finterface.F90 \
                              vftr_mpi_ibcast_f2c_finterface.F90 \
                              vftr_mpi_gather_f2c_finterface.F90 \
                              vftr_mpi_igather_f2c_finterface.F90 \
                              vftr_mpi_gatherv_f2c_finterface.F90 \
                              vftr_mpi_igatherv_f2c_finterface.F90 \
                              vftr_mpi_scatter_f2c_finterface.F90 \
                              vftr_mpi_iscatter_f2c_finterface.F90 \
                              vftr_mpi_scatterv_f2c_finterface.F90 \
                              vftr_mpi_iscatterv_f2c_finterface.F90 \
                              vftr_mpi_allgather_f2c_finterface.F90 \
                              vftr_mpi_iallgather_f2c_finterface.F90 \
                              vftr_mpi_allgatherv_f2c_finterface.F90 \
                              vftr_mpi_iallgatherv_f2c_finterface.F90 \
                              vftr_mpi_alltoall_f2c_finterface.F90 \
                              vftr_mpi_ialltoall_f2c_finterface.F90 \
                              vftr_mpi_alltoallv_f2c_finterface.F90 \
                              vftr_mpi_ialltoallv_f2c_finterface.F90 \
                              vftr_mpi_alltoallw_f2c_finterface.F90 \
                              vftr_mpi_ialltoallw_f2c_finterface.F90

####################
# global reduction #
####################
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_reduce_f2c.c \
                              vftr_mpi_ireduce_f2c.c \
                              vftr_mpi_allreduce_f2c.c \
                              vftr_mpi_iallreduce_f2c.c \
                              vftr_mpi_reduce_scatter_block_f2c.c \
                              vftr_mpi_ireduce_scatter_block_f2c.c \
                              vftr_mpi_reduce_scatter_f2c.c \
                              vftr_mpi_ireduce_scatter_f2c.c \
                              vftr_mpi_scan_f2c.c \
                              vftr_mpi_iscan_f2c.c \
                              vftr_mpi_exscan_f2c.c \
                              vftr_mpi_iexscan_f2c.c 
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_reduce_f2c_finterface.F90 \
                              vftr_mpi_ireduce_f2c_finterface.F90 \
                              vftr_mpi_allreduce_f2c_finterface.F90 \
                              vftr_mpi_iallreduce_f2c_finterface.F90 \
                              vftr_mpi_reduce_scatter_block_f2c_finterface.F90 \
                              vftr_mpi_ireduce_scatter_block_f2c_finterface.F90 \
                              vftr_mpi_reduce_scatter_f2c_finterface.F90 \
                              vftr_mpi_ireduce_scatter_f2c_finterface.F90 \
                              vftr_mpi_scan_f2c_finterface.F90 \
                              vftr_mpi_iscan_f2c_finterface.F90 \
                              vftr_mpi_exscan_f2c_finterface.F90 \
                              vftr_mpi_iexscan_f2c_finterface.F90

#############
# one sided #
#############
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_put_f2c.c \
                              vftr_mpi_rput_f2c.c \
                              vftr_mpi_get_f2c.c \
                              vftr_mpi_rget_f2c.c \
                              vftr_mpi_accumulate_f2c.c \
                              vftr_mpi_raccumulate_f2c.c \
                              vftr_mpi_get_accumulate_f2c.c \
                              vftr_mpi_rget_accumulate_f2c.c \
                              vftr_mpi_fetch_and_op_f2c.c \
                              vftr_mpi_compare_and_swap_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_put_f2c_finterface.F90 \
                              vftr_mpi_rput_f2c_finterface.F90 \
                              vftr_mpi_get_f2c_finterface.F90 \
                              vftr_mpi_rget_f2c_finterface.F90 \
                              vftr_mpi_accumulate_f2c_finterface.F90 \
                              vftr_mpi_raccumulate_f2c_finterface.F90 \
                              vftr_mpi_get_accumulate_f2c_finterface.F90 \
                              vftr_mpi_rget_accumulate_f2c_finterface.F90 \
                              vftr_mpi_fetch_and_op_f2c_finterface.F90 \
                              vftr_mpi_compare_and_swap_f2c_finterface.F90

############
# pcontrol #
############
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_pcontrol_f2c.c
libvftr_mpi_f2c_la_SOURCES += vftr_mpi_pcontrol_f2c_finterface.F90

libvftr_mpi_f2c_la_CFLAGS =
libvftr_mpi_f2c_la_FCFLAGS =

libvftr_mpi_f2c_la_CFLAGS += -D_MPI
libvftr_mpi_f2c_la_FCFLAGS += -D_MPI
libvftr_mpi_f2c_la_CFLAGS += -I${srcdir}/../../ \
			     -I${srcdir}/../vftr_mpi/
