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

##################
# persistent p2p #
##################
libvftr_mpi_c2vftr_la_SOURCES += start_c2vftr.c \
                                 startall_c2vftr.c \
                                 send_init_c2vftr.c \
                                 bsend_init_c2vftr.c \
                                 ssend_init_c2vftr.c \
                                 rsend_init_c2vftr.c \
                                 recv_init_c2vftr.c

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

########################
# request deallocation #
########################
libvftr_mpi_c2vftr_la_SOURCES += request_free_c2vftr.c

###############
# collectives #
###############
libvftr_mpi_c2vftr_la_SOURCES += barrier_c2vftr.c \
                                 ibarrier_c2vftr.c \
                                 bcast_c2vftr.c \
                                 ibcast_c2vftr.c 
#                                gather_c2vftr.c \
#                                igather_c2vftr.c \
#                                gatherv_c2vftr.c \
#                                igatherv_c2vftr.c \
#                                scatter_c2vftr.c \
#                                iscatter_c2vftr.c \
#                                scatterv_c2vftr.c \
#                                iscatterv_c2vftr.c \
#                                allgather_c2vftr.c \
#                                iallgather_c2vftr.c \
#                                allgatherv_c2vftr.c \
#                                iallgatherv_c2vftr.c \
#                                alltoall_c2vftr.c \
#                                ialltoall_c2vftr.c \
#                                alltoallv_c2vftr.c \
#                                ialltoallv_c2vftr.c \
#                                alltoallw_c2vftr.c \
#                                ialltoallw_c2vftr.c

####################
# global reduction #
####################
libvftr_mpi_c2vftr_la_SOURCES += reduce_c2vftr.c \
                                 ireduce_c2vftr.c \
                                 allreduce_c2vftr.c \
                                 iallreduce_c2vftr.c \
                                 reduce_scatter_block_c2vftr.c \
                                 ireduce_scatter_block_c2vftr.c \
                                 reduce_scatter_c2vftr.c \
                                 ireduce_scatter_c2vftr.c \
                                 scan_c2vftr.c \
                                 iscan_c2vftr.c \
                                 exscan_c2vftr.c \
                                 iexscan_c2vftr.c 

#############
# one sided #
#############
libvftr_mpi_c2vftr_la_SOURCES += put_c2vftr.c \
                                 rput_c2vftr.c \
                                 get_c2vftr.c \
                                 rget_c2vftr.c \
                                 accumulate_c2vftr.c \
                                 raccumulate_c2vftr.c \
                                 get_accumulate_c2vftr.c \
                                 rget_accumulate_c2vftr.c \
                                 fetch_and_op_c2vftr.c \
                                 compare_and_swap_c2vftr.c

############
# pcontrol #
############
libvftr_mpi_c2vftr_la_SOURCES += pcontrol_c2vftr.c

libvftr_mpi_c2vftr_la_CFLAGS = -D_MPI
libvftr_mpi_c2vftr_la_CFLAGS += -I${srcdir}/../../
libvftr_mpi_c2vftr_la_CFLAGS += -I${srcdir}/../vftr_mpi/
