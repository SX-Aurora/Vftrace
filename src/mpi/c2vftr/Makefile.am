noinst_LTLIBRARIES = libvftr_mpi_c2vftr.la

libvftr_mpi_c2vftr_la_SOURCES =

if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += init_c2vftr.c \
				 finalize_c2vftr.c
endif
if HAS_MPI_STD_2_1
libvftr_mpi_c2vftr_la_SOURCES += init_thread_c2vftr.c
endif

##################
# point to point #
##################
if HAS_MPI_STD_1_0
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
endif

##################
# persistent p2p #
##################
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += start_c2vftr.c \
                                 startall_c2vftr.c \
                                 send_init_c2vftr.c \
                                 bsend_init_c2vftr.c \
                                 ssend_init_c2vftr.c \
                                 rsend_init_c2vftr.c \
                                 recv_init_c2vftr.c
endif

#########
# tests #
#########
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += test_c2vftr.c \
                                 testany_c2vftr.c \
                                 testsome_c2vftr.c \
                                 testall_c2vftr.c
endif

########
# wait #
########
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += wait_c2vftr.c \
                                 waitany_c2vftr.c \
                                 waitsome_c2vftr.c \
                                 waitall_c2vftr.c
endif

########################
# request deallocation #
########################
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += request_free_c2vftr.c
endif

###############
# collectives #
###############
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += barrier_c2vftr.c \
				 bcast_c2vftr.c \
				 gather_c2vftr.c \
				 gatherv_c2vftr.c \
				 scatter_c2vftr.c \
				 scatterv_c2vftr.c \
				 allgather_c2vftr.c \
				 allgatherv_c2vftr.c \
				 alltoall_c2vftr.c \
				 alltoallv_c2vftr.c
endif
if HAS_MPI_STD_2_0
libvftr_mpi_c2vftr_la_SOURCES += alltoallw_c2vftr.c
endif
if HAS_MPI_STD_3_0
libvftr_mpi_c2vftr_la_SOURCES += ibarrier_c2vftr.c \
				 ibcast_c2vftr.c \
				 igather_c2vftr.c \
				 igatherv_c2vftr.c \
				 iscatter_c2vftr.c \
				 iscatterv_c2vftr.c \
				 iallgather_c2vftr.c \
				 iallgatherv_c2vftr.c \
				 ialltoall_c2vftr.c \
				 ialltoallv_c2vftr.c \
				 ialltoallw_c2vftr.c
endif

########################
# neighbor collectives #
########################

if HAS_MPI_STD_3_0
libvftr_mpi_c2vftr_la_SOURCES += neighbor_allgather_c2vftr.c \
				 ineighbor_allgather_c2vftr.c \
				 neighbor_allgatherv_c2vftr.c \
				 ineighbor_allgatherv_c2vftr.c \
				 neighbor_alltoall_c2vftr.c \
				 ineighbor_alltoall_c2vftr.c \
				 neighbor_alltoallv_c2vftr.c \
				 ineighbor_alltoallv_c2vftr.c \
				 neighbor_alltoallw_c2vftr.c \
				 ineighbor_alltoallw_c2vftr.c
endif

####################
# global reduction #
####################
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += reduce_c2vftr.c \
				 allreduce_c2vftr.c \
				 reduce_scatter_c2vftr.c \
				 scan_c2vftr.c
endif
if HAS_MPI_STD_2_0
libvftr_mpi_c2vftr_la_SOURCES += exscan_c2vftr.c
endif
if HAS_MPI_STD_2_2
libvftr_mpi_c2vftr_la_SOURCES += reduce_scatter_block_c2vftr.c
endif
if HAS_MPI_STD_3_0
libvftr_mpi_c2vftr_la_SOURCES += ireduce_c2vftr.c \
				 iallreduce_c2vftr.c \
				 ireduce_scatter_block_c2vftr.c \
				 ireduce_scatter_c2vftr.c \
				 iscan_c2vftr.c \
				 iexscan_c2vftr.c 
endif

#############
# one sided #
#############
if HAS_MPI_STD_2_0
libvftr_mpi_c2vftr_la_SOURCES += get_c2vftr.c \
			       put_c2vftr.c \
			       accumulate_c2vftr.c
endif
if HAS_MPI_STD_3_0
libvftr_mpi_c2vftr_la_SOURCES += rget_c2vftr.c \
				 rput_c2vftr.c \
				 raccumulate_c2vftr.c \
				 get_accumulate_c2vftr.c \
				 rget_accumulate_c2vftr.c \
				 fetch_and_op_c2vftr.c \
				 compare_and_swap_c2vftr.c
endif

############
# pcontrol #
############
if HAS_MPI_STD_1_0
libvftr_mpi_c2vftr_la_SOURCES += pcontrol_c2vftr.c
endif

libvftr_mpi_c2vftr_la_CFLAGS = -D_MPI
libvftr_mpi_c2vftr_la_CFLAGS += -I${srcdir}/../utils/
libvftr_mpi_c2vftr_la_CFLAGS += -I${srcdir}/../vftr_mpi/
libvftr_mpi_c2vftr_la_CFLAGS += -I${srcdir}/../../cuda

if ENABLE_CUDAPROF
libvftr_mpi_c2vftr_la_CFLAGS += -D_CUDA
endif

if ENABLE_ACCPROF
libvftr_mpi_c2vftr_la_CFLAGS += -D_ACCPROF
endif
