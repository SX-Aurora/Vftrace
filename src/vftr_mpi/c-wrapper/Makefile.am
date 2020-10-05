noinst_LTLIBRARIES = libvftr_cmpiwrap.la

libvftr_cmpiwrap_la_SOURCES = vftr_mpi_init_cwrap.c \
			      vftr_mpi_finalize_cwrap.c

##################
# point to point #
##################
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_send_cwrap.c \
			       vftr_mpi_isend_cwrap.c \
			       vftr_mpi_bsend_cwrap.c \
			       vftr_mpi_ibsend_cwrap.c \
			       vftr_mpi_ssend_cwrap.c \
			       vftr_mpi_issend_cwrap.c \
			       vftr_mpi_rsend_cwrap.c \
			       vftr_mpi_irsend_cwrap.c \
			       vftr_mpi_recv_cwrap.c \
			       vftr_mpi_irecv_cwrap.c \
			       vftr_mpi_sendrecv_cwrap.c \
			       vftr_mpi_sendrecv_replace_cwrap.c \
			       vftr_mpi_probe_cwrap.c \
			       vftr_mpi_iprobe_cwrap.c

#########
# tests #
#########
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_test_cwrap.c \
			       vftr_mpi_testany_cwrap.c \
			       vftr_mpi_testsome_cwrap.c \
			       vftr_mpi_testall_cwrap.c

########
# wait #
########
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_wait_cwrap.c \
			       vftr_mpi_waitany_cwrap.c \
			       vftr_mpi_waitsome_cwrap.c \
			       vftr_mpi_waitall_cwrap.c

###############
# collectives #
###############
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_barrier_cwrap.c \
			       vftr_mpi_bcast_cwrap.c \
			       vftr_mpi_ibcast_cwrap.c \
			       vftr_mpi_gather_cwrap.c \
			       vftr_mpi_igather_cwrap.c \
			       vftr_mpi_gatherv_cwrap.c \
			       vftr_mpi_igatherv_cwrap.c \
			       vftr_mpi_scatter_cwrap.c \
			       vftr_mpi_iscatter_cwrap.c \
			       vftr_mpi_scatterv_cwrap.c \
			       vftr_mpi_iscatterv_cwrap.c \
			       vftr_mpi_allgather_cwrap.c \
			       vftr_mpi_iallgather_cwrap.c \
			       vftr_mpi_allgatherv_cwrap.c \
			       vftr_mpi_iallgatherv_cwrap.c \
			       vftr_mpi_alltoall_cwrap.c \
			       vftr_mpi_ialltoall_cwrap.c \
			       vftr_mpi_alltoallv_cwrap.c \
			       vftr_mpi_ialltoallv_cwrap.c \
			       vftr_mpi_alltoallw_cwrap.c \
			       vftr_mpi_ialltoallw_cwrap.c

####################
# global reduction #
####################
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_reduce_cwrap.c \
			       vftr_mpi_ireduce_cwrap.c \
			       vftr_mpi_allreduce_cwrap.c \
			       vftr_mpi_iallreduce_cwrap.c \
			       vftr_mpi_reduce_scatter_block_cwrap.c \
			       vftr_mpi_ireduce_scatter_block_cwrap.c \
			       vftr_mpi_reduce_scatter_cwrap.c \
			       vftr_mpi_ireduce_scatter_cwrap.c \
			       vftr_mpi_scan_cwrap.c \
			       vftr_mpi_iscan_cwrap.c \
			       vftr_mpi_exscan_cwrap.c \
			       vftr_mpi_iexscan_cwrap.c 

#############
# one sided #
#############
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_put_cwrap.c \
			       vftr_mpi_rput_cwrap.c \
			       vftr_mpi_get_cwrap.c \
			       vftr_mpi_rget_cwrap.c \
			       vftr_mpi_accumulate_cwrap.c \
			       vftr_mpi_raccumulate_cwrap.c \
			       vftr_mpi_get_accumulate_cwrap.c \
			       vftr_mpi_rget_accumulate_cwrap.c \
			       vftr_mpi_fetch_and_op_cwrap.c \
			       vftr_mpi_compare_and_swap_cwrap.c

############
# pcontrol #
############
libvftr_cmpiwrap_la_SOURCES += vftr_mpi_pcontrol_cwrap.c

libvftr_cmpiwrap_la_CFLAGS = -I${srcdir}/../
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../../
libvftr_cmpiwrap_la_CFLAGS += -finstrument-functions
if MPI_PROFILING
libvftr_cmpiwrap_la_CFLAGS += -D_MPI
endif
