noinst_LTLIBRARIES = libvftr_cmpiwrap.la

libvftr_cmpiwrap_la_SOURCES =

if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += init.c \
			       finalize.c
endif
if HAS_MPI_STD_2_1
libvftr_cmpiwrap_la_SOURCES += init_thread.c
endif

##################
# point to point #
##################
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += send.c \
                               isend.c \
                               bsend.c \
                               ibsend.c \
                               ssend.c \
                               issend.c \
                               rsend.c \
                               irsend.c \
                               recv.c \
                               irecv.c \
                               sendrecv.c \
                               sendrecv_replace.c \
                               probe.c \
                               iprobe.c
endif

##################
# persistent p2p #
##################
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += start.c \
                               startall.c \
                               send_init.c \
                               bsend_init.c \
                               ssend_init.c \
                               rsend_init.c \
                               recv_init.c
endif

#########
# tests #
#########
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += test.c \
                               testany.c \
                               testsome.c \
                               testall.c
endif

########
# wait #
########
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += wait.c \
                               waitany.c \
                               waitsome.c \
                               waitall.c
endif

########################
# request deallocation #
########################
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += request_free.c
endif

###############
# collectives #
###############
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += barrier.c \
			       bcast.c \
			       gather.c \
			       gatherv.c \
			       scatter.c \
			       scatterv.c \
			       allgather.c \
			       allgatherv.c \
			       alltoall.c \
			       alltoallv.c
endif
if HAS_MPI_STD_2_0
libvftr_cmpiwrap_la_SOURCES += alltoallw.c
endif
if HAS_MPI_STD_3_0
libvftr_cmpiwrap_la_SOURCES += ibarrier.c \
			       ibcast.c \
			       igather.c \
			       igatherv.c \
			       iscatter.c \
			       iscatterv.c \
			       iallgather.c \
			       iallgatherv.c \
			       ialltoall.c \
			       ialltoallv.c \
			       ialltoallw.c
endif

########################
# neighbor collectives #
########################
if HAS_MPI_STD_3_0
libvftr_cmpiwrap_la_SOURCES += neighbor_allgather.c \
			       ineighbor_allgather.c \
			       neighbor_allgatherv.c \
			       ineighbor_allgatherv.c \
			       neighbor_alltoall.c \
			       ineighbor_alltoall.c \
			       neighbor_alltoallv.c \
			       ineighbor_alltoallv.c \
			       neighbor_alltoallw.c \
			       ineighbor_alltoallw.c
endif

####################
# global reduction #
####################
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += reduce.c \
			       allreduce.c \
			       reduce_scatter.c \
			       scan.c
endif
if HAS_MPI_STD_2_0
libvftr_cmpiwrap_la_SOURCES += exscan.c
endif
if HAS_MPI_STD_2_2
libvftr_cmpiwrap_la_SOURCES += reduce_scatter_block.c
endif
if HAS_MPI_STD_3_0
libvftr_cmpiwrap_la_SOURCES += ireduce.c \
			       iallreduce.c \
			       ireduce_scatter_block.c \
			       ireduce_scatter.c \
			       iscan.c \
			       iexscan.c 
endif

#############
# one sided #
#############
if HAS_MPI_STD_2_0
libvftr_cmpiwrap_la_SOURCES += get.c \
			       put.c \
			       accumulate.c
endif
if HAS_MPI_STD_3_0
libvftr_cmpiwrap_la_SOURCES += rget.c \
			       rput.c \
			       raccumulate.c \
			       get_accumulate.c \
			       rget_accumulate.c \
			       fetch_and_op.c \
			       compare_and_swap.c
endif

############
# pcontrol #
############
if HAS_MPI_STD_1_0
libvftr_cmpiwrap_la_SOURCES += pcontrol.c
endif


libvftr_cmpiwrap_la_CFLAGS = -D_MPI
if ENABLE_CUDAPROF
libvftr_cmpiwrap_la_CFLAGS += -D_CUDA
endif

if ENABLE_ACCPROF
libvftr_cmpiwrap_la_CFLAGS += -D_ACCPROF
endif

libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../c2vftr/
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../utils/
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../../
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../../cuda
libvftr_cmpiwrap_la_CFLAGS += $(FINSTRUMENT_FLAG)
