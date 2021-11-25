noinst_LTLIBRARIES = libvftr_fmpiwrap.la

libvftr_fmpiwrap_la_SOURCES =

if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += init.F90 \
			       finalize.F90
endif

##################
# point to point #
##################
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += send.F90 \
			       isend.F90 \
			       bsend.F90 \
			       ibsend.F90 \
			       ssend.F90 \
			       issend.F90 \
			       rsend.F90 \
			       irsend.F90 \
			       recv.F90 \
			       irecv.F90 \
			       sendrecv.F90 \
			       sendrecv_replace.F90 \
			       probe.F90 \
			       iprobe.F90
endif

##################
# persistent p2p #
##################
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += start.F90 \
			       startall.F90 \
			       send_init.F90 \
			       bsend_init.F90 \
			       ssend_init.F90 \
			       rsend_init.F90 \
			       recv_init.F90
endif

#########
# tests #
#########
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += test.F90 \
			       testany.F90 \
			       testsome.F90 \
			       testall.F90
endif

########
# wait #
########
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += wait.F90 \
			       waitany.F90 \
			       waitsome.F90 \
			       waitall.F90
endif

########################
# request deallocation #
########################
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += request_free.F90
endif

###############
# collectives #
###############
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += barrier.F90 \
			       bcast.F90 \
			       gather.F90 \
			       gatherv.F90 \
			       scatter.F90 \
			       scatterv.F90 \
			       allgather.F90 \
			       allgatherv.F90 \
			       alltoall.F90 \
			       alltoallv.F90
endif
if HAS_MPI_STD_2_0
libvftr_fmpiwrap_la_SOURCES += alltoallw.F90
endif
if HAS_MPI_STD_3_0
libvftr_fmpiwrap_la_SOURCES += ibarrier.F90 \
			       ibcast.F90 \
			       igather.F90 \
			       igatherv.F90 \
			       iscatter.F90 \
			       iscatterv.F90 \
			       iallgather.F90 \
			       iallgatherv.F90 \
			       ialltoall.F90 \
			       ialltoallv.F90 \
			       ialltoallw.F90
endif

####################
# global reduction #
####################
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += reduce.F90 \
			       allreduce.F90 \
			       reduce_scatter.F90 \
			       scan.F90
endif
if HAS_MPI_STD_2_0
libvftr_fmpiwrap_la_SOURCES += exscan.F90
endif
if HAS_MPI_STD_2_2
libvftr_fmpiwrap_la_SOURCES += reduce_scatter_block.F90
endif
if HAS_MPI_STD_3_0
libvftr_fmpiwrap_la_SOURCES += ireduce_scatter.F90 \
			       ireduce.F90 \
			       iallreduce.F90 \
			       ireduce_scatter_block.F90 \
			       ireduce_scatter.F90 \
			       iscan.F90 \
			       iexscan.F90 
endif

#############
# one sided #
#############
if HAS_MPI_STD_2_0
libvftr_fmpiwrap_la_SOURCES += get.F90 \
			       put.F90 \
			       accumulate.F90
endif
if HAS_MPI_STD_3_0
libvftr_fmpiwrap_la_SOURCES += rget.F90 \
			       rput.F90 \
			       raccumulate.F90 \
			       get_accumulate.F90 \
			       rget_accumulate.F90 \
			       fetch_and_op.F90 \
			       compare_and_swap.F90
endif

############
# pcontrol #
############
if HAS_MPI_STD_1_0
libvftr_fmpiwrap_la_SOURCES += pcontrol.F90
endif

libvftr_fmpiwrap_la_FCFLAGS = -D_MPI
libvftr_fmpiwrap_la_FCFLAGS += -I${builddir}/../f2vftr/ \
			       -I${builddir}/../utils/
libvftr_fmpiwrap_la_FCFLAGS += -finstrument-functions

clean-local:
	rm -f *.mod
