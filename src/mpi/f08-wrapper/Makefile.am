noinst_LTLIBRARIES = libvftr_f08mpiwrap.la

libvftr_f08mpiwrap_la_SOURCES =

libvftr_f08mpiwrap_la_SOURCES += init.F90 \
				 finalize.F90

##################
# point to point #
##################
libvftr_f08mpiwrap_la_SOURCES += send.F90 \
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

##################
# persistent p2p #
##################
libvftr_f08mpiwrap_la_SOURCES += start.F90 \
				 startall.F90 \
				 send_init.F90 \
				 bsend_init.F90 \
				 ssend_init.F90 \
				 rsend_init.F90 \
				 recv_init.F90

#########
# tests #
#########
libvftr_f08mpiwrap_la_SOURCES += test.F90 \
				 testany.F90 \
				 testsome.F90 \
				 testall.F90

########
# wait #
########
libvftr_f08mpiwrap_la_SOURCES += wait.F90 \
				 waitany.F90 \
				 waitsome.F90 \
				 waitall.F90

########################
# request deallocation #
########################
libvftr_f08mpiwrap_la_SOURCES += request_free.F90

###############
# collectives #
###############
libvftr_f08mpiwrap_la_SOURCES += barrier.F90 \
				 ibarrier.F90 \
				 bcast.F90 \
				 ibcast.F90 \
				 gather.F90 \
				 igather.F90 \
				 gatherv.F90 \
				 igatherv.F90 \
				 scatter.F90 \
				 iscatter.F90 \
				 scatterv.F90 \
				 iscatterv.F90 \
				 allgather.F90 \
				 iallgather.F90 \
				 allgatherv.F90 \
				 iallgatherv.F90 \
				 alltoall.F90 \
				 ialltoall.F90 \
				 alltoallv.F90 \
				 ialltoallv.F90 \
				 alltoallw.F90 \
				 ialltoallw.F90

####################
# global reduction #
####################
libvftr_f08mpiwrap_la_SOURCES += reduce.F90 \
				 ireduce.F90 \
				 allreduce.F90 \
				 reduce_scatter_block.F90 \
				 reduce_scatter.F90 \
				 scan.F90 \
				 iscan.F90 \
				 exscan.F90 \
				 iexscan.F90

#############
# one sided #
#############
libvftr_f08mpiwrap_la_SOURCES += put.F90 \
				 rput.F90 \
				 get.F90 \
				 rget.F90 \
				 accumulate.F90 \
				 raccumulate.F90 \
				 get_accumulate.F90 \
				 rget_accumulate.F90 \
				 fetch_and_op.F90 \
				 compare_and_swap.F90

############
# pcontrol #
############
libvftr_f08mpiwrap_la_SOURCES += pcontrol.F90

libvftr_f08mpiwrap_la_FCFLAGS = -finstrument-functions
libvftr_f08mpiwrap_la_FCFLAGS += -I${builddir}/../utils/
libvftr_f08mpiwrap_la_FCFLAGS += -I${builddir}/../f082vftr/
libvftr_f08mpiwrap_la_FCFLAGS += -D_MPI

clean-local:
	rm -f *.mod
