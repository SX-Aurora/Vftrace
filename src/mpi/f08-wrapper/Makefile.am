noinst_LTLIBRARIES = libvftr_f08mpiwrap.la

libvftr_f08mpiwrap_la_SOURCES =

libvftr_f08mpiwrap_la_SOURCES += init.F90 \
				 finalize.F90

###################
## point to point #
###################
#libvftr_f08mpiwrap_la_SOURCES += send.F90 \
#				 bsend.F90 \
#				 ssend.F90 \
#				 rsend.F90
#
##########
## tests #
##########
#libvftr_f08mpiwrap_la_SOURCES +=
#
#########
## wait #
#########
#libvftr_f08mpiwrap_la_SOURCES +=

###############
# collectives #
###############
libvftr_f08mpiwrap_la_SOURCES += barrier.F90 \
				 bcast.F90 \
				 gather.F90 \
				 gatherv.F90 \
				 scatter.F90 \
				 scatterv.F90 \
				 allgather.F90 \
				 allgatherv.F90 \
				 alltoall.F90 \
				 alltoallv.F90 \
				 alltoallw.F90

####################
# global reduction #
####################
libvftr_f08mpiwrap_la_SOURCES += reduce.F90 \
				 allreduce.F90 \
				 reduce_scatter_block.F90 \
				 reduce_scatter.F90 \
				 scan.F90 
#				 exscan.F90
#
##############
## one sided #
##############
#libvftr_f08mpiwrap_la_SOURCES += put.F90 \
#				 get.F90 \
#				 accumulate.F90 \
#				 get_accumulate.F90 \
#				 fetch_and_op.F90 \
#				 compare_and_swap.F90
#
#############
## pcontrol #
#############
#libvftr_f08mpiwrap_la_SOURCES += pcontrol.F90

libvftr_f08mpiwrap_la_FCFLAGS = -finstrument-functions
libvftr_f08mpiwrap_la_FCFLAGS += -I${builddir}/../f082vftr/
libvftr_f08mpiwrap_la_FCFLAGS += -D_MPI

clean-local:
	rm -f *.mod
