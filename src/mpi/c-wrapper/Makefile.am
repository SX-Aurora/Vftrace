noinst_LTLIBRARIES = libvftr_cmpiwrap.la

libvftr_cmpiwrap_la_SOURCES = init.c \
                              finalize.c

##################
# point to point #
##################
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

##################
# persistent p2p #
##################
libvftr_cmpiwrap_la_SOURCES += start.c \
                               startall.c \
                               send_init.c \
                               bsend_init.c \
                               ssend_init.c \
                               rsend_init.c \
                               recv_init.c 

#########
# tests #
#########
libvftr_cmpiwrap_la_SOURCES += test.c \
                               testany.c \
                               testsome.c \
                               testall.c

########
# wait #
########
libvftr_cmpiwrap_la_SOURCES += wait.c \
                               waitany.c \
                               waitsome.c \
                               waitall.c

########################
# request deallocation #
########################
libvftr_cmpiwrap_la_SOURCES += request_free.c

###############
# collectives #
###############
libvftr_cmpiwrap_la_SOURCES += barrier.c \
                               ibarrier.c \
                               bcast.c \
                               ibcast.c \
                               gather.c \
                               igather.c \
			       gatherv.c \
                               igatherv.c \
                               scatter.c \
                               iscatter.c \
                               scatterv.c \
                               iscatterv.c \
                               allgather.c 
#                               iallgather.c \
#                               allgatherv.c \
#                               iallgatherv.c \
#                               alltoall.c \
#                               ialltoall.c \
#                               alltoallv.c \
#                               ialltoallv.c \
#                               alltoallw.c \
#                               ialltoallw.c

####################
# global reduction #
####################
libvftr_cmpiwrap_la_SOURCES += reduce.c \
                               ireduce.c \
                               allreduce.c \
                               iallreduce.c \
                               reduce_scatter_block.c \
                               ireduce_scatter_block.c \
                               reduce_scatter.c \
                               ireduce_scatter.c \
                               scan.c \
                               iscan.c \
                               exscan.c \
                               iexscan.c 

#############
# one sided #
#############
libvftr_cmpiwrap_la_SOURCES += put.c \
                               rput.c \
                               get.c \
                               rget.c \
                               accumulate.c \
                               raccumulate.c \
                               get_accumulate.c \
                               rget_accumulate.c \
                               fetch_and_op.c \
                               compare_and_swap.c

############
# pcontrol #
############
libvftr_cmpiwrap_la_SOURCES += pcontrol.c

libvftr_cmpiwrap_la_CFLAGS = -D_MPI
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../../
libvftr_cmpiwrap_la_CFLAGS += -I${srcdir}/../c2vftr/
libvftr_cmpiwrap_la_CFLAGS += -finstrument-functions
