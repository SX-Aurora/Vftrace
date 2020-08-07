ACLOCAL_AMFLAGS="-I m4"

noinst_LTLIBRARIES = libvftr_mpi.la

SUBDIRS =
if MPI_PROFILING
SUBDIRS += c-wrapper \
           f-wrapper \
           f08-wrapper
endif

libvftr_mpi_la_SOURCES = vftr_mpi_utils.c
if MPI_PROFILING
libvftr_mpi_la_SOURCES += vftr_sync_messages.c \
                          vftr_async_messages.c \
                          vftr_mpi_buf_addr_const.c \
                          vftr_mpi_buf_addr_const_F.F90 \
                          vftr_mpi_allgather.c \
                          vftr_mpi_allgather_f.c \
                          vftr_mpi_allgatherv.c \
                          vftr_mpi_allgatherv_f.c \
                          vftr_mpi_allreduce.c \
                          vftr_mpi_allreduce_f.c \
                          vftr_mpi_alltoall.c \
                          vftr_mpi_alltoall_f.c \
                          vftr_mpi_alltoallv.c \
                          vftr_mpi_alltoallv_f.c \
                          vftr_mpi_alltoallw.c \
                          vftr_mpi_alltoallw_f.c \
                          vftr_mpi_barrier.c \
                          vftr_mpi_barrier_f.c \
                          vftr_mpi_bcast.c \
                          vftr_mpi_bcast_f.c \
                          vftr_mpi_gather.c \
                          vftr_mpi_gather_f.c \
                          vftr_mpi_gatherv.c \
                          vftr_mpi_gatherv_f.c \
                          vftr_mpi_get.c \
                          vftr_mpi_get_f.c \
                          vftr_mpi_ialltoall.c \
                          vftr_mpi_ibcast.c \
                          vftr_mpi_irecv.c \
                          vftr_mpi_irecv_f.c \
                          vftr_mpi_isend.c \
                          vftr_mpi_isend_f.c \
                          vftr_mpi_pcontrol.c \
                          vftr_mpi_pcontrol_f08.c \
                          vftr_mpi_pcontrol_f.c \
                          vftr_mpi_put.c \
                          vftr_mpi_put_f.c \
                          vftr_mpi_recv.c \
                          vftr_mpi_recv_f.c \
                          vftr_mpi_reduce.c \
                          vftr_mpi_reduce_f.c \
                          vftr_mpi_rget.c \
                          vftr_mpi_rget_f.c \
                          vftr_mpi_rput.c \
                          vftr_mpi_rput_f.c \
                          vftr_mpi_scatter.c \
                          vftr_mpi_scatter_f.c \
                          vftr_mpi_scatterv.c \
                          vftr_mpi_scatterv_f.c \
                          vftr_mpi_send.c \
                          vftr_mpi_send_f.c \
                          vftr_mpi_sendrecv.c \
                          vftr_mpi_sendrecv_f.c \
                          vftr_mpi_sendrecv_replace.c \
                          vftr_mpi_sendrecv_replace_f.c \
                          vftr_mpi_testall.c \
                          vftr_mpi_testall_f.c \
                          vftr_mpi_testany.c \
                          vftr_mpi_testany_f.c \
                          vftr_mpi_test.c \
                          vftr_mpi_test_f.c \
                          vftr_mpi_testsome.c \
                          vftr_mpi_testsome_f.c \
                          vftr_mpi_waitall.c \
                          vftr_mpi_waitall_f.c \
                          vftr_mpi_waitany.c \
                          vftr_mpi_waitany_f.c \
                          vftr_mpi_wait.c \
                          vftr_mpi_wait_f.c \
                          vftr_mpi_waitsome.c \
                          vftr_mpi_waitsome_f.c
endif

libvftr_mpi_la_CFLAGS =
libvftr_mpi_la_FCFLAGS =

libvftr_mpi_la_CFLAGS += -I${srcdir}/../

if MPI_PROFILING
libvftr_mpi_la_CFLAGS += -D_MPI
libvftr_mpi_la_FCFLAGS += -D_MPI
endif

clean-local:
	rm -f *.mod
