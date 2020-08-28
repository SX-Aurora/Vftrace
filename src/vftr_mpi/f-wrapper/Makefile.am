noinst_LTLIBRARIES = libvftr_fmpiwrap.la

libvftr_fmpiwrap_la_SOURCES = 

libvftr_fmpiwrap_la_SOURCES += vftr_after_mpi_init_c2f.F90 \
			       vftr_finalize_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_init_fwrap.F90 \
			       vftr_mpi_finalize_fwrap.F90

##################
# point to point #
##################
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_irecv_c2f.F90 \
			       vftr_mpi_isend_c2f.F90 \
			       vftr_mpi_recv_c2f.F90 \
			       vftr_mpi_send_c2f.F90 \
			       vftr_mpi_sendrecv_c2f.F90 \
			       vftr_mpi_sendrecv_replace_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_irecv_fwrap.F90 \
			       vftr_mpi_isend_fwrap.F90 \
			       vftr_mpi_recv_fwrap.F90 \
			       vftr_mpi_send_fwrap.F90 \
			       vftr_mpi_sendrecv_fwrap.F90 \
			       vftr_mpi_sendrecv_replace_fwrap.F90

#########
# tests #
#########
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_test_c2f.F90 \
			       vftr_mpi_testany_c2f.F90 \
			       vftr_mpi_testsome_c2f.F90 \
			       vftr_mpi_testall_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_test_fwrap.F90 \
			       vftr_mpi_testany_fwrap.F90 \
			       vftr_mpi_testsome_fwrap.F90 \
			       vftr_mpi_testall_fwrap.F90

########
# wait #
########
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_wait_c2f.F90 \
			       vftr_mpi_waitany_c2f.F90 \
			       vftr_mpi_waitsome_c2f.F90 \
			       vftr_mpi_waitall_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_wait_fwrap.F90 \
			       vftr_mpi_waitany_fwrap.F90 \
			       vftr_mpi_waitsome_fwrap.F90 \
			       vftr_mpi_waitall_fwrap.F90

###############
# collectives #
###############
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_barrier_c2f.F90 \
			       vftr_mpi_bcast_c2f.F90 \
			       vftr_mpi_ibcast_c2f.F90 \
			       vftr_mpi_gather_c2f.F90 \
			       vftr_mpi_igather_c2f.F90 \
			       vftr_mpi_gatherv_c2f.F90 \
			       vftr_mpi_igatherv_c2f.F90 \
			       vftr_mpi_scatter_c2f.F90 \
			       vftr_mpi_scatterv_c2f.F90 \
			       vftr_mpi_allgather_c2f.F90 \
			       vftr_mpi_allgatherv_c2f.F90 \
			       vftr_mpi_alltoall_c2f.F90 \
			       vftr_mpi_alltoallv_c2f.F90 \
			       vftr_mpi_alltoallw_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_barrier_fwrap.F90 \
			       vftr_mpi_bcast_fwrap.F90 \
			       vftr_mpi_ibcast_fwrap.F90 \
			       vftr_mpi_gather_fwrap.F90 \
			       vftr_mpi_igather_fwrap.F90 \
			       vftr_mpi_gatherv_fwrap.F90 \
			       vftr_mpi_igatherv_fwrap.F90 \
			       vftr_mpi_scatter_fwrap.F90 \
			       vftr_mpi_scatterv_fwrap.F90 \
			       vftr_mpi_allgather_fwrap.F90 \
			       vftr_mpi_allgatherv_fwrap.F90 \
			       vftr_mpi_alltoall_fwrap.F90 \
			       vftr_mpi_alltoallv_fwrap.F90 \
			       vftr_mpi_alltoallw_fwrap.F90

####################
# global reduction #
####################
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_allreduce_c2f.F90 \
			       vftr_mpi_reduce_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_allreduce_fwrap.F90 \
			       vftr_mpi_reduce_fwrap.F90

#############
# one sided #
#############
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_put_c2f.F90 \
			       vftr_mpi_get_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_put_fwrap.F90 \
			       vftr_mpi_get_fwrap.F90

############
# pcontrol #
############
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_pcontrol_c2f.F90
libvftr_fmpiwrap_la_SOURCES += vftr_mpi_pcontrol_fwrap.F90

libvftr_fmpiwrap_la_FCFLAGS = -finstrument-functions
if MPI_PROFILING
libvftr_fmpiwrap_la_FCFLAGS += -D_MPI
endif

if USES_INTEL_MPI
libvftr_fmpiwrap_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftr_fmpiwrap_la_FCFLAGS += -DPMPI_MODULE=mpi
endif

clean-local:
	rm -f *.mod
