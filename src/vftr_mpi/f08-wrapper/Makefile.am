noinst_LTLIBRARIES = libvftr_f08mpiwrap.la

libvftr_f08mpiwrap_la_SOURCES =

libvftr_f08mpiwrap_la_SOURCES += vftr_after_mpi_init_c2f08.F90 \
				 vftr_finalize_c2f08.F90
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_init_f08wrap.F90 \
				 vftr_mpi_finalize_f08wrap.F90

##################
# point to point #
##################
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_send_f08wrap.F90 \
				 vftr_mpi_bsend_f08wrap.F90 \
				 vftr_mpi_ssend_f08wrap.F90 \
				 vftr_mpi_rsend_f08wrap.F90

#########
# tests #
#########
libvftr_f08mpiwrap_la_SOURCES +=

########
# wait #
########
libvftr_f08mpiwrap_la_SOURCES +=

###############
# collectives #
###############
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_barrier_f08wrap.F90 \
				 vftr_mpi_bcast_f08wrap.F90 \
				 vftr_mpi_gather_f08wrap.F90 \
				 vftr_mpi_gatherv_f08wrap.F90 \
				 vftr_mpi_scatter_f08wrap.F90 \
				 vftr_mpi_scatterv_f08wrap.F90 \
				 vftr_mpi_allgather_f08wrap.F90 \
				 vftr_mpi_allgatherv_f08wrap.F90 \
				 vftr_mpi_alltoall_f08wrap.F90 \
				 vftr_mpi_alltoallv_f08wrap.F90 \
				 vftr_mpi_alltoallw_f08wrap.F90

####################
# global reduction #
####################
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_reduce_f08wrap.F90 \
				 vftr_mpi_allreduce_f08wrap.F90 \
				 vftr_mpi_reduce_scatter_block_f08wrap.F90 \
				 vftr_mpi_reduce_scatter_f08wrap.F90 \
				 vftr_mpi_scan_f08wrap.F90 \
				 vftr_mpi_exscan_f08wrap.F90

#############
# one sided #
#############
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_put_f08wrap.F90 \
				 vftr_mpi_get_f08wrap.F90 \
				 vftr_mpi_accumulate_f08wrap.F90 \
				 vftr_mpi_get_accumulate_f08wrap.F90 \
				 vftr_mpi_fetch_and_op_f08wrap.F90 \
				 vftr_mpi_compare_and_swap_f08wrap.F90

############
# pcontrol #
############
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_pcontrol_c2f08.F90
libvftr_f08mpiwrap_la_SOURCES += vftr_mpi_pcontrol_f08wrap.F90

libvftr_f08mpiwrap_la_FCFLAGS = -finstrument-functions
libvftr_f08mpiwrap_la_FCFLAGS += -I${builddir}/../f-wrapper/
if MPI_PROFILING
libvftr_f08mpiwrap_la_FCFLAGS += -D_MPI
endif

if USES_INTEL_MPI
libvftr_f08mpiwrap_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftr_f08mpiwrap_la_FCFLAGS += -DPMPI_MODULE=mpi
endif

clean-local:
	rm -f *.mod
