noinst_LTLIBRARIES = libvftr_fmpiwrap.la

libvftr_fmpiwrap_la_SOURCES = vftr_mpi_environment_c2F.F90 \
			      vftr_mpi_environment_Fwrap.F90 \
			      vftr_mpi_point2point_c2F.F90 \
			      vftr_mpi_point2point_Fwrap.F90 \
			      vftr_mpi_collective_c2F.F90 \
			      vftr_mpi_collective_Fwrap.F90 \
			      vftr_mpi_onesided_c2F.F90 \
			      vftr_mpi_onesided_Fwrap.F90 \
			      vftr_mpi_testwait_c2F.F90 \
			      vftr_mpi_testwait_Fwrap.F90

libvftr_fmpiwrap_la_FCFLAGS = -finstrument-functions
if MPI_PROFILING
libvftr_fmpiwrap_la_FCFLAGS += -D_MPI
endif

if USES_INTEL_MPI
libvftr_fmpiwrap_la_FCFLAGS += -DPMPI_MODULE=pmpi_f08
else
libvftr_fmpiwrap_la_FCFLAGS += -DPMPI_MODULE=mpi
endif
