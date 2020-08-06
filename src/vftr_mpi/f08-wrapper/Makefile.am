noinst_LTLIBRARIES = libvftr_f08mpiwrap.la

libvftr_f08mpiwrap_la_SOURCES = vftr_mpi_environment_c2F08.F90 \
				vftr_mpi_environment_F08wrap.F90 \
				vftr_mpi_point2point_F08wrap.F90 \
				vftr_mpi_collective_F08wrap.F90 \
				vftr_mpi_onesided_F08wrap.F90


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
