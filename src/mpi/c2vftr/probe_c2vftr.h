#ifndef PROBE_C2VFTR_H
#define PROBE_C2VFTR_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Probe_c2vftr(int source, int tag, MPI_Comm comm, MPI_Status *status);

#endif
#endif
