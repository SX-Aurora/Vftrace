#ifndef PROBE_H
#define PROBE_H

#include <mpi.h>

int vftr_MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);

#endif
