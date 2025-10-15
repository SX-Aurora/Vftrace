#ifndef _MPI_CONTROL_H
#define _MPI_CONTROL_H

#define _MPI_CALL(prefix, func) prefix##_##func

#ifdef _MPI_REDUCED
#define MPI_CALL(func) _MPI_CALL(MPI, func)
#else
#define MPI_CALL(func) _MPI_CALL(PMPI, func)
#endif

#endif
