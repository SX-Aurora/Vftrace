#ifndef SYNC_MESSAGES_H
#define SYNC_MESSAGES_H

#include "mpi_util_types.h"

void vftr_store_sync_message_info(message_direction dir, int count, MPI_Datatype type,
                                  int peer_rank, int tag, MPI_Comm comm,
                                  long long tstart, long long tend);

#endif
