#ifndef VFTR_P2P_REQUESTS_H
#define VFTR_P2P_REQUESTS_H

#include "requests.h"

void vftr_register_p2p_request(message_direction dir, int count,
                               MPI_Datatype type, int peer_rank, int tag,
                               MPI_Comm comm, MPI_Request request,
                               long long tstart);

void vftr_register_pers_p2p_request(message_direction dir, int count,
                                    MPI_Datatype type, int peer_rank, int tag,
                                    MPI_Comm comm, MPI_Request request);

void vftr_clear_completed_p2p_request(vftr_request_t *request);

#endif
