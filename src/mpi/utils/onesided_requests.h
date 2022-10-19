#ifndef VFTR_ONESIDED_REQUESTS_H
#define VFTR_ONESIDED_REQUESTS_H

#include "requests.h"

void vftr_register_onesided_request(message_direction dir, int count,
                                    MPI_Datatype type, int peer_rank,
                                    MPI_Comm comm, MPI_Request request,
                                    long long tstart);

void vftr_clear_completed_onesided_request(vftr_request_t *request);

#endif
