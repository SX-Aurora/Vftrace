#ifndef VFTR_COLLECTIVE_REQUESTS_H
#define VFTR_COLLECTIVE_REQUESTS_H

#include "requests.h"

void vftr_register_collective_request(message_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int *peer_rank,
                                      MPI_Comm comm, MPI_Request request,
                                      int n_tmp_ptr, void **tmp_ptrs,
                                      long long tstart);

void vftr_clear_completed_collective_request(vftr_request_t *request);

#endif
