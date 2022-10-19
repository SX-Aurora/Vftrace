#ifndef VFTR_REQUESTS_H
#define VFTR_REQUESTS_H

#include <mpi.h>

#include "request_types.h"
#include "mpi_state_types.h"
#include "mpi_util_types.h"

// create new request to be stored
vftr_request_t* vftr_register_request(message_direction dir, int nmsg, int *count,
                                      MPI_Datatype *type, int tag,
                                      MPI_Comm comm, MPI_Request request,
                                      int n_tmp_ptr, void **tmp_ptrs,
                                      long long tstart);

// clear the requests and log the messaging
void vftr_clear_completed_requests();
void vftr_clear_completed_requests_from_hooks();
void vftr_clear_completed_requests_from_test();
void vftr_clear_completed_requests_from_wait();

void vftr_activate_pers_request(MPI_Request request, long long tstart);

void vftr_remove_request(vftr_request_t *request);

// deallocate entire request list
void vftr_free_request_list(mpi_state_t *mpi_state);

// find a specific request in the request list.
vftr_request_t *vftr_search_request(MPI_Request request);

#endif
