#ifndef VFTR_REQUEST_TYPES_H
#define VFTR_REQUEST_TYPES_H

#include <stdbool.h>

#include <mpi.h>

#include "mpi_util_types.h"

typedef enum vftr_request_kind_t {
   p2p,
   collective,
   onesided
} vftr_request_kind;

// open requests
typedef struct vftr_request_type {
   bool valid;
   bool persistent;
   bool active;
   bool marked_for_deallocation;
   MPI_Request request;
   vftr_request_kind request_kind;
   MPI_Comm comm;
   int nmsg;
   message_direction dir;
   int *count;
   MPI_Datatype *type;
   int *type_idx;
   int *type_size;
   int *rank;
   int tag;
   long long tstart;
   int callingstackID;
   int callingthreadID;
   int n_tmp_ptr;
   void **tmp_ptrs;
} vftr_request_t;

#endif
