/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdbool.h>
#include <stdio.h>

#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_environment.h"
#include "vftr_setup.h"
#include "vftr_hwcounters.h"
#include "vftr_filewrite.h"
#include "vftr_pause.h"
#include "vftr_mpi_utils.h"
#include "vftr_mpi_pcontrol.h"

void vftr_reset_counts (function_t *func);

// set of all C-MPI defined datatype including the derived type
#define NVFTR_TYPES 38

#ifdef _MPI
   #include <mpi.h>
#endif

struct vftr_mpi_type_t {
#ifdef _MPI
   MPI_Datatype mpi_type;
#else
   int mpi_type;
#endif
   char *name;
};
//define all mpi types if mpi is not used to allow for the usage of types in programs like tracedump
#ifndef _MPI
   #define MPI_CHAR                  1
   #define MPI_SHORT                 1
   #define MPI_INT                   1
   #define MPI_LONG                  1
   #define MPI_LONG_LONG_INT         1
   #define MPI_LONG_LONG             1
   #define MPI_SIGNED_CHAR           1
   #define MPI_UNSIGNED_CHAR         1
   #define MPI_UNSIGNED_SHORT        1
   #define MPI_UNSIGNED              1
   #define MPI_UNSIGNED_LONG         1
   #define MPI_UNSIGNED_LONG_LONG    1
   #define MPI_FLOAT                 1
   #define MPI_DOUBLE                1
   #define MPI_LONG_DOUBLE           1
   #define MPI_WCHAR                 1
   #define MPI_C_BOOL                1
   #define MPI_INT8_T                1
   #define MPI_INT16_T               1
   #define MPI_INT32_T               1
   #define MPI_INT64_T               1
   #define MPI_UINT8_T               1
   #define MPI_UINT16_T              1
   #define MPI_UINT32_T              1
   #define MPI_UINT64_T              1
   #define MPI_C_COMPLEX             1
   #define MPI_C_FLOAT_COMPLEX       1
   #define MPI_C_DOUBLE_COMPLEX      1
   #define MPI_C_LONG_DOUBLE_COMPLEX 1
   #define MPI_INTEGER               1
   #define MPI_LOGICAL               1
   #define MPI_REAL                  1
   #define MPI_DOUBLE_PRECISION      1
   #define MPI_COMPLEX               1
   #define MPI_CHARACTER             1
   #define MPI_BYTE                  1
   #define MPI_PACKED                1
#endif

struct vftr_mpi_type_t all_mpi_types[NVFTR_TYPES] = {
   {.mpi_type = MPI_CHAR,                  .name = "MPI_CHAR" },
   {.mpi_type = MPI_SHORT,                 .name = "MPI_SHORT" },
   {.mpi_type = MPI_INT,                   .name = "MPI_INT" },
   {.mpi_type = MPI_LONG,                  .name = "MPI_LONG" },
   {.mpi_type = MPI_LONG_LONG_INT,         .name = "MPI_LONG_LONG_INT" },
   {.mpi_type = MPI_LONG_LONG,             .name = "MPI_LONG_LONG" },
   {.mpi_type = MPI_SIGNED_CHAR,           .name = "MPI_SIGNED_CHAR" },
   {.mpi_type = MPI_UNSIGNED_CHAR,         .name = "MPI_UNSIGNED_CHAR" },
   {.mpi_type = MPI_UNSIGNED_SHORT,        .name = "MPI_UNSIGNED_SHORT" },
   {.mpi_type = MPI_UNSIGNED,              .name = "MPI_UNSIGNED" },
   {.mpi_type = MPI_UNSIGNED_LONG,         .name = "MPI_UNSIGNED_LONG" },
   {.mpi_type = MPI_UNSIGNED_LONG_LONG,    .name = "MPI_UNSIGNED_LONG_LONG" },
   {.mpi_type = MPI_FLOAT,                 .name = "MPI_FLOAT" },
   {.mpi_type = MPI_DOUBLE,                .name = "MPI_DOUBLE" },
   {.mpi_type = MPI_LONG_DOUBLE,           .name = "MPI_LONG_DOUBLE" },
   {.mpi_type = MPI_WCHAR,                 .name = "MPI_WCHAR" },
   {.mpi_type = MPI_C_BOOL,                .name = "MPI_C_BOOL" },
   {.mpi_type = MPI_INT8_T,                .name = "MPI_INT8_T" },
   {.mpi_type = MPI_INT16_T,               .name = "MPI_INT16_T" },
   {.mpi_type = MPI_INT32_T,               .name = "MPI_INT32_T" },
   {.mpi_type = MPI_INT64_T,               .name = "MPI_INT64_T" },
   {.mpi_type = MPI_UINT8_T,               .name = "MPI_UINT8_T" },
   {.mpi_type = MPI_UINT16_T,              .name = "MPI_UINT16_T" },
   {.mpi_type = MPI_UINT32_T,              .name = "MPI_UINT32_T" },
   {.mpi_type = MPI_UINT64_T,              .name = "MPI_UINT64_T" },
   {.mpi_type = MPI_C_COMPLEX,             .name = "MPI_C_COMPLEX" },
   {.mpi_type = MPI_C_FLOAT_COMPLEX,       .name = "MPI_C_FLOAT_COMPLEX" },
   {.mpi_type = MPI_C_DOUBLE_COMPLEX,      .name = "MPI_C_DOUBLE_COMPLEX" },
   {.mpi_type = MPI_C_LONG_DOUBLE_COMPLEX, .name = "MPI_C_LONG_DOUBLE_COMPLEX" },
   {.mpi_type = MPI_INTEGER,               .name = "MPI_INTEGER" },
   {.mpi_type = MPI_LOGICAL,               .name = "MPI_LOGICAL" },
   {.mpi_type = MPI_REAL,                  .name = "MPI_REAL" }, 
   {.mpi_type = MPI_DOUBLE_PRECISION,      .name = "MPI_DOUBLE_PRECISION" }, 
   {.mpi_type = MPI_COMPLEX,               .name = "MPI_COMPLEX" }, 
   {.mpi_type = MPI_CHARACTER,             .name = "MPI_CHARACTER" }, 
   {.mpi_type = MPI_BYTE,                  .name = "MPI_BYTE" },
   {.mpi_type = MPI_PACKED,                .name = "MPI_PACKED" }
};

#ifdef _MPI
// Adjust vftrace to the just initialized MPI-environment
void vftr_after_mpi_init() {
   if (vftr_off()) return;
   vftr_reset_counts (vftr_froots);
   if (!vftr_env_do_sampling ()) return;

   vftr_prevsampletime = 0;
   vftr_nextsampletime = 0ll;
   vftr_function_samplecount = 0;
   vftr_message_samplecount = 0;
   vftr_prog_cycles = 0ll;

   fseek (vftr_vfd_file, vftr_samples_offset, SEEK_SET);
}

// Translates an MPI-Datatype into the vftr type index
int vftr_get_mpitype_idx(MPI_Datatype mpi_type) {
   for (int i=0; i<NVFTR_TYPES; i++) {
      if (mpi_type == all_mpi_types[i].mpi_type) {
         return i;
      }
   }
   // if no datatype could be found return the derived datatype idx
   return -1;
}

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string(MPI_Datatype mpi_type) {
   int idx = vftr_get_mpitype_idx(mpi_type);
   if (idx >= 0 && idx < NVFTR_TYPES) {
      // search for MPI_Type string was successful
      return all_mpi_types[idx].name;
   } else if (idx == -1) {
      // search failed.
      // MPI_Type can not be a standard type
      return "MPI_DERIVED_TYPE";
   } else {
      return "MPI_UNDEFINED_TYPE";
   }
}

// determine based on several criteria if
// the communication should just be executed or also logged
bool vftr_no_mpi_logging() {
   return vftrace_Pcontrol_level == 0 || 
          vftr_off() ||
          !vftr_environment.mpi_log->value ||
          vftr_paused;
}
// int version of above function for well defined fortran-interoperability
int vftr_no_mpi_logging_int() {
   return vftr_no_mpi_logging() ? 1 : 0;
}

// Translate a rank from a local group to the global rank
int vftr_local2global_rank(MPI_Comm comm, int local_rank) {

   // no translation of ranks neccessary as it is already in the global group
   if (comm == MPI_COMM_WORLD) {
      return local_rank;
   }
   
   // determine global group for the global communicator once
   static MPI_Group global_group = MPI_GROUP_NULL;
   if (global_group == MPI_GROUP_NULL) {
      PMPI_Comm_group(MPI_COMM_WORLD, &global_group);
   }

   // determine the local group for the given communicator
   MPI_Group local_group;
   PMPI_Comm_group(comm, &local_group);

   // Translate rank from local to global group.
   int global_rank;
   PMPI_Group_translate_ranks(local_group,   // local group
                              1,             // number of ranks to translate 
                              &local_rank,   // addr of local rank variable
                              global_group,  // group to translate rank to
                              &global_rank); // addr of global rank variable
   return global_rank;
}

// Translate a rank from a remote group to the global rank
int vftr_remote2global_rank(MPI_Comm comm, int remote_rank) {

   // no translation of ranks neccessary as it is already in the global group
   if (comm == MPI_COMM_WORLD) {
      return remote_rank;
   }
   
   // determine global group for the global communicator once
   static MPI_Group global_group = MPI_GROUP_NULL;
   if (global_group == MPI_GROUP_NULL) {
      PMPI_Comm_group(MPI_COMM_WORLD, &global_group);
   }

   // determine the remote group for the given communicator
   MPI_Group remote_group;
   PMPI_Comm_remote_group(comm, &remote_group);

   // Translate rank from remote to global group.
   int global_rank;
   PMPI_Group_translate_ranks(remote_group,  // remote group
                              1,             // number of ranks to translate 
                              &remote_rank,  // addr of remote rank variable
                              global_group,  // group to translate rank to
                              &global_rank); // addr of global rank variable
   return global_rank;
}
#endif

// Converts an mpi-datatype into a name string for that type
const char *vftr_get_mpitype_string_from_idx(int mpi_type_idx) {
   if (mpi_type_idx >= 0 && mpi_type_idx < NVFTR_TYPES) {
      return all_mpi_types[mpi_type_idx].name;
   } else if (mpi_type_idx == -1) {
      return "MPI_DERIVED_TYPE";
   }
   return "MPI_UNDEFINED_TYPE";
}

#ifdef _MPI
// mark a MPI_Status as empty
void vftr_empty_mpi_status(MPI_Status *status) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // A status is empty if its members have the following values:
   // MPI_TAG == MPI_ANY_TAG, MPI_SOURCE == MPI_ANY_SOURCE, MPI_ERROR == MPI_SUCCESS
   status->MPI_TAG = MPI_ANY_TAG;
   status->MPI_SOURCE = MPI_ANY_SOURCE;
   status->MPI_ERROR = MPI_SUCCESS;
   return;
}

// check if a status is empty
bool vftr_mpi_status_is_empty(MPI_Status *status) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // a status is empty if its members have the following values:
   // MPI_TAG == MPI_ANY_TAG, MPI_SOURCE == MPI_ANY_SOURCE, MPI_ERROR == MPI_SUCCESS
   return (status->MPI_TAG == MPI_ANY_TAG &&
           status->MPI_SOURCE == MPI_ANY_SOURCE &&
           status->MPI_ERROR == MPI_SUCCESS);
}

// check if a request is active
bool vftr_mpi_request_is_active(MPI_Request request) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // a request is active if it is neither a null request
   // nor returns an empty status for Request_get_status 
   // (the function returns an empty status if it is inactive)
   
   if (request == MPI_REQUEST_NULL) {
      return false;
   }

   MPI_Status status;
   int flag;
   PMPI_Request_get_status(request, &flag, &status);

   return !vftr_mpi_status_is_empty(&status);
}

#endif

