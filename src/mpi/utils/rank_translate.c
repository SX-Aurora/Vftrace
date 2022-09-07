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

#include <mpi.h>

#include "self_profile.h"

// Translate a rank from a local group to the global rank
int vftr_local2global_rank(MPI_Comm comm, int local_rank) {
   SELF_PROFILE_START_FUNCTION;
   // no translation of ranks neccessary as it is already in the global group
   if (comm == MPI_COMM_WORLD) {
      SELF_PROFILE_END_FUNCTION;
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
   SELF_PROFILE_END_FUNCTION;
   return global_rank;
}

// Translate a rank from a remote group to the global rank
int vftr_remote2global_rank(MPI_Comm comm, int remote_rank) {
   SELF_PROFILE_START_FUNCTION;
   // no translation of ranks neccessary as it is already in the global group
   if (comm == MPI_COMM_WORLD) {
      SELF_PROFILE_END_FUNCTION;
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
   SELF_PROFILE_END_FUNCTION;
   return global_rank;
}
