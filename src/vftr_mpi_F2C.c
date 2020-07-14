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

// defines an intermediate layer between the fortran interface and the 
// C framework in order to properly translate mpi structures to C

#include "vftr_mpi_utils.h"
#include "vftr_sync_messages.h"
#include "vftr_async_messages.h"
#include "vftr_cycles.h"
#include "vftr_timer.h"

#ifdef _MPI
///////////////////////////////
// From vftr_sync_messages.c //
///////////////////////////////

// store message info for synchronous mpi-communication
void vftr_store_sync_message_info_F(vftr_direction dir, int count, MPI_Fint ftype,
                                    int peer_rank, int tag, MPI_Fint fcomm,
                                    long long tstart, long long tend) {
   // create c mpi structures 
   // translate the fortran structures to them
   MPI_Datatype ctype = MPI_Type_f2c(ftype);
   MPI_Comm ccomm = MPI_Comm_f2c(fcomm);

   // call the c function to store the synchronous communication information
   vftr_store_sync_message_info(dir, count, ctype,
                                peer_rank, tag, ccomm,
                                tstart, tend);
   return;
}

////////////////////////////////
// From vftr_async_messages.c //
////////////////////////////////

// store message info for asynchronous mpi-communication
void vftr_register_request_F(vftr_direction dir, int count, MPI_Fint ftype,  
                             int peer_rank, int tag, MPI_Fint fcomm,
                             MPI_Fint frequest, long long tstart) {

   // create c mpi structures 
   // translate the fortran structures to them
   MPI_Datatype ctype = MPI_Type_f2c(ftype);
   MPI_Comm ccomm = MPI_Comm_f2c(fcomm);
   MPI_Request crequest = MPI_Request_f2c(frequest);

   // call the c function to register a request for asynchronous mpi-communication
   vftr_register_request(dir, count, ctype,
                         peer_rank, tag, ccomm,
                         crequest, tstart);

   return;
}

// test the entire list of open request for completed communication
void vftr_clear_completed_request_F() {
   // As this function has no arguments it does not require
   // translation of fortran mpi-structures
   // However, it is included here, to have a 1:1 mapping
   // to the fortran interface
   vftr_clear_completed_request();
   
   return;
}

///////////////////////////
// From vftr_mpi_utils.c //
///////////////////////////

// Translate a rank from a local group to the global rank
int vftr_local2global_rank_F(MPI_Comm fcomm, int local_rank) {
   
   MPI_Comm ccomm = MPI_Comm_f2c(fcomm);

   return vftr_local2global_rank(fcomm, local_rank);
}

// Translate a rank from a remote group to the global rank
int vftr_remote2global_rank_F(MPI_Comm fcomm, int remote_rank) {

   MPI_Comm ccomm = MPI_Comm_f2c(fcomm);

   return vftr_remote2global_rank(fcomm, remote_rank);
}

// mark a MPI_Status as empty 
void vftr_empty_mpi_status_F(MPI_Fint *fstatus) {

   // create a c mpi-status structure
   MPI_Status *cstatus;
   // translate fortran mpi-status to C
   MPI_Status_f2c(fstatus, cstatus);
   // call the c function to empty the status
   vftr_empty_mpi_status(cstatus);
   // translate the status back to a fortran mpi-status
   MPI_Status_c2f(cstatus, fstatus);

   return;
}

//// mark a MPI_Status as empty
//void vftr_emtpy_mpi_status_F(const MPI_F08_status *fstatus) {
//
//    // create a c mpi-status structure
//    MPI_Status *cstatus;
//    // translate fortran mpi-status to C
//    MPI_Status_f082c(fstatus, cstatus);
//    // translate the status back to a fortran mpi-status
//    MPI_Statusc2f08(cstatus, fstatus);
//
//    return;
//}

// check if a status is empty 
bool vftr_mpi_status_is_empty_F(MPI_Fint *fstatus) {

   // create a c mpi-status structure
   MPI_Status *cstatus;
   // translate fortran mpi-status to C
   MPI_Status_f2c(fstatus, cstatus);
   // call the c function to check if status is empty

   return vftr_mpi_status_is_empty(cstatus);
}

//// check if a status is empty
//bool vftr_mpi_status_is_empty_F(const MPI_F08_status *fstatus) {
//
//   // create a c mpi-status structure
//   MPI_Status *cstatus;
//   // translate fortran mpi-status to c
//   MPI_Status_f082c(fstatus, cstatus);
//   // call the c function to check if status is empty
//
//   return vftr_mpi_status_is_empty(cstatus);
//}

// check if a request is active 
bool vftr_mpi_request_is_active_F(MPI_Fint frequest) {

   // create a c mpi request structure and
   // translate fortran to c request
   MPI_Request crequest = MPI_Request_f2c(frequest);
   // call the c function to check if the request is active

   return vftr_mpi_request_is_active(crequest);
}

///////////////////////
// From vftr_timer.c //
///////////////////////

long long vftr_get_runtime_usec_F() {
   return vftr_get_runtime_usec();
}

#endif
