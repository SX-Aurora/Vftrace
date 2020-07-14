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

#ifdef _MPI
#include <mpi.h>

#include "vftr_mpi_testwait.h"

int MPI_Barrier(MPI_Comm comm) {
   return vftr_MPI_Barrier(comm);
}

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
   return vftr_MPI_Test(request, flag, status);
}

int MPI_Testany(int count, MPI_Request array_of_requests[],
                int *index, int *flag, MPI_Status *status) {
   return vftr_MPI_Testany(count, array_of_requests, index, flag, status);
}

int MPI_Testsome(int incount, MPI_Request array_of_requests[],
                 int *outcount, int array_of_indices[],
                 MPI_Status array_of_statuses[]) {
   return vftr_MPI_Testsome(incount, array_of_requests, outcount,
                            array_of_indices, array_of_statuses);
}

int MPI_Testall(int count, MPI_Request array_of_requests[],
                int *flag, MPI_Status array_of_statuses[]) {
   return vftr_MPI_Testall(count, array_of_requests, flag, array_of_statuses);
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
   return vftr_MPI_Wait(request, status);
}

int MPI_Waitany(int count, MPI_Request array_of_requests[],
                int *index, MPI_Status *status) {
   return vftr_MPI_Waitany(count, array_of_requests, index, status);
}

int MPI_Waitsome(int incount, MPI_Request array_of_requests[],
                 int *outcount, int array_of_indices[],
                 MPI_Status array_of_statuses[]) {
   return vftr_MPI_Waitsome(incount, array_of_requests, outcount,
                            array_of_indices, array_of_statuses);
}

int MPI_Waitall(int count, MPI_Request array_of_requests[],
                MPI_Status array_of_statuses[]) {
   return vftr_MPI_Waitall(count, array_of_requests, array_of_statuses);
}

#endif
