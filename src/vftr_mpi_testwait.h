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

#ifndef VFTR_MPI_TESTWAIT_H
#define VFTR_MPI_TESTWAIT_H

#ifdef _MPI
#include <mpi.h>

int vftr_MPI_Barrier(MPI_Comm comm);

int vftr_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);

int vftr_MPI_Testany(int count, MPI_Request array_of_requests[],
                     int *index, int *flag, MPI_Status *status);

int vftr_MPI_Testsome(int incount, MPI_Request array_of_requests[],
                      int *outcount, int array_of_indices[],
                      MPI_Status array_of_statuses[]);

int vftr_MPI_Testall(int count, MPI_Request array_of_requests[],
                     int *flag, MPI_Status array_of_statuses[]);

int vftr_MPI_Wait(MPI_Request *request, MPI_Status *status);

int vftr_MPI_Waitany(int count, MPI_Request array_of_requests[],
                     int *index, MPI_Status *status);

int vftr_MPI_Waitsome(int incount, MPI_Request array_of_requests[],
                      int *outcount, int array_of_indices[],
                      MPI_Status array_of_statuses[]);

int vftr_MPI_Waitall(int count, MPI_Request array_of_requests[],
                     MPI_Status array_of_statuses[]);

#endif
#endif
