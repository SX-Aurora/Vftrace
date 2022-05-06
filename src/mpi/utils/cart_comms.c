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

#include <stdlib.h>

#include <mpi.h>

void vftr_mpi_cart_neighbor_ranks(MPI_Comm cart_comm,
                                  int *nneighbors_ptr,
                                  int **neighbors_ptr) {
   int ndims;
   PMPI_Cartdim_get(cart_comm, &ndims);
   int *dims = (int*) malloc(ndims*sizeof(int));
   int *periodic = (int*) malloc(ndims*sizeof(int));
   int *coords = (int*) malloc(ndims*sizeof(int));
   PMPI_Cart_get(cart_comm, ndims, dims, periodic, coords);
   int nneighbors = 2*ndims;
   int *neighbors = (int*) malloc(nneighbors*sizeof(int));
   int ineighbor = 0;
   for (int idim=0; idim<ndims; idim++) {
      neighbors[ineighbor] = -1;
      // check if the own coordinate is on the lower grid end.
      // if non-periodic then don't record the neighbors rank.
      if (periodic[idim] || coords[idim] > 0) {
         coords[idim]--;
         PMPI_Cart_rank(cart_comm, coords, neighbors+ineighbor);
         coords[idim]++;
      }
      ineighbor++;

      neighbors[ineighbor] = -1;
      // check if the own coordinate is on the upper grid end.
      // if non-periodic then con't record the neighbors rank.
      if (periodic[idim] || coords[idim] < dims[idim]-1) {
         coords[idim]++;
         PMPI_Cart_rank(cart_comm, coords, neighbors+ineighbor);
         coords[idim]--;
      }
      ineighbor++;
   }
   // deallocate temporary arrays
   free(dims);
   free(periodic);
   free(coords);
   // assign result pointers
   *nneighbors_ptr = nneighbors;
   *neighbors_ptr = neighbors;
}
