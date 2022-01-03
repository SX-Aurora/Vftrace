! This file is part of Vftrace.
!
! Vftrace is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! Vftrace is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

MODULE vftr_mpi_ineighbor_alltoallw_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Ineighbor_alltoallw_f2vftr

   INTERFACE 

      SUBROUTINE vftr_MPI_Ineighbor_alltoallw_f2vftr(SENDBUF, F_SENDCOUNTS, &
                                                     F_SDISPLS, F_SENDTYPES, &
                                                     RECVBUF, F_RECVCOUNTS, &
                                                     F_RDISPLS, F_RECVTYPES, &
                                                     F_COMM, F_REQUEST, &
                                                     F_ERROR) &
         BIND(C, name="vftr_MPI_Ineighbor_alltoallw_f2vftr")
         USE mpi, ONLY : MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER F_SENDCOUNTS(*)
         INTEGER(KIND=MPI_ADDRESS_KIND) F_SDISPLS(*)
         INTEGER F_SENDTYPES(*)
         INTEGER RECVBUF
         INTEGER F_RECVCOUNTS(*)
         INTEGER(KIND=MPI_ADDRESS_KIND) F_RDISPLS(*)
         INTEGER F_RECVTYPES(*)
         INTEGER F_COMM
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Ineighbor_alltoallw_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_ineighbor_alltoallw_f2vftr_fi
