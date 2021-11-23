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

MODULE vftr_mpi_sendrecv_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Sendrecv_f082vftr

   INTERFACE 

      SUBROUTINE vftr_MPI_Sendrecv_f082vftr(sendbuf, sendcount, f_sendtype, dest, sendtag, &
                                            recvbuf, recvcount, f_recvtype, source, recvtag, &
                                            f_comm, f_status, f_error) &
         BIND(C, name="vftr_MPI_Sendrecv_f082vftr")
         USE mpi_f08, &
            ONLY : MPI_Status
         IMPLICIT NONE
         INTEGER :: sendbuf
         INTEGER :: sendcount
         INTEGER :: f_sendtype
         INTEGER :: dest
         INTEGER :: sendtag
         INTEGER :: recvbuf
         INTEGER :: recvcount
         INTEGER :: f_recvtype
         INTEGER :: source
         INTEGER :: recvtag
         INTEGER :: f_comm
         TYPE(MPI_Status) :: f_status
         INTEGER :: f_error
      END SUBROUTINE vftr_MPI_Sendrecv_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_sendrecv_f082vftr_f08i
