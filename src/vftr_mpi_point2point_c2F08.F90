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

MODULE vftr_mpi_point2point_c2F08
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Send_F08, &
             vftr_MPI_Recv_F08

   INTERFACE 
      SUBROUTINE vftr_MPI_Send_F08(buf, count, fdatatype, dest, tag, fcomm, ferror) &
         BIND(C, name="vftr_MPI_Send_F08")
         USE mpi_f08, ONLY : MPI_Datatype, &
                             MPI_Comm
         IMPLICIT NONE

         TYPE(*), DIMENSION(..), INTENT(IN) :: buf
         INTEGER, INTENT(IN) :: count
         INTEGER, INTENT(IN) :: fdatatype
         INTEGER, INTENT(IN) :: dest
         INTEGER, INTENT(IN) :: tag
         INTEGER, INTENT(IN) :: fcomm
         INTEGER, INTENT(OUT) :: ferror
      END SUBROUTINE vftr_MPI_Send_F08

      SUBROUTINE vftr_MPI_Recv_F08(buf, count, fdatatype, source, tag, fcomm, fstatus, ferror) &
         BIND(C, name="vftr_MPI_Recv_F08")
         USE mpi_f08, ONLY : MPI_Datatype, &
                             MPI_Comm
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE

         TYPE(*), DIMENSION(..) :: buf
         INTEGER, INTENT(IN) :: count
         INTEGER, INTENT(IN) :: fdatatype
         INTEGER, INTENT(IN) :: source
         INTEGER, INTENT(IN) :: tag
         INTEGER, INTENT(IN) :: fcomm
         !TYPE(MPI_Status) :: fstatus
         INTEGER :: fstatus(MPI_STATUS_SIZE)
         INTEGER, INTENT(OUT) :: ferror
      END SUBROUTINE vftr_MPI_Recv_F08

   END INTERFACE
END MODULE vftr_mpi_point2point_c2F08

#endif
