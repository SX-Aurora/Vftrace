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

MODULE vftr_mpi_point2point_c2F
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Send_F, &
             vftr_MPI_Bsend_F, &
             vftr_MPI_Bsend_init_F, &
             vftr_MPI_Isend_F, &
             vftr_MPI_Ibsend_F, &
             vftr_MPI_Ssend_F, &
             vftr_MPI_Issend_F, &
             vftr_MPI_Rsend_F, &
             vftr_MPI_Irsend_F, &
             vftr_MPI_Recv_F, &
             vftr_MPI_Irecv_F, &
             vftr_MPI_Sendrecv_F, &
             vftr_MPI_Sendrecv_replace_F

   INTERFACE 
      SUBROUTINE vftr_MPI_Send_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Send_F")
         IMPLICIT NONE
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Send_F

      SUBROUTINE vftr_MPI_Bsend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Bsend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Bsend_F

      SUBROUTINE vftr_MPI_Bsend_init_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Bsend_init_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Bsend_init_F

      SUBROUTINE vftr_MPI_Isend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Isend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Isend_F

      SUBROUTINE vftr_MPI_Ibsend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Ibsend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Ibsend_F

      SUBROUTINE vftr_MPI_Ssend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Ssend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Ssend_F

      SUBROUTINE vftr_MPI_Issend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Issend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Issend_F

      SUBROUTINE vftr_MPI_Rsend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Rsend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Rsend_F

      SUBROUTINE vftr_MPI_Irsend_F(BUF, COUNT, F_DATATYPE, DEST, TAG, F_COMM, F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Irsend_F")
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Irsend_F

      SUBROUTINE vftr_MPI_Recv_F(BUF, COUNT, F_DATATYPE, SOURCE, TAG, F_COMM, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Recv_F")
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER SOURCE
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Recv_F

      SUBROUTINE vftr_MPI_Irecv_F(BUF, COUNT, F_DATATYPE, SOURCE, TAG, F_COMM, F_STATUS, F_REQUEST, F_ERROR) &
         BIND(C, name="vftr_MPI_Irecv_F")
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER SOURCE
         INTEGER TAG
         INTEGER F_COMM
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_REQUEST
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Irecv_F

      SUBROUTINE vftr_MPI_Sendrecv_F(SENDBUF, SENDCOUNT, F_SENDTYPE, DEST, SENDTAG, &
                                     RECVBUF, RECVCOUNT, F_RECVTYPE, SOURCE, RECVTAG, &
                                     F_COMM, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Sendrecv_F")
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER SENDBUF
         INTEGER SENDCOUNT
         INTEGER F_SENDTYPE
         INTEGER DEST
         INTEGER SENDTAG
         INTEGER RECVBUF
         INTEGER RECVCOUNT
         INTEGER F_RECVTYPE
         INTEGER SOURCE
         INTEGER RECVTAG
         INTEGER F_COMM
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Sendrecv_F

      SUBROUTINE vftr_MPI_Sendrecv_replace_F(BUF, COUNT, F_DATATYPE, DEST, SENDTAG, SOURCE, &
                                             RECVTAG, F_COMM, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Sendrecv_replace_F")
         USE mpi, ONLY : MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER BUF
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER DEST
         INTEGER SENDTAG
         INTEGER SOURCE
         INTEGER RECVTAG
         INTEGER F_COMM
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Sendrecv_replace_F

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_point2point_c2F

