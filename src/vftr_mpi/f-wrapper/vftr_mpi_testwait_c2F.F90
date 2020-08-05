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

MODULE vftr_mpi_testwait_c2F
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Barrier_F, &
             vftr_MPI_Test_F, &
             vftr_MPI_Testany_F, &
             vftr_MPI_Testsome_F, &
             vftr_MPI_Testall_F, &
             vftr_MPI_Wait_F, &
             vftr_MPI_Waitany_F, &
             vftr_MPI_Waitsome_F, &
             vftr_MPI_Waitall_F

   INTERFACE

      SUBROUTINE vftr_MPI_Barrier_F(F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Barrier_F")
         IMPLICIT NONE
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Barrier_F

      SUBROUTINE vftr_MPI_Test_F(F_REQUEST, F_FLAG, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Test_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_REQUEST
         LOGICAL F_FLAG
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Test_F

      SUBROUTINE vftr_MPI_Testany_F(F_COUNT, F_ARRAY_OF_REQUESTS, F_INDEX, &
                                    F_FLAG, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Testany_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_COUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_INDEX
         LOGICAL F_FLAG
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Testany_F

      SUBROUTINE vftr_MPI_Testsome_F(F_INCOUNT, F_ARRAY_OF_REQUESTS, &
                                     F_OUTCOUNT, F_ARRAY_OF_INDICES, &
                                     F_ARRAY_OF_STATUSES, F_ERROR) &
         BIND(C, name="vftr_MPI_Testsome_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_INCOUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_OUTCOUNT
         INTEGER F_ARRAY_OF_INDICES(*)
         INTEGER F_ARRAY_OF_STATUSES(MPI_STATUS_SIZE,*)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Testsome_F

      SUBROUTINE vftr_MPI_Testall_F(F_COUNT, F_ARRAY_OF_REQUESTS, &
                                    F_FLAG, F_ARRAY_OF_STATUSES, &
                                    F_ERROR) &
         BIND(C, name="vftr_MPI_Testall_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_COUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         LOGICAL F_FLAG
         INTEGER F_ARRAY_OF_STATUSES(MPI_STATUS_SIZE,*)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Testall_F

      SUBROUTINE vftr_MPI_Wait_F(F_REQUEST, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Wait_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_REQUEST
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Wait_F

      SUBROUTINE vftr_MPI_Waitany_F(F_COUNT, F_ARRAY_OF_REQUESTS, &
                                    F_INDEX, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Waitany_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_COUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_INDEX
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Waitany_F

      SUBROUTINE vftr_MPI_Waitsome_F(F_INCOUNT, F_ARRAY_OF_REQUESTS, &
                                     F_OUTCOUNT, F_ARRAY_OF_INDICES, &
                                     F_ARRAY_OF_STATUSES, F_ERROR) &
         BIND(C, name="vftr_MPI_Waitsome_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_INCOUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_OUTCOUNT
         INTEGER F_ARRAY_OF_INDICES(*)
         INTEGER F_ARRAY_OF_STATUSES(MPI_STATUS_SIZE,*)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Waitsome_F

      SUBROUTINE vftr_MPI_Waitall_F(F_COUNT, F_ARRAY_OF_REQUESTS, &
                                    F_ARRAY_OF_STATUSES, F_ERROR) &
         BIND(C, name="vftr_MPI_Waitall_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_COUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_ARRAY_OF_STATUSES(MPI_STATUS_SIZE,*)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Waitall_F

   END INTERFACE

#endif 

CONTAINS

END MODULE vftr_mpi_testwait_c2F
