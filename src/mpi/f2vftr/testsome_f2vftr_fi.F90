MODULE vftr_mpi_testsome_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Testsome_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Testsome_f2vftr(F_INCOUNT, F_ARRAY_OF_REQUESTS, &
                                       F_OUTCOUNT, F_ARRAY_OF_INDICES, &
                                       F_ARRAY_OF_STATUSES, F_ERROR) &
         BIND(C, name="vftr_MPI_Testsome_f2vftr")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_INCOUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_OUTCOUNT
         INTEGER F_ARRAY_OF_INDICES(*)
         INTEGER F_ARRAY_OF_STATUSES(MPI_STATUS_SIZE,*)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Testsome_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_testsome_f2vftr_fi
