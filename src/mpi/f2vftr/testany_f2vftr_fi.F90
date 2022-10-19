MODULE vftr_mpi_testany_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Testany_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Testany_f2vftr(F_COUNT, F_ARRAY_OF_REQUESTS, F_INDEX, &
                                      F_FLAG, F_STATUS, F_ERROR) &
         BIND(C, name="vftr_MPI_Testany_f2vftr")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         INTEGER F_COUNT
         INTEGER F_ARRAY_OF_REQUESTS(*)
         INTEGER F_INDEX
         INTEGER F_FLAG
         INTEGER F_STATUS(MPI_STATUS_SIZE)
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Testany_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_testany_f2vftr_fi
