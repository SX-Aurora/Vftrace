MODULE vftr_mpi_compare_and_swap_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Compare_and_swap_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Compare_and_swap_f2vftr(ORIGIN_ADDR, COMPARE_ADDR, RESULT_ADDR, &
                                               F_DATATYPE, TARGET_RANK, TARGET_DISP, &
                                               F_WIN, F_ERROR) &
         BIND(C, name="vftr_MPI_Compare_and_swap_f2vftr")
         USE mpi, ONLY : MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER ORIGIN_ADDR
         INTEGER COMPARE_ADDR
         INTEGER RESULT_ADDR
         INTEGER F_DATATYPE
         INTEGER TARGET_RANK
         INTEGER(KIND=MPI_ADDRESS_KIND) TARGET_DISP
         INTEGER F_WIN
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Compare_and_swap_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_compare_and_swap_f2vftr_fi
