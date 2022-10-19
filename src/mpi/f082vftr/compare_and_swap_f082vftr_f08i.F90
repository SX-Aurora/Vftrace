MODULE vftr_mpi_compare_and_swap_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Compare_and_swap_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Compare_and_swap_f082vftr(origin_addr, compare_addr, result_addr, &
                                                    f_datatype, target_rank, target_disp, &
                                                    f_win, f_error) &
         BIND(C, name="vftr_MPI_Compare_and_swap_f082vftr")
         USE mpi_f08, ONLY : MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: origin_addr
         INTEGER, INTENT(IN) :: compare_addr
         INTEGER, INTENT(IN) :: result_addr
         INTEGER, INTENT(IN) :: f_datatype
         INTEGER, INTENT(IN) :: target_rank
         INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
         INTEGER, INTENT(IN) :: f_win
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Compare_and_swap_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_compare_and_swap_f082vftr_f08i
