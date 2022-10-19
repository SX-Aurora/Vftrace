MODULE vftr_mpi_raccumulate_f082vftr_f08i
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Raccumulate_f082vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Raccumulate_f082vftr(origin_addr, origin_count, f_origin_datatype, &
                                               target_rank, target_disp, target_count, &
                                               f_target_datatype, f_op, f_win, f_request, &
                                               f_error) &
         BIND(C, name="vftr_MPI_Raccumulate_f082vftr")
         USE mpi_f08, &
            ONLY: MPI_ADDRESS_KIND
         IMPLICIT NONE
         INTEGER, INTENT(IN) :: origin_addr
         INTEGER, INTENT(IN) :: origin_count
         INTEGER, INTENT(IN) :: f_origin_datatype
         INTEGER, INTENT(IN) :: target_rank
         INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
         INTEGER, INTENT(IN) :: target_count
         INTEGER, INTENT(IN) :: f_target_datatype
         INTEGER, INTENT(IN) :: f_op
         INTEGER, INTENT(IN) :: f_win
         INTEGER, INTENT(OUT) :: f_request
         INTEGER, INTENT(OUT) :: f_error
      END SUBROUTINE vftr_MPI_Raccumulate_f082vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_raccumulate_f082vftr_f08i
