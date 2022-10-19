#ifdef _MPI

SUBROUTINE MPI_Compare_and_swap_f08(origin_addr, compare_addr, result_addr, &
                                    datatype, target_rank, target_disp, &
                                    win, error)
   USE vftr_mpi_compare_and_swap_f082vftr_f08i, &
      ONLY : vftr_MPI_Compare_and_swap_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Compare_and_swap, &
             MPI_Datatype, &
             MPI_Win, &
             MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: origin_addr
   INTEGER, INTENT(IN) :: compare_addr
   INTEGER, INTENT(IN) :: result_addr
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER target_rank
   INTEGER(KIND=MPI_ADDRESS_KIND) target_disp
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Compare_and_swap(origin_addr, compare_addr, result_addr, &
                                 datatype, target_rank, target_disp, &
                                 win, tmperror)
   ELSE
      CALL vftr_MPI_Compare_and_swap_f082vftr(origin_addr, compare_addr, result_addr, &
                                           datatype%MPI_VAL, target_rank, target_disp, &
                                           win%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Compare_and_swap_f08

#endif
