#ifdef _MPI

SUBROUTINE MPI_Raccumulate_f08(origin_addr, origin_count, origin_datatype, &
                               target_rank, target_disp, target_count, &
                               target_datatype, op, win, request, error)
   USE vftr_mpi_raccumulate_f082vftr_f08i, &
      ONLY : vftr_MPI_Raccumulate_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE mpi_f08, &
      ONLY : PMPI_Raccumulate, &
             MPI_Datatype, &
             MPI_Op, &
             MPI_Win, &
             MPI_ADDRESS_KIND, &
             MPI_Request
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: origin_addr
   INTEGER, INTENT(IN) :: origin_count
   TYPE(MPI_Datatype), INTENT(IN) :: origin_datatype
   INTEGER, INTENT(IN) :: target_rank
   INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
   INTEGER, INTENT(IN) :: target_count
   TYPE(MPI_Datatype), INTENT(IN) :: target_datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Win), INTENT(IN) :: win
   TYPE(MPI_Request), INTENT(OUT) :: request
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Raccumulate(origin_addr, origin_count, origin_datatype, &
                            target_rank, target_disp, target_count, &
                            target_datatype, op, win, request, &
                            tmperror)
   ELSE
      CALL vftr_MPI_Raccumulate_f082vftr(origin_addr, origin_count, origin_datatype%MPI_VAL, &
                                     target_rank, target_disp, target_count, &
                                     target_datatype%MPI_VAL, op%MPI_VAL, win%MPI_VAL, &
                                     request%MPI_VAL, tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Raccumulate_f08

#endif
