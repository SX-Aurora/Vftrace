#ifdef _MPI

SUBROUTINE MPI_COMPARE_AND_SWAP(ORIGIN_ADDR, COMPARE_ADDR, RESULT_ADDR, &
                                DATATYPE, TARGET_RANK, TARGET_DISP, &
                                WIN, ERROR)
   USE vftr_mpi_compare_and_swap_f2vftr_fi, &
      ONLY : vftr_MPI_Compare_and_swap_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi, &
      ONLY : PMPI_COMPARE_AND_SWAP, &
             MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER ORIGIN_ADDR
   INTEGER COMPARE_ADDR
   INTEGER RESULT_ADDR
   INTEGER DATATYPE
   INTEGER TARGET_RANK
   INTEGER(KIND=MPI_ADDRESS_KIND) TARGET_DISP
   INTEGER WIN
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_COMPARE_AND_SWAP(ORIGIN_ADDR, COMPARE_ADDR, RESULT_ADDR, &
                                 DATATYPE, TARGET_RANK, TARGET_DISP, &
                                 WIN, ERROR)
   ELSE
      CALL vftr_MPI_Compare_and_swap_f2vftr(ORIGIN_ADDR, COMPARE_ADDR, RESULT_ADDR, &
                                         DATATYPE, TARGET_RANK, TARGET_DISP, &
                                         WIN, ERROR)
   END IF

END SUBROUTINE MPI_COMPARE_AND_SWAP

#endif
