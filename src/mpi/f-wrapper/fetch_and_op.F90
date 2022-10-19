#ifdef _MPI

SUBROUTINE MPI_FETCH_AND_OP(ORIGIN_ADDR, RESULT_ADDR, DATATYPE, &
                            TARGET_RANK, TARGET_DISP, OP, WIN, &
                            ERROR)
   USE vftr_mpi_fetch_and_op_f2vftr_fi, &
      ONLY : vftr_MPI_Fetch_and_op_f2vftr
   USE vftr_mpi_logging_F, &
      ONLY : vftr_no_mpi_logging_F
   USE mpi,&
      ONLY : PMPI_FETCH_AND_OP, &
             MPI_ADDRESS_KIND
   IMPLICIT NONE
   INTEGER ORIGIN_ADDR
   INTEGER RESULT_ADDR
   INTEGER DATATYPE
   INTEGER TARGET_RANK
   INTEGER(KIND=MPI_ADDRESS_KIND) TARGET_DISP
   INTEGER OP
   INTEGER WIN
   INTEGER ERROR

   IF (vftr_no_mpi_logging_F()) THEN
      CALL PMPI_FETCH_AND_OP(ORIGIN_ADDR, RESULT_ADDR, DATATYPE, &
                             TARGET_RANK, TARGET_DISP, OP, WIN, &
                             ERROR)
   ELSE
      CALL vftr_MPI_Fetch_and_op_f2vftr(ORIGIN_ADDR, RESULT_ADDR, DATATYPE, &
                                     TARGET_RANK, TARGET_DISP, OP, WIN, &
                                     ERROR)
   END IF

END SUBROUTINE MPI_FETCH_AND_OP

#endif
