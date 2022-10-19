#ifdef _MPI

SUBROUTINE MPI_Reduce_scatter_block_f08(sendbuf, recvbuf, recvcount, &
                                        datatype, op, comm, &
                                        error)
   USE vftr_mpi_reduce_scatter_block_f082vftr_f08i, &
      ONLY : vftr_MPI_Reduce_scatter_block_f082vftr
   USE vftr_mpi_logging_f08, &
      ONLY : vftr_no_mpi_logging_f08
   USE vftr_sync_time_F08, &
      ONLY : vftr_estimate_sync_time
   USE mpi_f08, &
      ONLY : PMPI_Reduce_scatter_block, &
             MPI_Datatype, &
             MPI_Comm, &
             MPI_Op
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_estimate_sync_time("MPI_Reduce_scatter_block_sync", comm)

   IF (vftr_no_mpi_logging_f08()) THEN
      CALL PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, &
                                     datatype, op, comm, &
                                     tmperror)
   ELSE
      CALL vftr_MPI_Reduce_scatter_block_f082vftr(sendbuf, recvbuf, recvcount, &
                                               datatype%MPI_VAL, op%MPI_VAL, comm%MPI_VAL, &
                                               tmperror)
   END IF
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Reduce_scatter_block_f08

#endif
