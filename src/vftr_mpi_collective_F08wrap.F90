! This file is part of Vftrace.
!
! Vftrace is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! Vftrace is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#ifdef _MPI

SUBROUTINE MPI_Allgather_f08(sendbuf, sendcount, sendtype, &
                             recvbuf, recvcount, recvtype, &
                             comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Allgather_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Allgather_F(sendbuf, sendcount, sendtype%MPI_VAL, &
                             recvbuf, recvcount, recvtype%MPI_VAL, &
                             comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Allgather_f08

SUBROUTINE MPI_Allgatherv_f08(sendbuf, sendcount, sendtype, &
                              recvbuf, recvcounts, displs, &
                              recvtype, comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Allgatherv_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: displs(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL PMPI_Allgatherv(sendbuf, sendcount, sendtype, &
                        recvbuf, recvcounts, displs, &
                        recvtype, comm, tmperror)
!TODO: properly call this
!   CALL vftr_MPI_Allgatherv_F(SENDBUF, SENDCOUNT, SENDTYPE, &
!                              RECVBUF, RECVCOUNTS, DISPLS, &
!                              RECVTYPE, COMM, ERROR)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Allgatherv_f08

SUBROUTINE MPI_Allreduce_f08(sendbuf, recvbuf, count, &
                             datatype, op, comm, &
                             error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Allreduce_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm, &
                       MPI_Op
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Allreduce_F(sendbuf, recvbuf, count, &
                             datatype%MPI_VAL, op%MPI_VAL, comm%MPI_VAL, &
                             tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Allreduce_f08

SUBROUTINE MPI_Alltoall_f08(sendbuf, sendcount, sendtype, &
                            recvbuf, recvcount, recvtype, &
                            comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Alltoall_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Alltoall_F(sendbuf, sendcount, sendtype%MPI_VAL, &
                            recvbuf, recvcount, recvtype%MPI_VAL, &
                            comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Alltoall_f08

SUBROUTINE MPI_Alltoallv_f08(sendbuf, sendcounts, sdispls, sendtype, &
                             recvbuf, recvcounts, rdispls, recvtype, &
                             comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Alltoallv_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Alltoallv_F(sendbuf, sendcounts, sdispls, sendtype%MPI_VAL, &
                             recvbuf, recvcounts, rdispls, recvtype%MPI_VAL, &
                             comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Alltoallv_f08

SUBROUTINE MPI_Alltoallw_f08(sendbuf, sendcounts, sdispls, sendtypes, &
                             recvbuf, recvcounts, rdispls, recvtypes, &
                             comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Alltoallw_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: sdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtypes(*)
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: rdispls(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtypes(*)
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   INTEGER, DIMENSION(:), ALLOCATABLE :: tmpsendtypes
   INTEGER, DIMENSION(:), ALLOCATABLE :: tmprecvtypes
   INTEGER :: comm_size, i

   CALL PMPI_Comm_size(comm, comm_size, tmperror)
   ALLOCATE(tmprecvtypes(comm_size))
   ALLOCATE(tmprecvtypes(comm_size))
   DO i = 1, comm_size
      tmpsendtypes(i) = sendtypes(i)%MPI_VAL
   END DO
   DO i = 1, comm_size
      tmprecvtypes(i) = recvtypes(i)%MPI_VAL
   END DO

   CALL vftr_MPI_Alltoallw_F(sendbuf, sendcounts, sdispls, tmpsendtypes, &
                             recvbuf, recvcounts, rdispls, tmprecvtypes, &
                             comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

   DEALLOCATE(tmpsendtypes)
   DEALLOCATE(tmprecvtypes)

END SUBROUTINE MPI_Alltoallw_f08

SUBROUTINE MPI_Bcast_f08(buffer, count, datatype, &
                         root, comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Bcast_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: buffer
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Bcast_F(buffer, count, datatype%MPI_VAL, &
                         root, comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Bcast_f08

SUBROUTINE MPI_Gather_f08(sendbuf, sendcount, sendtype, &
                          recvbuf, recvcount, recvtype, &
                          root, comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Gather_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Gather_F(sendbuf, sendcount, sendtype%MPI_VAL, &
                          recvbuf, recvcount, recvtype%MPI_VAL, &
                          root, comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Gather_f08

SUBROUTINE MPI_Gatherv_f08(sendbuf, sendcount, sendtype, &
                           recvbuf, recvcounts, displs, &
                           recvtype, root, comm, &
                           error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Gatherv_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   INTEGER, INTENT(IN) :: displs(*)
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Gatherv_F(sendbuf, sendcount, sendtype%MPI_VAL, &
                           recvbuf, recvcounts, displs, &
                           recvtype%MPI_VAL, root, comm%MPI_VAL, &
                           tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Gatherv_f08

SUBROUTINE MPI_Reduce_f08(sendbuf, recvbuf, count, datatype, &
                      op, root, comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Reduce_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm, &
                       MPI_Op
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: count
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Reduce_F(sendbuf, recvbuf, count, datatype%MPI_VAL, &
                          op%MPI_VAL, root, comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Reduce_f08

SUBROUTINE MPI_Reduce_scatter_f08(sendbuf, recvbuf, recvcounts, &
                                  datatype, op, comm, &
                                  error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Reduce_scatter_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm, &
                       MPI_Op
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcounts(*)
   TYPE(MPI_Datatype), INTENT(IN) :: datatype
   TYPE(MPI_Op), INTENT(IN) :: op
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Reduce_scatter_F(sendbuf, recvbuf, recvcounts, &
                                  datatype%MPI_VAL, op%MPI_VAL, comm%MPI_VAL, &
                                  tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Reduce_scatter_f08

SUBROUTINE MPI_Scatter_f08(sendbuf, sendcount, sendtype, &
                           recvbuf, recvcount, recvtype, &
                           root, comm, error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Scatter_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcount
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Scatter_F(sendbuf, sendcount, sendtype%MPI_VAL, &
                           recvbuf, recvcount, recvtype%MPI_VAL, &
                           root, comm%MPI_VAL, tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Scatter_f08

SUBROUTINE MPI_Scatterv_f08(sendbuf, sendcounts, displs, &
                            sendtype, recvbuf, recvcount, &
                            recvtype, root, comm, &
                            error)
   USE vftr_mpi_collective_c2F, &
      ONLY : vftr_MPI_Scatterv_F
   USE mpi_f08, ONLY : MPI_Datatype, &
                       MPI_Comm
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: sendbuf
   INTEGER, INTENT(IN) :: sendcounts(*)
   INTEGER, INTENT(IN) :: displs(*)
   TYPE(MPI_Datatype), INTENT(IN) :: sendtype
   INTEGER :: recvbuf
   INTEGER, INTENT(IN) :: recvcount
   TYPE(MPI_Datatype), INTENT(IN) :: recvtype
   INTEGER, INTENT(IN) :: root
   TYPE(MPI_Comm), INTENT(IN) :: comm
   INTEGER, OPTIONAL, INTENT(OUT) :: error
   INTEGER :: tmperror

   CALL vftr_MPI_Scatterv_F(sendbuf, sendcounts, displs, &
                            sendtype%MPI_VAL, recvbuf, recvcount, &
                            recvtype%MPI_VAL, root, comm%MPI_VAL, &
                            tmperror)
   IF (PRESENT(error)) error = tmperror

END SUBROUTINE MPI_Scatterv_f08

#endif
