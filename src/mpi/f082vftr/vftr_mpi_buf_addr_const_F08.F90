#ifdef _MPI

MODULE mpi_buf_addr_const_F08

   IMPLICIT NONE

   PRIVATE
   PUBLIC :: vftr_is_F08_MPI_BOTTOM, &
             vftr_is_F08_MPI_IN_PLACE

CONTAINS

   FUNCTION vftr_is_F08_MPI_BOTTOM(addr) &
      RESULT(is) &
      BIND(C, name="vftr_is_F08_MPI_BOTTOM")
      USE, INTRINSIC :: ISO_C_BINDING
      USE mpi_f08, ONLY : MPI_BOTTOM, &
                          MPI_ADDRESS_KIND
      INTEGER, INTENT(IN) :: addr
      INTEGER(KIND=c_int) :: is
      INTEGER(KIND=MPI_ADDRESS_KIND) :: addr_MPI_BOTTOM
      INTEGER(KIND=MPI_ADDRESS_KIND) :: addr_addr
      INTEGER :: ierr

      CALL PMPI_Get_address(MPI_BOTTOM, addr_MPI_BOTTOM, ierr)
      CALL PMPI_Get_address(addr, addr_addr, ierr)

      IF (addr_addr == addr_MPI_BOTTOM) THEN
         is = 1
      ELSE
         is = 0
      END IF
   END FUNCTION vftr_is_F08_MPI_BOTTOM

   FUNCTION vftr_is_F08_MPI_IN_PLACE(addr) &
      RESULT(is) &
      BIND(C, name="vftr_is_F08_MPI_IN_PLACE")
      USE, INTRINSIC :: ISO_C_BINDING
      USE mpi_f08, ONLY : MPI_IN_PLACE, &
                          MPI_ADDRESS_KIND
      INTEGER, INTENT(IN) :: addr
      INTEGER(KIND=c_int) :: is
      INTEGER(KIND=MPI_ADDRESS_KIND) :: addr_MPI_IN_PLACE
      INTEGER(KIND=MPI_ADDRESS_KIND) :: addr_addr
      INTEGER :: ierr

      CALL PMPI_Get_address(MPI_IN_PLACE, addr_MPI_IN_PLACE, ierr)
      CALL PMPI_Get_address(addr, addr_addr, ierr)

      IF (addr_addr == addr_MPI_IN_PLACE) THEN
         is = 1
      ELSE
         is = 0
      END IF
   END FUNCTION vftr_is_F08_MPI_IN_PLACE

END MODULE mpi_buf_addr_const_F08

#endif
