MODULE vftr_sync_time_F08

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_estimate_sync_time

   INTERFACE

      SUBROUTINE vftr_estimate_sync_time_F082C(routine_name, comm) &
         BIND(c, NAME="vftr_estimate_sync_time_F082C")
         USE ISO_C_BINDING, ONLY : C_CHAR
         IMPLICIT NONE
         CHARACTER(KIND=C_CHAR), INTENT(IN) :: routine_name(*)
         INTEGER, INTENT(IN) :: comm
      END SUBROUTINE vftr_estimate_sync_time_F082C

   END INTERFACE

CONTAINS

   SUBROUTINE vftr_estimate_sync_time(routine_name_F, comm)
      USE ISO_C_BINDING, ONLY : C_CHAR, C_NULL_CHAR
      USE mpi_f08, ONLY : MPI_Comm
      IMPLICIT NONE
      CHARACTER(LEN=*), INTENT(IN) :: routine_name_F
      TYPE(MPI_Comm), INTENT(IN) :: comm
      INTEGER :: name_len
      CHARACTER(KIND=C_CHAR, LEN=:), ALLOCATABLE :: routine_name_C
      name_len = LEN(ADJUSTL(TRIM(routine_name_F)))
      ! null terminator space
      name_len = name_len + 1
      ALLOCATE(CHARACTER(LEN=name_len) :: routine_name_C)
      routine_name_C(:) = ADJUSTL(TRIM(routine_name_F))
      routine_name_C(name_len:name_len) = C_NULL_CHAR
      CALL vftr_estimate_sync_time_F082C(routine_name_C, comm%MPI_VAL)
      DEALLOCATE(routine_name_C)
   END SUBROUTINE vftr_estimate_sync_time

END MODULE vftr_sync_time_F08
