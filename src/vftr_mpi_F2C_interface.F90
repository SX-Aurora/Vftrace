MODULE vftr_mpi_F2C_interface
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: send, recv, &
             vftr_initialize, &
             vftr_after_mpi_init, &
             vftr_finalize, &
             vftr_store_sync_message_info, &
             vftr_register_request, &
             vftr_clear_completed_request, &
             vftr_local2global_rank, &
             vftr_remote2global_rank, &
             vftr_empty_mpi_status, &
             vftr_mpi_status_is_empty, &
             vftr_mpi_request_is_active, &
             vftr_get_runtime_usec
        
#include "mpi_directions.h"
   INTEGER, PARAMETER :: send = SENDING_DIR
   INTEGER, PARAMETER :: recv = RECIEVING_DIR

   INTERFACE
      ! Initialize vftrace
      SUBROUTINE vftr_initialize() &
         BIND(c, NAME="vftr_initialize")
         IMPLICIT NONE
      END SUBROUTINE vftr_initialize

      ! Things to do after mpi initialization
      SUBROUTINE vftr_after_mpi_init() &
         BIND(c, NAME="vftr_after_mpi_init")
         IMPLICIT NONE
      END SUBROUTINE vftr_after_mpi_init

      ! Finalize vftrace
      SUBROUTINE vftr_finalize() &
         BIND(c, NAME="vftr_finalize")
         IMPLICIT NONE
      END SUBROUTINE vftr_finalize

!      ! store message info for synchronous mpi-communication
!      SUBROUTINE vftr_store_sync_message_info(dir, count, ftype, &
!                                              peer_rank, tag, fcomm, &
!                                              tstart, tend) &
!         BIND(c, NAME="store_sync_message_info_F")
!         USE mpi, ONLY : MPI_Datatype, &
!                         MPI_Comm
!         IMPORT c_int, c_long_long
!
!         IMPLICIT NONE
!
!         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: dir
!         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: count
!         TYPE(MPI_Datatype),        VALUE, INTENT(IN) :: ftype
!         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: peer_rank
!         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: tag
!         TYPE(MPI_Comm),            VALUE, INTENT(IN) :: fcomm
!         INTEGER(KIND=c_long_long), VALUE, INTENT(IN) :: tstart
!         INTEGER(KIND=c_long_long), VALUE, INTENT(IN) :: tend
!      END SUBROUTINE vftr_store_sync_message_info

      ! store message info for synchronous mpi-communication
      SUBROUTINE vftr_store_sync_message_info(dir, count, ftype, &
                                              peer_rank, tag, fcomm, &
                                              tstart, tend) &
         BIND(c, NAME="vftr_store_sync_message_info_F")
         IMPORT c_int, c_long_long

         IMPLICIT NONE

         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: dir
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: count
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: ftype
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: peer_rank
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: tag
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: fcomm
         INTEGER(KIND=c_long_long), VALUE, INTENT(IN) :: tstart
         INTEGER(KIND=c_long_long), VALUE, INTENT(IN) :: tend
      END SUBROUTINE vftr_store_sync_message_info

      ! store message info for asynchronous mpi-communication
      SUBROUTINE vftr_register_request(dir, count, ftype, &
                                       peer_rank, tag, fcomm, &
                                       frequest, tstart) &
         BIND(c, NAME="vftr_register_request_F")
         IMPORT c_int, c_long_long

         IMPLICIT NONE

         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: dir
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: count
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: ftype
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: peer_rank
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: tag
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: fcomm
         INTEGER(KIND=c_int),       VALUE, INTENT(IN) :: frequest
         INTEGER(KIND=c_long_long), VALUE, INTENT(IN) :: tstart
      END SUBROUTINE vftr_register_request

      ! test the entire list of open request for completed communication
      SUBROUTINE vftr_clear_completed_request() &
         BIND(c, NAME="vftr_clear_completed_request_F")
         IMPLICIT NONE
      END SUBROUTINE vftr_clear_completed_request

      ! Translate a rank from a local group to the global rank
      FUNCTION vftr_local2global_rank(fcomm, local_rank) RESULT(grank) &
         BIND(c, NAME="vftr_local2global_rank_F")
         IMPORT c_int
         IMPLICIT NONE

         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: fcomm
         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: local_rank
         INTEGER(KIND=c_int) :: grank
      END FUNCTION vftr_local2global_rank

      ! Translate a rank from a remote group to the global rank
      FUNCTION vftr_remote2global_rank(fcomm, remote_rank) RESULT(grank) &
         BIND(c, NAME="vftr_remote2global_rank_F")
         IMPORT c_int
         IMPLICIT NONE

         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: fcomm
         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: remote_rank
         INTEGER(KIND=c_int) :: grank
      END FUNCTION vftr_remote2global_rank

      ! mark a MPI_Status as empty
      SUBROUTINE vftr_empty_mpi_status(fstatus) &
         BIND(c, NAME="vftr_empty_mpi_status_F")
         USE mpi, ONLY: MPI_STATUS_SIZE
         IMPLICIT NONE
         
         INTEGER, INTENT(INOUT) :: fstatus(MPI_STATUS_SIZE)
      END SUBROUTINE vftr_empty_mpi_status

      ! check if a status is empty
      FUNCTION vftr_mpi_status_is_empty(fstatus) RESULT(isempty) &
         BIND(c, NAME="vftr_mpi_status_is_empty_F")
         IMPORT c_bool
         IMPLICIT NONE

         INTEGER, VALUE, INTENT(IN) :: fstatus
         LOGICAL(KIND=c_bool) :: isempty
      END FUNCTION vftr_mpi_status_is_empty

      ! check if a request is active
      FUNCTION vftr_mpi_request_is_active(frequest) RESULT(isactive) &
         BIND(c, NAME="vftr_mpi_request_is_active_F")
         IMPORT c_int, c_bool
         IMPLICIT NONE
         
         INTEGER(KIND=c_int), VALUE, INTENT(IN) :: frequest
         LOGICAL(KIND=c_bool) :: isactive
      END FUNCTION vftr_mpi_request_is_active

      ! Obtain current cycle count for timing purposes
      FUNCTION vftr_get_runtime_usec() RESULT(time) &
         BIND(c, NAME="vftr_get_runtime_usec_F")
         IMPORT c_long_long
         IMPLICIT NONE

         INTEGER(KIND=c_long_long) :: time
      END FUNCTION vftr_get_runtime_usec
   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_F2C_interface
