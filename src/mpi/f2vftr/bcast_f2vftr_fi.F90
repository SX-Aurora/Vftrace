MODULE vftr_mpi_bcast_f2vftr_fi
#ifdef _MPI

   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE, INTRINSIC :: ISO_C_BINDING

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: vftr_MPI_Bcast_f2vftr

   INTERFACE

      SUBROUTINE vftr_MPI_Bcast_f2vftr(BUFFER, COUNT, F_DATATYPE, &
                                    ROOT, F_COMM, F_ERROR) &
         BIND(C, name="vftr_MPI_Bcast_f2vftr")
         IMPLICIT NONE
         INTEGER BUFFER
         INTEGER COUNT
         INTEGER F_DATATYPE
         INTEGER ROOT
         INTEGER F_COMM
         INTEGER F_ERROR
      END SUBROUTINE vftr_MPI_Bcast_f2vftr

   END INTERFACE

#endif

CONTAINS

END MODULE vftr_mpi_bcast_f2vftr_fi
