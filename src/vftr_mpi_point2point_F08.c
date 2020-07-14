/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifdef _MPI
#include <mpi.h>

#include "vftr_mpi_point2point.h"

void vftr_MPI_Send_F08(char *buf, MPI_Fint *count, MPI_Fint *fdatatype,
                       MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *fcomm,
                       MPI_Fint *ferror) {

   MPI_Comm ccomm = PMPI_Comm_f2c(*fcomm);
   MPI_Datatype cdatatype = PMPI_Type_f2c(*fdatatype);

   int cerror = vftr_MPI_Send(buf,
                              (int)(*count),
                              cdatatype,
                              (int)(*dest),
                              (int)(*tag),
                              ccomm);

   *ferror = (MPI_Fint) (cerror);
}

int vftr_MPI_Recv_F08(char *buf, MPI_Fint *count, MPI_Fint *fdatatype,
                      MPI_Fint *source, MPI_Fint *tag, MPI_Fint *fcomm,
                      MPI_Fint *fstatus, MPI_Fint *ferror) {
   
   MPI_Comm ccomm = PMPI_Comm_f2c(*fcomm);
   MPI_Datatype cdatatype = PMPI_Type_f2c(*fdatatype);
   MPI_Status cstatus;
   // PMPI_Status_f082c(fstatus, cstatus);
   PMPI_Status_f2c(fstatus, &cstatus);

   int cerror = vftr_MPI_Recv(buf,
                              (int)(*count),
                              cdatatype,
                              (int)(*source),
                              (int)(*tag),
                              ccomm,
                              &cstatus);

   //PMPI_Status_f082c(fstatus, &cstatus);
   PMPI_Status_f2c(fstatus, &cstatus);

   *ferror = (MPI_Fint) (cerror);

}

#endif
