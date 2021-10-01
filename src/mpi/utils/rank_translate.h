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

#ifndef RANK_TRANSLATE_H
#define RANK_TRANSLATE_H

#include <mpi.h>

// Translate a rank from a local group to the global rank
int vftr_local2global_rank(MPI_Comm comm, int local_rank);

// Translate a rank from a remote group to the global rank
int vftr_remote2global_rank(MPI_Comm comm, int remote_rank);

#endif
