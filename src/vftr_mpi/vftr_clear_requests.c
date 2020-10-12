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
#include "vftr_p2p_requests.h"
#include "vftr_persistent_requests.h"
#include "vftr_collective_requests.h"
#include "vftr_onesided_requests.h"

void vftr_clear_completed_requests() {
   vftr_clear_completed_P2P_requests();
   vftr_deactivate_completed_persistent_requests();
   vftr_clear_completed_collective_requests();
   vftr_clear_completed_onesided_requests();

   vftr_deallocate_marked_p2p_requests();
   vftr_deallocate_marked_persistent_requests();
}

#endif
