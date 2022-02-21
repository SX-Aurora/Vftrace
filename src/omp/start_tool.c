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

#include <stdlib.h>

#include <omp.h>
#include <omp-tools.h>

#include "initialize.h"
#include "finalize.h"

ompt_start_tool_result_t vftr_ompt_start_tool_result;

ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version,
                                          const char *runtime_version) {
   if (omp_version == 0 && runtime_version == NULL) {return NULL;}
printf("starting tool\n");

   vftr_ompt_start_tool_result.initialize = ompt_initialize_ptr;
   vftr_ompt_start_tool_result.finalize = ompt_finalize_ptr;

   return &vftr_ompt_start_tool_result; // success: registers tool
}
