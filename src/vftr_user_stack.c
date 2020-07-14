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
#include <string.h>

#include "vftr_omp.h"
#include "vftr_stacks.h"

int vftrace_get_stack_string_length() {
   int me = OMP_GET_THREAD_NUM;
   function_t *func = vftr_fstack[me];

   int stackstrlength = strlen(func->name);
   function_t *tmpfunc = func;
   // go down the stack until the bottom is reached
   // record the length of the function names each
   while (tmpfunc->ret) {
      tmpfunc = tmpfunc->ret;
      // add one chars for function division by "<"
      stackstrlength += 1;
      stackstrlength += strlen(tmpfunc->name);
   }

   return stackstrlength;
}

char *vftrace_get_stack() {
   int me = OMP_GET_THREAD_NUM;
   function_t *func = vftr_fstack[me];

   // determine the length of the stack string
   int stackstrlength = vftrace_get_stack_string_length();

   // allocate space to hold the complete string
   char *stackstr = (char*) malloc((1+stackstrlength)*sizeof(char));
   char *strptr = stackstr;
   function_t *tmpfunc = func;
   // copy the first string in and move the strpointer forward
   strcpy(strptr, tmpfunc->name);
   strptr += strlen(tmpfunc->name);
   // go down the stack until the bottom is reached
   // copy the function names onto the string
   while (tmpfunc->ret) {
      tmpfunc = tmpfunc->ret;
      strcpy(strptr, "<");
      strptr += 1;
      strcpy(strptr, tmpfunc->name);
      strptr += strlen(tmpfunc->name);
   }

   return stackstr;
}
