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

#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

// returns the processes current maximum resident set size
long vftr_current_memory_usage() {

#ifdef __ve__
   struct rusage myrusage;
   getrusage(RUSAGE_SELF, &myrusage);
   return myrusage.ru_maxrss;
#else
   static bool need_pagesize = true;
   static int pagesize = 0;
   if (need_pagesize) {
      pagesize = getpagesize();
   }

   FILE *fileptr = fopen("/proc/self/statm", "r");

   long size = 0;
   long resident = 0;

   fscanf(fileptr,"%ld %ld", &size, &resident);
   fclose(fileptr);

   return (double) resident * pagesize;
#endif
}
