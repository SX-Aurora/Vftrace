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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>

#define MAX_CMDLINE 1024
char *get_application_name () {
	char *program_path;

	char proccmd[40];
	char cmdline[MAX_CMDLINE];	

    	snprintf (proccmd, 39, "/proc/%d/cmdline", getpid());
        int last = 0;
        int fd = open (proccmd, O_RDONLY);
        char *p;
        if (fd > 0) {
            p = cmdline;
	    char *pend = p + read (fd, cmdline, MAX_CMDLINE);
#ifdef __ve__
            /* Skip ve_exec and its options */
            while (p < pend && !last) {
                last = !strcmp(p,"--");
                p += strlen(p) + 1;
            }
#endif
            program_path = strdup(p);
    	    close (fd);
        } else {
            program_path = NULL;
        }
        return program_path;
}

/**********************************************************************/

int count_digits (int value) {
  if (value == 0) {
     return 1;
  } else {
     int count = 0;
     for (int c = value; c > 0; c /= 10) {
           count++;
     }
     return count;
  }
}

/**********************************************************************/

int count_digits_double (double value) {
  return count_digits((int)floor(value));
}

/**********************************************************************/

char *vftr_bool_to_string (bool value) {
	return value ? "true" : "false";
}

/**********************************************************************/
