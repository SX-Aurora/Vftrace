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
#include <ctype.h>

#include "vftr_environment.h"
#include "vftr_setup.h"

#define MAX_CMDLINE 1024
char *vftr_get_application_name () {
	char *program_path;

	char proccmd[40];
	char cmdline[MAX_CMDLINE];	

    	snprintf (proccmd, 39, "/proc/%d/cmdline", getpid());
        int fd = open (proccmd, O_RDONLY);
        char *p;
        if (fd > 0) {
            p = cmdline;
	    char *pend = p + read (fd, cmdline, MAX_CMDLINE);
#ifdef __ve__
            int last = 0;
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

int vftr_count_digits_long (long long value) {
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

int vftr_count_digits_int (int value) {
  return vftr_count_digits_long ((long long )value);
}

/**********************************************************************/

int vftr_count_digits_double (double value) {
  return vftr_count_digits_long ((long long)floor(value));
}

/**********************************************************************/

char *vftr_bool_to_string (bool value) {
	return value ? "true" : "false";
}

/**********************************************************************/

void vftr_print_dashes (FILE *fp, int n) {
	for (int i = 0; i < n; i++) fprintf (fp, "-");
	fprintf (fp, "\n");
}

/**********************************************************************/

char *vftr_to_lowercase (char *s_orig) {
   char *s_lower = strdup(s_orig);
   for (int i = 0; i < strlen(s_orig); i++) {
      s_lower[i] = tolower(s_orig[i]); 
   }
   return s_lower;
}

/**********************************************************************/

// Output an error message to the log files, but only write to rank 0 VFTR_LOGFILE_ALL_RANKS
// is not set. Otherwise, i.e. if only an fprintf is used, there might be a multitude of unwanted
// log files containing only this message.
void vftr_logfile_warning (FILE *fp, char *message) {
  if (vftr_rank_needs_logfile()) fprintf (fp, message);
} 
