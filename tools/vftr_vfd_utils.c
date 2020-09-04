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

#include "vftr_filewrite.h"
#include "vftr_vfd_utils.h"

void read_fileheader (vfd_header_t *vfd_header, FILE *fp) {
    fread (&vfd_header->fileid, 1, VFTR_FILEIDSIZE, fp);
    fread (&vfd_header->date, 1, 24, fp);
    fread (&vfd_header->interval, 1, sizeof(long long), fp);
    fread (&vfd_header->threads, 1, sizeof(int), fp);
    fread (&vfd_header->thread,	1, sizeof(int), fp);
    fread (&vfd_header->tasks, 1, sizeof(int), fp);
    fread (&vfd_header->task, 1, sizeof(int), fp);
    fread (&vfd_header->cycletime.l, 1, sizeof(long long), fp);
    fread (&vfd_header->inittime, 1, sizeof(long long), fp);
    fread (&vfd_header->runtime.l, 1, sizeof(long long), fp);
    fread (&vfd_header->samplecount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->stackscount, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->stacksoffset, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->sampleoffset, 1, sizeof(unsigned int), fp);
    fread (&vfd_header->reserved, 1, sizeof(unsigned int), fp);
}

/**********************************************************************/

bool is_precise (char *s) {
 	return s[strlen(s)-1] == '*';
}

/**********************************************************************/

char *strip_trailing_asterisk (char *s) {
	char *sdup = strdup(s);
	int n = strlen(sdup);
	if (sdup[n-1] == '*') {
		sdup[n-1] = '\0';
	}
	return sdup;
}

/**********************************************************************/

void read_stacks (FILE *fp, stack_entry_t **stacks, function_entry_t **functions, 
		  unsigned int stacks_count, unsigned int stacks_offset,
                  int *n_precise_functions, long *max_fp) {

    struct StackInfo {
        int id, levels, caller, len;
    } stackInfo;

    char record[RECORD_LENGTH];

    *stacks = (stack_entry_t *) malloc (stacks_count * sizeof(stack_entry_t));
    if (max_fp && ftell(fp) > *max_fp) {
       *max_fp = ftell(fp);
    }
    fseek (fp, stacks_offset, SEEK_SET);
    
    for (int i = 0; i < stacks_count; i++) {
	fread (&stackInfo, sizeof(int), 4, fp);
        int len = stackInfo.len < RECORD_LENGTH ? stackInfo.len : RECORD_LENGTH - 1;
	fread (record, sizeof(char), len, fp);
	record[len] = '\0';
        (*stacks)[stackInfo.id].name = strip_trailing_asterisk(record);
	(*stacks)[stackInfo.id].caller = stackInfo.caller;
	(*stacks)[stackInfo.id].precise = is_precise(record);

        (*stacks)[stackInfo.id].fun = -1;
        if ((*stacks)[stackInfo.id].precise) {
            for (int j = 0; j < *n_precise_functions; j++) {
                if (!strcmp (record, (*functions)[j].name)) {
                    (*stacks)[stackInfo.id].fun = j;
                    break;
                }
            }
	    
            if ((*stacks)[stackInfo.id].fun == -1 ) {
                (*n_precise_functions)++;
                if  (*n_precise_functions == 1) {
                    *functions = (struct FunctionEntry *) malloc (*n_precise_functions * sizeof(struct FunctionEntry) );
                } else {
                    *functions = (struct FunctionEntry *) realloc (*functions, *n_precise_functions * sizeof(struct FunctionEntry) );
                }

                (*functions)[*n_precise_functions - 1].name = strip_trailing_asterisk (record);
                (*functions)[*n_precise_functions - 1].elapse_time = 0.0;
            }
        }

        for (int j = 0; j < *n_precise_functions; j++) {
            if (!strcmp ((*stacks)[stackInfo.id].name, (*functions)[j].name)) {
                (*stacks)[stackInfo.id].fun = j;
                break;
            }
        }
    }
}

/**********************************************************************/

void print_fileheader (vfd_header_t vfd_header) {
    int i;
    // We require a seperate datestring which is one element larger than
    // the field in the vfd file to add a terminating null character. This
    // is not necessary for the version string since it is written using sprintf
    // in the main code.
    char       datestring[25], record[RECORD_LENGTH], *s;
    datestring[24] = 0;
    strncpy (datestring, vfd_header.date, 24);

    printf ("Version ID: %s\n", vfd_header.fileid    );
    printf ("Date:       %s\n", datestring );
    printf ("MPI tasks:  rank=%d count=%d\n", vfd_header.task, vfd_header.tasks); 
    printf ("OpenMP threads:  thread=%d count=%d\n", vfd_header.thread, vfd_header.threads);
    printf ("Sample interval: %12.6le seconds\n", vfd_header.interval*1.0e-6);
    printf ("Init time:     %lld\n", vfd_header.inittime);
    printf ("Job runtime:   %.3lf seconds\n", vfd_header.runtime.d);
    printf ("Samples:       %d\n", vfd_header.samplecount );
    printf ("Unique stacks: %d\n", vfd_header.stackscount);
    printf ("Stacks offset: %d\n", vfd_header.stacksoffset);
    printf ("Sample offset: %d\n", vfd_header.sampleoffset);
}

/**********************************************************************/


