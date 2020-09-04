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
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <byteswap.h>
#include "vftr_filewrite.h"
#include "vftr_mpi_utils.h"

#include "vftr_vfd_utils.h"

#define RECORD_LENGTH 10240

static int
cmpstring( const void *p1, const void *p2 )
{
    function_entry_t *e1 = ( function_entry_t * ) p1;
    function_entry_t *e2 = ( function_entry_t * ) p2;

    return strcmp( e1->name, e2->name );
}

int main (int argc, char **argv) {
    FILE      *fp;
    char       record[RECORD_LENGTH], *s;
    char      *typename[34];
    int        i, j, samples, nextID, levels, caller, n_precise_functions; 
    long file_size, max_fp;
    double     dtime = 0.;
    char *filename;
    function_entry_t *functions = NULL;
    vfd_header_t vfd_header;

    stack_entry_t *stacks = NULL;

    struct StackInfo {
        int id, levels, caller, len;
    } stackInfo;
    
    int this_vfd_version;

    n_precise_functions   = 0;
    
    if (argc < 2) {
	    printf ("Usage: tracedump <vfd-file>\n");
	    return -1;
    } else if (argc > 2) {
	    	printf ("Attention: More than one input file. Only the first is processed\n");
    }

    filename = argv[1];
    fp = fopen (filename, "r");
    assert (fp);
    fseek (fp, 0L, SEEK_END);
    file_size = ftell(fp);
    max_fp = 0;
    rewind(fp);

    printf ("Reading: %s; Size: %ld bytes\n", filename, file_size);

    fread (&this_vfd_version, 1, sizeof(int), fp);
    printf ("VFD version: %d\n", this_vfd_version);
    if (this_vfd_version != VFD_VERSION) {
	printf ("The file %s does not have the most recent VFD version (%d)!\n",
		filename, VFD_VERSION);  
	return -1;
    }

    read_fileheader (&vfd_header, fp);

    printf ("header size = %ld offset = %ld\n",
	     sizeof(struct FileHeader), ftell(fp));
    print_fileheader (vfd_header);

    fread (&(vfd_header.n_perf_types), sizeof(int), 1, fp);
    printf ("n_perf_types: %d\n", vfd_header.n_perf_types);
    double *perf_values = NULL;
    if (vfd_header.n_perf_types > 0) {
	init_hw_observables (fp, vfd_header.n_perf_types, &perf_values);
    }
    
    printf( "Unique stacks:   %d\n",                 vfd_header.stackscount  );
    printf ("Stacks list:\n");
    read_stacks (fp, &stacks, &functions,
                 vfd_header.stackscount, vfd_header.stacksoffset,
		 &n_precise_functions, &max_fp, false);
    for (int i = 0; i < vfd_header.stackscount; i++) {
	if (stacks[i].name) printf ("      %d,%d,%d,%s\n",
		i, stacks[i].levels, stacks[i].caller, stacks[i].name);
    }
			
    
    if (ftell(fp) > max_fp) max_fp = ftell(fp);
    fseek (fp, vfd_header.sampleoffset, SEEK_SET);

    printf ("\nStack and message samples:\n\n");

    for(i = 0; i < vfd_header.samplecount; i++ ) {
        int        sidw;
	long       pos;

        fread (&sidw, sizeof(int), 1, fp);

        if (sidw == SID_MESSAGE) {
	    int direction, rank, tag, count;
	    int type_size, type_index;
	    double dt_start, dt_stop, rate;
	    read_mpi_message_sample (fp, &direction, &rank, &type_index, &type_size,
				     &count, &tag, &dt_start, &dt_stop, &rate);
					
            printf("%16.6f %s\n", dt_start, direction ? "recv" : "send");
            printf("%16s count=%d type=%s(%iBytes) rate= %8.4lf MiB/s peer=%d tag=%d\n",
                   "", count, vftr_get_mpitype_string_from_idx(type_index), 
                   type_size, rate, rank, tag);
            printf("%16.6f %s end\n", dt_stop, direction ? "recv" : "send");
        } else if (sidw == SID_ENTRY || sidw == SID_EXIT) {
	    int stack_id;
	    long long sample_time;
	    read_stack_sample (fp, vfd_header.n_perf_types, &stack_id, &sample_time, &perf_values);
            double sample_time_s = (double)sample_time * 1.0e-6;

            if (stacks[stack_id].fun != -1) {
                if (sidw == SID_ENTRY) {
			stacks[stack_id].entry_time = sample_time_s;
                } else {
			functions[stacks[stack_id].fun].elapse_time += (sample_time_s - stacks[stack_id].entry_time);
                }
            }

            printf("%16.6f %s ", sample_time_s, sidw == SID_ENTRY ? "call" : "exit");
            for( ;; stack_id = stacks[stack_id].caller ) {
               printf( "%s%s", stacks[stack_id].name, stack_id ? "<" : "" );
               if(stack_id == 0) break;
             }
	     printf("\n");
	} else {
            printf("ERROR: Invalid sample type: %d\n", sidw);
            return 1;
        }
    }
    printf ("\n");

    if (file_size != max_fp) {
	    printf ("WARNING: Not all data have been read!\n");
	    printf ("Currently at %ld, but the file has %ld bytes.\n",
		    max_fp, file_size);
    } else {
	    printf ("SUCCESS: All bytes have been read\n");
    }
    fclose( fp );

    if(n_precise_functions != 0) {
        int l = 0;
        char fmt[16];

        qsort(functions, n_precise_functions, sizeof(function_entry_t), cmpstring);

        for( i = 0; i < n_precise_functions; i++ ) {
            if( strlen(functions[i].name) > l )
                l = strlen(functions[i].name);
        }
        sprintf( fmt, "%%%ds %%12.6f\n", l + 2 );
        fprintf( stdout, "\nElapse time for \"precise\" functions (including sub routine):\n\n" );
        for( i = 0; i < n_precise_functions; i++ ) {
            fprintf( stdout, fmt, functions[i].name, functions[i].elapse_time );
        }
    }

    if (perf_values) free (perf_values);
    free (stacks);
    free (functions);

    return 0;
}
