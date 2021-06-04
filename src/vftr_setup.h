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
#ifndef VFTR_SETUP_H
#define VFTR_SETUP_H

#include <stdio.h>
#include <stdbool.h>

extern bool vftr_timer_end;

extern int vftr_mpirank;
extern int vftr_mpisize;
extern unsigned int vftr_function_samplecount;
extern unsigned int vftr_message_samplecount;

extern bool vftr_do_stack_normalization;

extern char *vftr_start_date;
extern char *vftr_end_date;
extern bool in_vftr_finalize;

void vftr_initialize ();
void vftr_finalize () ;

void vftr_get_mpi_info (int *rank, int *size);

// test functions
int vftr_setup_test_1 (FILE *fp);
int vftr_setup_test_2 (FILE *fp);

#endif
