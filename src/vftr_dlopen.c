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
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "vftr_dlopen.h"

static void (*real_dlopen)(const char *filename, int flag)=NULL;

static void real_dlopen_init () {
	real_dlopen = dlsym(RTLD_NEXT, "dlopen");
	if (real_dlopen == NULL) {
		printf ("dlopen: Internal error\n");
	}
}

void *dlopen (const char *filename, int flag) {
	if (real_dlopen == NULL) {
		real_dlopen_init();
	}
	
	filename += 2;
	dlopened_lib = malloc (strlen(filename));
	strcpy(dlopened_lib, filename);
	lib_opened = 1;
	real_dlopen (filename, flag);
}
