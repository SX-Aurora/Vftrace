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


