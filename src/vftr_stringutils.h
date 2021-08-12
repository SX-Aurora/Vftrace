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
#ifndef VFTR_STRINGUTILS_H
#define VFTR_STRINGUTILS_H

#include <stdarg.h>
#include <stdbool.h>

void vftr_rank0_printf (const char *fmt, ...);
int vftr_levenshtein_distance (char *a, char *b);

void vftr_has_control_character (char *s, int *pos, int *char_num);

bool vftr_string_is_number (char *s_check);

#endif
