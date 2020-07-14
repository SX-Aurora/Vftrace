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

#ifndef VFTR_HASHING_H
#define VFTR_HASHING_H

#include <stdint.h>

uint64_t vftr_jenkins_murmur_64_hash(size_t length, const uint8_t* key);

void vftr_remove_multiple_hashes(int *n, uint64_t *hashlist);

void vftr_synchronise_hashes(int *nptr, uint64_t **hashlistptr);

#endif
