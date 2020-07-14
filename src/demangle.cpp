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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cxxabi.h>

/****************************************************************************

demangle - Demangle C++ symbol while omitting types and arguments

int demangle( char *mangled_name, char **demangled_name )

mangled_name     C++ generated name
demangled_name   Original function names without details
demangled_full   Original function names with details

Return           1 if demangled, 0 if original

****************************************************************************/
extern "C" int
demangle( char *mangled_name, char **demangled_name, char **demangled_full ) {
    int status;
    char *demangled = abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
    *demangled_full = demangled;
    if( !status ) {
        // Remove spaces, types and arguments
        char *cout = (char *) malloc( strlen(demangled) * sizeof(char) );
        int level = 0;
        *demangled_name = cout;
        for( ; *demangled; demangled++ ) {
            if( *demangled == '(' ) break;
            if( *demangled == ' ' ) continue;
            if( *demangled == '<' ) { level++; continue; }
            if( *demangled == '>' ) { level--; continue; }
            if( !level            ) *cout++ = *demangled;
        }
        *cout = 0;
        /* Consider demangling as failed if string is empty */
        if( !strcmp(*demangled_name,"std::") ||
            (strlen(*demangled_name) == 0  )    )
            status = -1;
    }
    if( status ) *demangled_name = strdup(mangled_name);
    return status ? 0 : 1;
}
