#include <stdio.h>

void vftr_print_licence_short(FILE *fp, char *licence_verbose_name) {
   fprintf(fp,
           "This is free software with ABSOLUTELY NO WARRANTY.\n"
           "For details: use vftrace with environment variable %s=yes\n\n",
           licence_verbose_name);
}

void vftr_print_licence(FILE *fp) {
   fprintf(fp,
           "This program is free software; you can redistribute it and/or modify\n"
           "it under the terms of the GNU General Public License as published by\n"
           "the Free Software Foundation; either version 2 of the License , or\n"
           "(at your option) any later version.\n\n"
           "This program is distributed in the hope that it will be useful,\n"
           "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
           "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
           "GNU General Public License for more details.\n\n"
           "You should have received a copy of the GNU General Public License\n"
           "along with this program. If not, write to\n\n"
           "   The Free Software Foundation, Inc.\n"
           "   51 Franklin Street, Fifth Floor\n"
           "   Boston, MA 02110-1301  USA\n\n" );
}
