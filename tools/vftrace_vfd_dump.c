#include <stdlib.h>
#include <stdio.h>

#include <assert.h>

#include "vfd_types.h"
#include "vfd_utils.h"

int main(int argc, char **argv) {

   char *filename = argv[1];

   FILE *fp = fopen(filename, "r");
   assert(fp);

   vfd_header_t vfd_header = read_vfd_header(fp);

   print_vfd_header(stdout, vfd_header);
   free_vfd_header(&vfd_header);
}
