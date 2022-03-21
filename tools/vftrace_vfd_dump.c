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

   stack_t *stacklist = read_stacklist(fp,
                                       vfd_header.stacks_offset,
                                       vfd_header.nstacks);

   print_stacklist(stdout, vfd_header.nstacks, stacklist);

   print_samples(fp, stdout, vfd_header, stacklist);


   free_vfd_header(&vfd_header);
   free_stacklist(vfd_header.nstacks, stacklist);

   fclose(fp);
}
