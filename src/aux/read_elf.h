#ifndef READ_ELF_H
#define READ_ELF_H

#define LINESIZE 200

off_t base_addr;
off_t symbol_addr;
off_t enter_addr;
int read_maps ();
int read_elf (char *this_fn);

#endif
