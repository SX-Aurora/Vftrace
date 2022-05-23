#ifndef VFTR_DLOPEN_H
#define VFTR_DLOPEN_H

extern int lib_opened;
extern char *dlopened_lib;

void vftr_copy_symtab(int rank, char *foo);

#endif
