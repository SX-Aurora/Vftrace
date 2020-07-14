#include <stdio.h>
#include "read_elf.h"

void __cyg_profile_func_enter (char *fn_addr, char *call_addr) __attribute__((no_instrument_function));
void __cyg_profile_func_enter (char *fn_addr, char *call_addr) {
	int val;
	val = read_maps ();
	val = read_elf (fn_addr);
}

void __cyg_profile_func_exit (char *fn_addr, char *call_addr) __attribute__((no_instrument_function));
void __cyg_profile_func_exit (char *fn_addr, char *call_addr) {};
