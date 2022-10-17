#ifndef CUPTI_VFTR_CALLBACKS_H
#define CUPTI_VFTR_CALLBACKS_H

// You might think that this file might be better called
// "cupti_callbacks.h" to better suit the names of the other 
// files in this directory. However, there is a file named
// "cupti_callbacks.h" in the cupti include directory, which 
// is also included by the statement below. To avoid this
// collision, we have added the "vftr" interfix.

#include <cupti.h>

void CUPTIAPI vftr_cupti_event_callback (void *userdata,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
                                         const CUpti_CallbackData *cb_info);

enum {T_CUPTI_COMP, T_CUPTI_MEMCP, T_CUPTI_OTHER};
enum {CUPTI_NOCOPY=-1,CUPTI_COPY_IN=0, CUPTI_COPY_OUT=1};

#endif
