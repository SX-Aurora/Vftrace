#ifndef CUDA_VFTR_CALLBACKS_H
#define CUDA_VFTR_CALLBACKS_H

// You might think that this file might be better called
// "cupti_callbacks.h" to better suit the names of the other 
// files in this directory. However, there is a file named
// "cupti_callbacks.h" in the cuda include directory, which 
// is also included by the statement below. To avoid this
// collision, we have added the "vftr" interfix.

#include <cupti.h>

void CUPTIAPI vftr_cupti_event_callback (void *userdata,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
                                         const CUpti_CallbackData *cb_info);

enum {T_CUDA_COMP, T_CUDA_MEMCP, T_CUDA_OTHER};
enum {CUDA_NOCOPY=-1, CUDA_COPY_IN=0, CUDA_COPY_OUT=1};

#endif
