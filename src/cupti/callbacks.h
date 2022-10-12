#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <cupti.h>

void CUPTIAPI vftr_cupti_event_callback (void *userdata,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid,
                                         const CUpti_CallbackData *cb_info);

enum {T_CUPTI_COMP, T_CUPTI_MEMCP, T_CUPTI_OTHER};
enum {CUPTI_NOCOPY=-1,CUPTI_COPY_IN=0, CUPTI_COPY_OUT=1};

#endif
