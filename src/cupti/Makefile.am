noinst_LTLIBRARIES = libvftr_cupti.la

libvftr_cupti_la_SOURCES = 

libvftr_cupti_la_SOURCES += initialize.c \
                            callbacks.c \
                            cupti_events.c \
                            cuptiprofiling.c \
                            hook_interface.c

libvftr_cupti_la_CFLAGS = -I${srcdir}/../ -D_CUPTI

libvftr_cupti_la_LDFLAGS = -L/usr/local/cuda-11.2/lib64 -lcupti -lcudart

