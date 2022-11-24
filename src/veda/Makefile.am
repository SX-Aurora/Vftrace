noinst_LTLIBRARIES = libvftr_veda.la

libvftr_veda_la_SOURCES =

libvftr_veda_la_SOURCES += veda_callback.c

libvftr_veda_la_CFLAGS = -I${srcdir}/../ -I${srcdir}/../sorting -D_VEDA

if SELF_PROFILING
libvftr_veda_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif
