noinst_LTLIBRARIES = libvftr_veda.la

libvftr_veda_la_SOURCES =

libvftr_veda_la_SOURCES += veda_regions.c

libvftr_veda_la_SOURCES += vedaprofiling.c \
			   collate_vedaprofiles.c \
			   logfile_veda_table.c \
			   ranklogfile_veda_table.c

libvftr_veda_la_SOURCES += veda_callback.c \
			   veda_callback_launch_kernel.c \
			   veda_callback_launch_host.c \
			   veda_callback_mem_cpy_htod.c \
			   veda_callback_mem_cpy_dtoh.c

libvftr_veda_la_CFLAGS =
if SELF_PROFILING
libvftr_veda_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif
