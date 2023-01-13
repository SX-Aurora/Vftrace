noinst_LTLIBRARIES = libvftr_sorting.la
libvftr_sorting_la_LIBADD =
libvftr_sorting_la_SOURCES = sort_utils.c \
			     sorting_symboltable.c \
			     sorting_stacks.c \
			     sorting_collated_stacks.c
libvftr_sorting_la_CFLAGS = -I${srcdir}/../
if ENABLE_MPI
libvftr_sorting_la_CFLAGS += -D_MPI -I${srcdir}/../mpi/utils/
endif
if ENABLE_OMP
libvftr_sorting_la_CFLAGS += -D_OMP -I${srcdir}/../omp/
endif
if ENABLE_CUDAPROF
libvftr_sorting_la_CFLAGS += -D_CUDA -I${srcdir}/../cuda
endif
if ENABLE_ACCPROF
libvftr_sorting_la_CFLAGS += -D_ACCPROF -I${srcdir}/../accprof
endif
if ENABLE_HWPROF
libvftr_sorting_la_CFLAGS += -D_HWPROF -I${srcdir}/../hwprof
endif

noinst_LTLIBRARIES += libvftr_sorting_int8.la
libvftr_sorting_la_LIBADD += libvftr_sorting_int8.la
libvftr_sorting_int8_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_int8_la_CFLAGS = -DTYPE="int8_t" -DTYPESTR="int8"

noinst_LTLIBRARIES += libvftr_sorting_int.la
libvftr_sorting_la_LIBADD += libvftr_sorting_int.la
libvftr_sorting_int_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_int_la_CFLAGS = -DTYPE="int" -DTYPESTR="int"

noinst_LTLIBRARIES += libvftr_sorting_uint8.la
libvftr_sorting_la_LIBADD += libvftr_sorting_uint8.la
libvftr_sorting_uint8_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_uint8_la_CFLAGS = -DTYPE="uint8_t" -DTYPESTR="uint8"

noinst_LTLIBRARIES += libvftr_sorting_uint64.la
libvftr_sorting_la_LIBADD += libvftr_sorting_uint64.la
libvftr_sorting_uint64_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_uint64_la_CFLAGS = -DTYPE="uint64_t" -DTYPESTR="uint64"

noinst_LTLIBRARIES += libvftr_sorting_long_long.la
libvftr_sorting_la_LIBADD += libvftr_sorting_long_long.la
libvftr_sorting_long_long_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_long_long_la_CFLAGS = -DTYPE="long long" -DTYPESTR="longlong"

noinst_LTLIBRARIES += libvftr_sorting_float.la
libvftr_sorting_la_LIBADD += libvftr_sorting_float.la
libvftr_sorting_float_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_float_la_CFLAGS = -DTYPE="float" -DTYPESTR="float" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_sorting_double.la
libvftr_sorting_la_LIBADD += libvftr_sorting_double.la
libvftr_sorting_double_la_SOURCES = sorting_stdtypes.c
libvftr_sorting_double_la_CFLAGS = -DTYPE="double" -DTYPESTR="double" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_int8.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_int8.la
libvftr_apply_perm_int8_la_SOURCES = apply_perm.c
libvftr_apply_perm_int8_la_CFLAGS = -DTYPE="int8_t" -DTYPESTR="int8" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_uint8.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_uint8.la
libvftr_apply_perm_uint8_la_SOURCES = apply_perm.c
libvftr_apply_perm_uint8_la_CFLAGS = -DTYPE="uint8_t" -DTYPESTR="uint8" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_float.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_float.la
libvftr_apply_perm_float_la_SOURCES = apply_perm.c
libvftr_apply_perm_float_la_CFLAGS = -DTYPE="float" -DTYPESTR="float" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_double.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_double.la
libvftr_apply_perm_double_la_SOURCES = apply_perm.c
libvftr_apply_perm_double_la_CFLAGS = -DTYPE="double" -DTYPESTR="double" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_longlong.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_longlong.la
libvftr_apply_perm_longlong_la_SOURCES = apply_perm.c
libvftr_apply_perm_longlong_la_CFLAGS = -DTYPE="long long" -DTYPESTR="longlong" -I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_charptr.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_charptr.la
libvftr_apply_perm_charptr_la_SOURCES = apply_perm.c
libvftr_apply_perm_charptr_la_CFLAGS = -DTYPE="char*" -DTYPESTR="charptr" -I${srcdir}/../


noinst_LTLIBRARIES += libvftr_apply_perm_stackptr.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_stackptr.la
libvftr_apply_perm_stackptr_la_SOURCES = apply_perm.c
libvftr_apply_perm_stackptr_la_CFLAGS = -DTYPE="vftr_stack_t*" -DTYPESTR="stackptr" \
					-I${srcdir}/../

noinst_LTLIBRARIES += libvftr_apply_perm_collated_stackptr.la
libvftr_sorting_la_LIBADD += libvftr_apply_perm_collated_stackptr.la
libvftr_apply_perm_collated_stackptr_la_SOURCES = apply_perm.c
libvftr_apply_perm_collated_stackptr_la_CFLAGS = -DTYPE="collated_stack_t*" \
						 -DTYPESTR="collated_stackptr" \
						 -I${srcdir}/../

if ENABLE_MPI
libvftr_apply_perm_stackptr_la_CFLAGS += -D_MPI -I${srcdir}/../mpi/utils/
endif
if ENABLE_OMP
libvftr_apply_perm_stackptr_la_CFLAGS += -D_OMP -I${srcdir}/../omp/
endif
