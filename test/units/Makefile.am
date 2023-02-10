if ENABLE_MPI
if USES_OPEN_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_openmpi.sh;
else
if USES_NEC_MPI
if ON_VECTOR_ENGINE
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_necvhmpi.sh;
endif
else
if USES_INTEL_MPI
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_intelmpi.sh;
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_genericmpi.sh;
endif
endif
endif
else
   AM_TESTS_ENVIRONMENT=. $(top_srcdir)/test/environment/set_nompi.sh;
endif
AM_TESTS_ENVIRONMENT+=. $(top_srcdir)/test/environment/unset_vftr_env.sh;

AM_CPPFLAGS =
AM_CFLAGS = -I$(top_srcdir)/src/ \
	    -I$(top_srcdir)/src/mpi/utils/ \
	    -I$(top_srcdir)/src/omp/ \
            -I$(top_srcdir)/src/logfiles
AM_LDFLAGS = -L$(top_builddir)/src/.libs -lvftrace -lm

if ENABLE_MPI
  AM_CFLAGS += -D_MPI
if USES_NEC_MPI
  AM_LDFLAGS += -mpiprof
endif
endif

  AM_CFLAGS += -I$(top_srcdir)/src/hwprof -I$(top_srcdir)/external/tinyexpr

if ENABLE_CUDAPROF
  AM_CFLAGS += -D_CUDA -I$(top_srcdir)/src/cuda
endif

if ENABLE_ACCPROF
  AM_CFLAGS += -D_ACCPROF -I$(top_srcdir)/src/accprof
endif

if ENABLE_OMP
  AM_CFLAGS += -D_OMP
endif

if SELF_PROFILING
  AM_CPPFLAGS += -D_SELF_PROFILE_VFTRACE
endif

check_PROGRAMS = 
check_PROGRAMS += sort_int8 \
		  sort_perm_int8 \
		  sort_uint8 \
		  sort_perm_uint8 \
		  sort_float \
		  sort_perm_float \
		  sort_int8 \
                  sort_perm_int8 \
                  sort_uint8 \
                  sort_perm_uint8 \
                  sort_float \
                  sort_perm_float \
		  sort_uint64 \
		  sort_longlong \
		  sort_double
check_PROGRAMS += search_uint64 \
		  search_int \
		  search_collated_stacks_name
check_PROGRAMS += chop_chars \
		  trim_left_delimiter \
		  range_expand \
		  levenshtein
check_PROGRAMS += symbols_1 \
		  symbols_2 \
		  symbols_precise \
		  pauseresume_precise
check_PROGRAMS += hashing \
		  stackhashing \
		  collatehashes
if ENABLE_MPI
check_PROGRAMS += collatehashes_parallel
endif
check_PROGRAMS += collatestacks
if ENABLE_MPI
check_PROGRAMS += collatestacks_parallel_1 \
		  collatestacks_parallel_2_1 \
		  collatestacks_parallel_2_2 \
		  collatestacks_parallel_3
endif
check_PROGRAMS += collatecallprofiles
if ENABLE_MPI
check_PROGRAMS += collatecallprofiles_parallel_1 \
		  collatecallprofiles_parallel_2_1 \
		  collatecallprofiles_parallel_2_2
endif
check_PROGRAMS += calltime_imbalances
if ENABLE_MPI
check_PROGRAMS += calltime_imbalances_parallel_1 \
		  calltime_imbalances_parallel_2_1 \
		  calltime_imbalances_parallel_2_2
endif
if ENABLE_MPI
check_PROGRAMS += collatempiprofiles \
		  collatempiprofiles_parallel_1 \
		  collatempiprofiles_parallel_2_1 \
		  collatempiprofiles_parallel_2_2
endif
check_PROGRAMS += namegroup_collatestacks

check_PROGRAMS += stacktree \
		  stacklist \
		  getstackstring

check_PROGRAMS += profilelist \
		  exclusive_times \
		  total_call_overhead
if ENABLE_MPI
check_PROGRAMS += total_mpi_overhead
endif
if ENABLE_OMP
check_PROGRAMS += total_omp_overhead
endif
check_PROGRAMS += sorting_stacks \
		  sorting_collated_stacks
check_PROGRAMS += print_prof_table \
		  print_name_grouped_prof_table \
		  print_name_grouped_prof_table_max_stack_ids
if ENABLE_MPI
check_PROGRAMS += print_mpi_table \
		  sorting_mpi_table
endif

check_PROGRAMS += threadlist \
		  threadtree

check_PROGRAMS += filenames \
                  signal_handling

sort_int8_SOURCES = $(srcdir)/sort_int8.c \
			 $(srcdir)/bad_rng.c
sort_int8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_int8_LDFLAGS = $(AM_LDFLAGS)

sort_perm_int8_SOURCES = $(srcdir)/sort_perm_int8.c \
			      $(srcdir)/bad_rng.c
sort_perm_int8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_perm_int8_LDFLAGS = $(AM_LDFLAGS)

sort_uint8_SOURCES = $(srcdir)/sort_uint8.c \
			  $(srcdir)/bad_rng.c
sort_uint8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_uint8_LDFLAGS = $(AM_LDFLAGS)

sort_perm_uint8_SOURCES = $(srcdir)/sort_perm_uint8.c \
			       $(srcdir)/bad_rng.c
sort_perm_uint8_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_perm_uint8_LDFLAGS = $(AM_LDFLAGS)

sort_float_SOURCES = $(srcdir)/sort_float.c \
			  $(srcdir)/bad_rng.c
sort_float_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_float_LDFLAGS = $(AM_LDFLAGS)

sort_perm_float_SOURCES = $(srcdir)/sort_perm_float.c \
			       $(srcdir)/bad_rng.c
sort_perm_float_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_perm_float_LDFLAGS = $(AM_LDFLAGS)

sort_uint64_SOURCES = $(srcdir)/sort_uint64.c \
			   $(srcdir)/bad_rng.c
sort_uint64_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_uint64_LDFLAGS = $(AM_LDFLAGS)

sort_longlong_SOURCES = $(srcdir)/sort_longlong.c \
			     $(srcdir)/bad_rng.c
sort_longlong_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_longlong_LDFLAGS = $(AM_LDFLAGS)

sort_double_SOURCES = $(srcdir)/sort_double.c \
			   $(srcdir)/bad_rng.c
sort_double_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/sorting/
sort_double_LDFLAGS = $(AM_LDFLAGS)

search_uint64_SOURCES = $(srcdir)/search_uint64.c
search_uint64_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/
search_uint64_LDFLAGS = $(AM_LDFLAGS)

search_int_SOURCES = $(srcdir)/search_int.c
search_int_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/
search_int_LDFLAGS = $(AM_LDFLAGS)

search_collated_stacks_name_SOURCES = $(srcdir)/search_collated_stacks_name.c
search_collated_stacks_name_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/ -g
search_collated_stacks_name_LDFLAGS = $(AM_LDFLAGS)

chop_chars_SOURCES = $(srcdir)/chop_chars.c
chop_chars_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/
chop_chars_LDFLAGS = $(AM_LDFLAGS)

trim_left_delimiter_SOURCES = $(srcdir)/trim_left_delimiter.c
trim_left_delimiter_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/
trim_left_delimiter_LDFLAGS = $(AM_LDFLAGS)

range_expand_SOURCES = $(srcdir)/range_expand.c
range_expand_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/
range_expand_LDFLAGS = $(AM_LDFLAGS)

levenshtein_SOURCES = $(srcdir)/levenshtein.c
levenshtein_CFLAGS = $(AM_CFLAGS) -I$(top_srcdir)/src/
levenshtein_LDFLAGS = $(AM_LDFLAGS)

symbols_1_SOURCES = $(srcdir)/symbols_1.c \
		    $(srcdir)/protofuncts.c
symbols_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_1_LDFLAGS = $(AM_LDFLAGS)

symbols_2_SOURCES = $(srcdir)/symbols_2.c
symbols_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_2_LDFLAGS = $(AM_LDFLAGS)

symbols_precise_SOURCES = $(srcdir)/symbols_precise.c \
	           $(srcdir)/protofuncts.c
symbols_precise_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
symbols_precise_LDFLAGS = $(AM_LDFLAGS)

pauseresume_precise_SOURCES = $(srcdir)/pauseresume_precise.c
pauseresume_precise_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
pauseresume_precise_LDFLAGS = $(AM_LDFLAGS)

hashing_SOURCES = $(srcdir)/hashing.c
hashing_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
hashing_LDFLAGS = $(AM_LDFLAGS)

stackhashing_SOURCES = $(srcdir)/stackhashing.c \
		       $(srcdir)/dummysymboltable.c 
stackhashing_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stackhashing_LDFLAGS = $(AM_LDFLAGS)

collatehashes_SOURCES = $(srcdir)/collatehashes.c \
		       $(srcdir)/dummy_stacktree.c
collatehashes_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatehashes_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collatehashes_parallel_SOURCES = $(srcdir)/collatehashes_parallel.c \
				 $(srcdir)/dummy_stacktree.c
collatehashes_parallel_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatehashes_parallel_LDFLAGS = $(AM_LDFLAGS)
endif

collatestacks_SOURCES = $(srcdir)/collatestacks.c \
			$(srcdir)/dummy_stacktree.c
collatestacks_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collatestacks_parallel_1_SOURCES = $(srcdir)/collatestacks_parallel_1.c \
			       $(srcdir)/dummy_stacktree.c
collatestacks_parallel_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_1_LDFLAGS = $(AM_LDFLAGS)

collatestacks_parallel_2_1_SOURCES = $(srcdir)/collatestacks_parallel_2_1.c \
				 $(srcdir)/dummy_stacktree.c
collatestacks_parallel_2_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_2_1_LDFLAGS = $(AM_LDFLAGS)

collatestacks_parallel_2_2_SOURCES = $(srcdir)/collatestacks_parallel_2_2.c \
				 $(srcdir)/dummy_stacktree.c
collatestacks_parallel_2_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_2_2_LDFLAGS = $(AM_LDFLAGS)

collatestacks_parallel_3_SOURCES = $(srcdir)/collatestacks_parallel_3.c \
			       $(srcdir)/dummy_stacktree.c
collatestacks_parallel_3_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatestacks_parallel_3_LDFLAGS = $(AM_LDFLAGS)
endif

collatecallprofiles_SOURCES = $(srcdir)/collatecallprofiles.c \
			      $(srcdir)/dummy_stacktree.c
collatecallprofiles_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatecallprofiles_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
collatecallprofiles_parallel_1_SOURCES = $(srcdir)/collatecallprofiles_parallel_1.c \
					 $(srcdir)/dummy_stacktree.c
collatecallprofiles_parallel_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatecallprofiles_parallel_1_LDFLAGS = $(AM_LDFLAGS)

collatecallprofiles_parallel_2_1_SOURCES = $(srcdir)/collatecallprofiles_parallel_2_1.c \
					   $(srcdir)/dummy_stacktree.c
collatecallprofiles_parallel_2_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatecallprofiles_parallel_2_1_LDFLAGS = $(AM_LDFLAGS)

collatecallprofiles_parallel_2_2_SOURCES = $(srcdir)/collatecallprofiles_parallel_2_2.c \
					   $(srcdir)/dummy_stacktree.c
collatecallprofiles_parallel_2_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatecallprofiles_parallel_2_2_LDFLAGS = $(AM_LDFLAGS)
endif

calltime_imbalances_SOURCES = $(srcdir)/calltime_imbalances.c \
			      $(srcdir)/dummy_stacktree.c
calltime_imbalances_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
calltime_imbalances_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
calltime_imbalances_parallel_1_SOURCES = $(srcdir)/calltime_imbalances_parallel_1.c \
					 $(srcdir)/dummy_stacktree.c
calltime_imbalances_parallel_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
calltime_imbalances_parallel_1_LDFLAGS = $(AM_LDFLAGS)

calltime_imbalances_parallel_2_1_SOURCES = $(srcdir)/calltime_imbalances_parallel_2_1.c \
					   $(srcdir)/dummy_stacktree.c
calltime_imbalances_parallel_2_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
calltime_imbalances_parallel_2_1_LDFLAGS = $(AM_LDFLAGS)

calltime_imbalances_parallel_2_2_SOURCES = $(srcdir)/calltime_imbalances_parallel_2_2.c \
					   $(srcdir)/dummy_stacktree.c
calltime_imbalances_parallel_2_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
calltime_imbalances_parallel_2_2_LDFLAGS = $(AM_LDFLAGS)
endif

if ENABLE_MPI
collatempiprofiles_SOURCES = $(srcdir)/collatempiprofiles.c \
			     $(srcdir)/dummysymboltable.c
collatempiprofiles_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatempiprofiles_LDFLAGS = $(AM_LDFLAGS)

collatempiprofiles_parallel_1_SOURCES = $(srcdir)/collatempiprofiles_parallel_1.c \
					$(srcdir)/dummysymboltable.c
collatempiprofiles_parallel_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatempiprofiles_parallel_1_LDFLAGS = $(AM_LDFLAGS)

collatempiprofiles_parallel_2_1_SOURCES = $(srcdir)/collatempiprofiles_parallel_2_1.c \
					  $(srcdir)/dummysymboltable.c
collatempiprofiles_parallel_2_1_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatempiprofiles_parallel_2_1_LDFLAGS = $(AM_LDFLAGS)

collatempiprofiles_parallel_2_2_SOURCES = $(srcdir)/collatempiprofiles_parallel_2_2.c \
					  $(srcdir)/dummysymboltable.c
collatempiprofiles_parallel_2_2_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
collatempiprofiles_parallel_2_2_LDFLAGS = $(AM_LDFLAGS)
endif

namegroup_collatestacks_SOURCES = $(srcdir)/namegroup_collatestacks.c \
				  $(srcdir)/dummy_stacktree.c
namegroup_collatestacks_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
namegroup_collatestacks_LDFLAGS = $(AM_LDFLAGS)

filenames_SOURCES = $(srcdir)/filenames.c
filenames_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
filenames_LDFLAGS = $(AM_LDFLAGS)

stacktree_SOURCES = $(srcdir)/stacktree.c \
		    $(srcdir)/dummysymboltable.c
stacktree_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stacktree_LDFLAGS = $(AM_LDFLAGS)

stacklist_SOURCES = $(srcdir)/stacklist.c \
		    $(srcdir)/dummysymboltable.c
stacklist_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
stacklist_LDFLAGS = $(AM_LDFLAGS)

getstackstring_SOURCES = $(srcdir)/getstackstring.c \
			 $(srcdir)/dummysymboltable.c
getstackstring_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
getstackstring_LDFLAGS = $(AM_LDFLAGS)

profilelist_SOURCES = $(srcdir)/profilelist.c
profilelist_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
profilelist_LDFLAGS = $(AM_LDFLAGS)

exclusive_times_SOURCES = $(srcdir)/exclusive_times.c \
			  $(srcdir)/dummy_stacktree.c
exclusive_times_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
exclusive_times_LDFLAGS = $(AM_LDFLAGS)

total_call_overhead_SOURCES = $(srcdir)/total_call_overhead.c \
			      $(srcdir)/dummy_stacktree.c
total_call_overhead_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
total_call_overhead_LDFLAGS = $(AM_LDFLAGS)

if ENABLE_MPI
total_mpi_overhead_SOURCES = $(srcdir)/total_mpi_overhead.c \
			     $(srcdir)/dummysymboltable.c
total_mpi_overhead_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
total_mpi_overhead_LDFLAGS = $(AM_LDFLAGS)
endif

if ENABLE_OMP
total_omp_overhead_SOURCES = $(srcdir)/total_omp_overhead.c \
			     $(srcdir)/dummysymboltable.c
total_omp_overhead_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
total_omp_overhead_LDFLAGS = $(AM_LDFLAGS)
endif

sorting_stacks_SOURCES = $(srcdir)/sorting_stacks.c \
                         $(srcdir)/dummy_stacktree.c
sorting_stacks_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/src/sorting/
sorting_stacks_LDFLAGS = $(AM_LDFLAGS)

sorting_collated_stacks_SOURCES = $(srcdir)/sorting_collated_stacks.c \
				  $(srcdir)/dummy_stacktree.c
sorting_collated_stacks_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/src/sorting/
sorting_collated_stacks_LDFLAGS = $(AM_LDFLAGS)

print_prof_table_SOURCES = $(srcdir)/print_prof_table.c \
			   $(srcdir)/dummysymboltable.c \
                           $(srcdir)/dummy_stacktree.c
print_prof_table_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
print_prof_table_LDFLAGS = $(AM_LDFLAGS)

print_name_grouped_prof_table_SOURCES = $(srcdir)/print_name_grouped_prof_table.c \
					$(srcdir)/dummysymboltable.c \
					$(srcdir)/dummy_stacktree.c
print_name_grouped_prof_table_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
print_name_grouped_prof_table_LDFLAGS = $(AM_LDFLAGS)

print_name_grouped_prof_table_max_stack_ids_SOURCES = \
      $(srcdir)/print_name_grouped_prof_table.c \
      max_stack_ids_  	$(srcdir)/dummysymboltable.c \
      max_stack_ids_  	$(srcdir)/dummy_stacktree.c
print_name_grouped_prof_table_max_stack_ids_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
print_name_grouped_prof_table_max_stack_ids_LDFLAGS = $(AM_LDFLAGS)

print_mpi_table_SOURCES = $(srcdir)/print_mpi_table.c \
			  $(srcdir)/dummysymboltable.c
print_mpi_table_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
print_mpi_table_LDFLAGS = $(AM_LDFLAGS)

sorting_mpi_table_SOURCES = $(srcdir)/sorting_mpi_table.c \
			    $(srcdir)/dummysymboltable.c
sorting_mpi_table_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src -I$(top_srcdir)/src/sorting/
sorting_mpi_table_LDFLAGS = $(AM_LDFLAGS)

threadlist_SOURCES = $(srcdir)/threadlist.c \
		     $(srcdir)/dummysymboltable.c
threadlist_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
threadlist_LDFLAGS = $(AM_LDFLAGS)

threadtree_SOURCES = $(srcdir)/threadtree.c \
		     $(srcdir)/dummysymboltable.c
threadtree_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src
threadtree_LDFLAGS = $(AM_LDFLAGS)

signal_handling_SOURCES = $(srcdir)/signal_handling.c
signal_handling_CFLAGS = $(AM_CFLAGS) -I$(top_builddir)/src $(FINSTRUMENT_FLAG)
signal_handling_LDFLAGS = $(AM_LDFLAGS) -L$(top_builddir)/src/.libs -lvftrace


dist_check_SCRIPTS =
dist_check_SCRIPTS += sort_int8.sh \
		      sort_perm_int8.sh \
		      sort_uint8.sh \
		      sort_perm_uint8.sh \
		      sort_float.sh \
		      sort_perm_float.sh \
		      sort_uint64.sh \
		      sort_longlong.sh \
		      sort_double.sh
dist_check_SCRIPTS += search_uint64.sh \
		      search_int.sh \
		      search_collated_stacks_name.sh
dist_check_SCRIPTS += chop_chars.sh \
		      trim_left_delimiter.sh \
		      range_expand.sh \
		      levenshtein.sh
dist_check_SCRIPTS += symbols_1.sh \
		      symbols_2.sh \
		      symbols_precise.sh \
		      pauseresume_precise.sh
dist_check_SCRIPTS += hashing.sh \
		      stackhashing.sh \
		      collatehashes.sh
if ENABLE_MPI
dist_check_SCRIPTS += collatehashes_parallel.sh
endif
dist_check_SCRIPTS += collatestacks.sh
if ENABLE_MPI
dist_check_SCRIPTS += collatestacks_parallel_1.sh \
		      collatestacks_parallel_2.sh \
		      collatestacks_parallel_3.sh
endif
dist_check_SCRIPTS += collatecallprofiles.sh
if ENABLE_MPI
dist_check_SCRIPTS += collatecallprofiles_parallel_1.sh \
		      collatecallprofiles_parallel_2.sh
endif

dist_check_SCRIPTS += calltime_imbalances.sh
if ENABLE_MPI
dist_check_SCRIPTS += calltime_imbalances_parallel_1.sh \
		      calltime_imbalances_parallel_2.sh
endif

if ENABLE_MPI
dist_check_SCRIPTS += collatempiprofiles.sh \
		      collatempiprofiles_parallel_1.sh \
		      collatempiprofiles_parallel_2.sh
endif
dist_check_SCRIPTS += namegroup_collatestacks.sh

dist_check_SCRIPTS += stacktree.sh \
		      stacklist.sh \
		      getstackstring.sh
dist_check_SCRIPTS += profilelist.sh \
		      exclusive_times.sh \
		      total_call_overhead.sh
if ENABLE_MPI
dist_check_SCRIPTS += total_mpi_overhead.sh
endif
if ENABLE_OMP
dist_check_SCRIPTS += total_omp_overhead.sh
endif
dist_check_SCRIPTS += sorting_stacks_none.sh \
		      sorting_stacks_calls.sh \
		      sorting_stacks_overhead.sh \
		      sorting_stacks_stackid.sh \
		      sorting_stacks_texcl.sh \
		      sorting_stacks_tincl.sh
dist_check_SCRIPTS += sorting_collated_stacks_none.sh \
		      sorting_collated_stacks_calls.sh \
		      sorting_collated_stacks_overhead.sh \
		      sorting_collated_stacks_stackid.sh \
		      sorting_collated_stacks_texcl.sh \
		      sorting_collated_stacks_tincl.sh
dist_check_SCRIPTS += print_prof_table.sh \
		      print_name_grouped_prof_table.sh \
		      print_name_grouped_prof_table_max_stack_ids.sh
if ENABLE_MPI
dist_check_SCRIPTS += print_mpi_table.sh
dist_check_SCRIPTS += sorting_mpi_table_none.sh \
		      sorting_mpi_table_messages.sh \
		      sorting_mpi_table_sendsize.sh \
		      sorting_mpi_table_recvsize.sh \
		      sorting_mpi_table_sendbw.sh \
		      sorting_mpi_table_recvbw.sh \
		      sorting_mpi_table_commtime.sh \
		      sorting_mpi_table_stackid.sh
endif
dist_check_SCRIPTS += threadlist.sh \
		      threadtree.sh \
                      filenames.sh \
                      signal_handling.sh

TEST_EXTENSIONS = .sh
SH_LOG_COMPILER = $(SHELL)

TESTS = $(dist_check_SCRIPTS)

CLEANFILES = vftrace_self_profile-* \
	     *.json
