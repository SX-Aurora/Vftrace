from collections import OrderedDict

import extrapolate

# The data given in the Vftrace logfile header.
class vftrace_overview:
  def __init__(self):
    self.mpi_size = 0
    self.total_runtime = 0.0
    self.estimated_application_runtime = 0.0
    self.estimated_overhead = 0.0
    self.sampling_overhead = 0.0
    self.mpi_overhead = 0.0
    self.recorded_time = 0.0
    self.n_recorded_functions = 0
    self.n_recorded_calls = 0
    self.profile_truncated = False
    # Cumulative time where the profile is truncated.
    self.truncated_at = 0.0

  def __str__(self):
    ratio_overhead = self.estimated_overhead / self.total_runtime * 100
    ratio_sampling = self.sampling_overhead / self.total_runtime * 100
    ratio_mpi = self.mpi_overhead / self.total_runtime * 100
    s =  "MPI size: " + str(self.mpi_size) + "\n" + \
         "Total runtime: " + str(self.total_runtime) + "s\n" + \
         "Est. application time: " + str(self.estimated_application_runtime) + "s\n" + \
         "Est. Overhead: " + str(self.estimated_overhead) + "s (" + str("%.2f"%ratio_overhead) + "%)\n" + \
         "   Sampling: " + str(self.sampling_overhead) + "s (" + str("%.2f"%ratio_sampling) + "%)\n" + \
         "   MPI: " + str(self.mpi_overhead) + "s (" + str("%.2f"%ratio_mpi) + "%)\n" + \
         "Nr. of recorded functions: " + str(self.n_recorded_functions) + "\n" + \
         "Nr. of recorded calls: " + str(self.n_recorded_calls) + "\n" + \
         "Recorded time: " + str("%.2f"%self.recorded_time) + "s"
    if self.is_truncated:
      s += " (truncated at " + str(self.truncated_at) + "%)"
    return s
 
    
  def __repr__(self):
    return self.__str__()

# A line in the profile table plus the stack string corresponding to the stack id.
class function_entry:
  def __init__(self, n_calls, t_excl, t_incl, percent_abs, percent_cum, function_name, caller_name, stack_string):
    self.n_calls = n_calls
    self.t_excl = t_excl
    self.t_incl = t_incl
    self.percent_abs = percent_abs
    self.percent_cum = percent_cum
    self.function_name = function_name
    self.caller_name = caller_name
    self.hash = abs(hash(stack_string))

  def __str__(self):
    return " n_calls: " + str(self.n_calls) + " t_excl: " + str(self.t_excl) +\
           " t_incl: " + str(self.t_incl) + " %abs: " + str(self.percent_abs) + \
           " %cum: " + str(self.percent_cum) + " function: " + str(self.function_name) +\
           " caller: " + str(self.caller_name) + " stack hash: " + str(self.hash)

  def __repr__(self):
    return self.__str__()

# For a reliable performance prediction, all stacks should be fully resolved.
# This means, that there should be no stand-alone function names in each stack line,
# except for "init". Moreover, each stack line has to end with "init".
# If these are not the case, we issue a warning.
def check_if_stack_line_is_consistent (stack_line):
  funcs = stack_line.split("<") # Individual functions are seperated by "<"
  if len(funcs) == 1 and funcs[0] != "init":
    print ("Warning: There is a function without stack trace: ", funcs[0], "No predictions can be made for it.")
  elif funcs[-1] != "init":
    print ("Warning: There is a stack which does not end in init")

# Read the logfile, go through all the lines and fill the data structures.
def create_dictionary (filename):
  f = open(filename, "r")
  lines = f.readlines()

  overview = vftrace_overview()
  countdown_profile = -1
  countdown_stacks = -1
  functions = []
  function_stacks = {}
  for line in lines:
    # First check for the header entries
    if "MPI size" in line:
      overview.mpi_size = int(line.split()[2])
    if "Total runtime" in line:
      overview.total_runtime = float(line.split()[2])
    if "Application time" in line:
      overview.estimated_application_runtime = float(line.split()[2])
    if "Overhead" in line:
      overview.estimated_overhead = float(line.split()[1])
    if "Sampling overhead" in line:
      overview.sampling_overhead = float(line.split()[2])
    if "MPI overhead" in line:
      overview.mpi_overhead = float(line.split()[2])
    # Indicates the start of the profile table. There are three more redundant lines after that (thus countdown = 3).
    if "Runtime profile for rank" in line:
      overview.is_truncated = "truncated" in line
      countdown_profile = 3
    elif countdown_profile > 0:
      countdown_profile -= 1 # Redundant line
    elif countdown_profile == 0:  
      # A string of dashes indicates the end of the profile table. If it is encountered, we reset the countdown.
      if not "-----------" in line:
        functions.append(line) 
      else:
        countdown_profile = -1

    # Indicates the start of the stack table. There are three redundant lines after that (countdown = 1).
    if "Global call stack" in line:
       countdown_stacks = 3
    elif countdown_stacks > 0: 
       countdown_stacks -=1
    elif countdown_stacks == 0:
      # A string of dashes indicates the end of the stack table.
      if not "-----------" in line:
        tmp = line.split()
        check_if_stack_line_is_consistent(tmp[1])
        function_stacks[tmp[0]] = tmp[1]
      else:
        countdown_stacks = -1
    
  overview.n_recorded_functions = len(functions)
  # The cumulative time the profile is truncated at. It's the %cum entry of the last function.
  overview.truncated_at = functions[-1].split()[4]
  
  # We use an OrderedDict, where the items keep their original position.
  # More importantly, this dictionary type allows for sorting. This is required
  # to sort the global dictionary with respect to the total time spent on a function entry.
  func_dict = OrderedDict()
  
  # Process each function line
  # We do not deal with optional additional entries yet, such as overhead or hardware counter.
  for i, function in enumerate(functions):
    tmp = function.split()
    n_calls = tmp[0]
    overview.n_recorded_calls += int(n_calls)
    t_excl = tmp[1]
    overview.recorded_time += float(t_excl)
    t_incl = tmp[2]
    percent_abs = tmp[3]
    percent_cum = tmp[4]
    function_name = tmp[5]
    caller_name = tmp[6]
    stack_id = tmp[7]
    func_dict[stack_id] = function_entry (n_calls, t_excl, t_incl, percent_abs, percent_cum,
			                  function_name, caller_name, function_stacks[stack_id])
  
  return overview, func_dict


def synchronize_dictionaries (global_x, overviews, dictos):
  global_dict = OrderedDict()
  for i_dict, dicto in enumerate(dictos):
    for stack_id, fe in dicto.items():
      # Check if the given hash value is already in the dictionary
      if fe.hash in global_dict:
        global_dict[fe.hash].append(global_x[i_dict], fe.n_calls, fe.t_excl)
      else:
        global_dict[fe.hash] = extrapolate.extrapolation_entry(fe.function_name, global_x[i_dict], fe.n_calls, fe.t_excl, stack_id = int(stack_id))
  # Sort by the total time spent in each extrapolation entry.
  global_dict = OrderedDict(sorted(global_dict.items(), key = lambda x: x[1].total_time, reverse = True))
  for i, overview in enumerate(overviews):
    # There are three additional entries: Total runtime, sampling overhead and MPI overhead.
    if i == 0:
      global_dict["total"] = extrapolate.extrapolation_entry("total", global_x[i], overview.n_recorded_calls, overview.recorded_time)
      global_dict["sampling_overhead"] = extrapolate.extrapolation_entry("sampling_overhead", global_x[i], overview.n_recorded_calls, overview.sampling_overhead)
      global_dict["mpi_overhead"] = extrapolate.extrapolation_entry("mpi_overhead", global_x[i], 0, overview.mpi_overhead)
    else:
      global_dict["total"].append(global_x[i], overview.n_recorded_calls, overview.recorded_time)
      global_dict["sampling_overhead"].append(global_x[i], overview.n_recorded_calls, overview.sampling_overhead)
      global_dict["mpi_overhead"].append(global_x[i], 0, overview.mpi_overhead)

  return global_dict




