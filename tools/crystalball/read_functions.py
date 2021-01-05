import progression

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

def create_dictionary (filename):
  f = open(filename, "r")
  lines = f.readlines()

  overview = vftrace_overview()
  countdown_profile = -1
  countdown_stacks = -1
  functions = []
  function_stacks = {}
  for line in lines:
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
    if "Runtime profile for rank" in line:
      overview.is_truncated = "truncated" in line
      countdown_profile = 3
    elif countdown_profile > 0:
      countdown_profile -= 1
    elif countdown_profile == 0:  
      if not "-----------" in line:
        functions.append(line) 
      else:
        countdown_profile = -1

    if "Function call stack" in line:
       countdown_stacks = 1
    elif countdown_stacks > 0: 
       countdown_stacks -=1
    elif countdown_stacks == 0:
      if not "-----------" in line:
        tmp = line.split()
        check_if_stack_line_is_consistent(tmp[1])
        function_stacks[tmp[0]] = tmp[1]
      else:
        countdown_stacks = -1
    
  
  func_dict = {}
  
  # A dictionary keeps its order, so the elements are automatically sorted in the same way as Vftrace has sorted them. 
  overview.n_recorded_functions = len(functions)
  overview.truncated_at = functions[-1].split()[4]
  
  for function in functions:
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
  global_dict = {}
  for i_dict, dicto in enumerate(dictos):
    for stack_id, fe in dicto.items():
      if fe.hash in global_dict:
        global_dict[fe.hash].append(global_x[i_dict], fe.n_calls, fe.t_excl)
      else:
        global_dict[fe.hash] = progression.progression_entry(fe.function_name, global_x[i_dict], fe.n_calls, fe.t_excl)
  for i, overview in enumerate(overviews):
    if i == 0:
      global_dict["total"] = progression.progression_entry("total", global_x[i], overview.n_recorded_calls, overview.recorded_time)
    else:
      global_dict["total"].append(global_x[i], overview.n_recorded_calls, overview.recorded_time)
  return global_dict




