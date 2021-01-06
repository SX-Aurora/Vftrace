from scipy import optimize 
import numpy as np
import scipy

def test_linear(x, a, b):
  #print ("type(a): ", type(a))
  #print ("type(b): ", type(b))
  #print ("type(x): ", type(x))
  return a * x + b

def test_constant(x, a):
  return a

def test_amdahl(x, a, b):
  return a / x + b 

def test_log(x, a, b):
  return a * x * np.log(x) + b

class progression_entry:

  def __init__(self, func_name, this_x, this_n_calls, this_t):
    self.func_name = func_name
    self.x = [float(this_x)]
    self.n_calls = [int(this_n_calls)]
    self.t = [float(this_t)] 
    self.total_time = float(this_t)
    self.extrapolate_function = None
    self.a = None
    self.b = None

  def __str__(self):
    s = ""
    # Use enumerate
    for i in range(len(self.x)):
      s += "(" + str(self.x[i]) + "," + str(self.n_calls[i]) + "," + str(self.t[i]) +  "," + str(float(self.t[i]) / float(self.n_calls[i])) + ")"
      if i != len(self.x) - 1:
        s += " -> "
       
    #if self.progression_type != None:
    #  s += "[" + str(self.progression_type) + "]"
    return s

  def append(self, this_x, this_n_calls, this_t):
    self.x.append(float(this_x))
    self.n_calls.append(int(this_n_calls))
    self.t.append(float(this_t))
    self.total_time += float(this_t)

  def extrapolate_constant(self, x):
    return self.a

  def extrapolate_linear(self, x):
    return self.a * x + self.b

  def extrapolate_log (self, x):
    return self.a * x * np.log(x) + self.b

  def extrapolate_amdahl (self, x):
    return self.a / x + self.b

  def test_models(self):
    #print ("Test: ", self.func_name)
    extrapolate_functions = [self.extrapolate_linear, self.extrapolate_constant, self.extrapolate_amdahl, self.extrapolate_log]
    residuals = [0.0 for i in range(4)]
    a = [0.0 for i in range(4)]
    b = [0.0 for i in range(4)]
    if len(self.x) > 1:
      popt, _ = scipy.optimize.curve_fit(test_linear, self.x, self.t)
      a[0], b[0] = popt
      for x, y in zip(self.x, self.t):
        residuals[0] += (test_linear(x, a[0], b[0]) - y)**2
    popt, _ = scipy.optimize.curve_fit(test_constant, self.x, self.t)
    a[1] = popt
    for x, y in zip(self.x, self.t):
      residuals[1] += (test_constant(x, a[1]) - y)**2
    if len(self.x) > 1:
      popt, _ = scipy.optimize.curve_fit(test_amdahl, self.x, self.t)
      a[2], b[2] = popt
      for x, y in zip(self.x, self.t):
        residuals[2] += (test_amdahl(x, a[2], b[2]) - y)**2
      popt, _ = scipy.optimize.curve_fit(test_log, self.x, self.t)
      a[3], b[3] = popt
      for x, y in zip(self.x, self.t):
        residuals[3] += (test_log(x, a[3], b[3]) - y)**2
    min_res = residuals.index(min(residuals))
    self.extrapolate_function = extrapolate_functions[min_res]
    self.a = a[min_res]
    self.b = b[min_res]
     

class prog_linear:
  def __init__(self, m, n):
    self.m = m
    self.n = n
    pass

  def __str__(self):
    return "PROG: Linear (" + str(self.m) + "," + str(self.n) + ")"

  def predict(self, x):
    return self.m * x + self.n

class prog_constant:
  def __init__(self, value):
    self.value = value
    pass

  def __str__(self):
    return "PROG: Constant (" + str(self.value) + ")"

  def predict(self, x):
    return self.value

class prog_undefined:
  def __init__(self):
    pass

  def __str__(self):
    return "PROG: Undefined"

def determine_progression_type (prog_entry, n_require):
  n = len(prog_entry.x)
  if n == n_require:
    delta_x = [prog_entry.x[i+1] - prog_entry.x[i] for i in range(n-1)]
    delta_calls = [prog_entry.n_calls[i+1] - prog_entry.n_calls[i] for i in range(n-1)] 
    if all(tmp == 0 for tmp in delta_calls):
      prog_entry.progression_type = prog_constant(prog_entry.t[0])
    else:
      m = (prog_entry.t[-1] - prog_entry.t[0]) / (prog_entry.x[-1] - prog_entry.x[0])
      n = prog_entry.t[-1] - m * prog_entry.x[-1]
      prog_entry.progression_type = prog_linear(m, n)
  else:
    prog_entry.progression_type = prog_undefined()
  params, params_covariance = optimize.curve_fit(test_linear, prog_entry.x, prog_entry.t)
  if np.all(params_covariance.all != None):
    pass
    #print ("progtype: ", prog_entry.progression_type, ", stderr: ", np.sqrt(np.diag(params_covariance))) 

    
