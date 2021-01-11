from scipy import optimize 
import numpy as np
import scipy

def test_linear(x, a, b):
  return a * x + b

def test_constant(x, a):
  return a

def test_amdahl(x, a, b):
  return a / x + b 

def test_log(x, a, b):
  return a * x * np.log(x) + b

class extrapolation_entry:

  def __init__(self, func_name, this_x, this_n_calls, this_t, stack_id=-1):
    self.func_name = func_name
    self.stack_id = stack_id
    self.x = [float(this_x)]
    self.n_calls = [int(this_n_calls)]
    self.t = [float(this_t)] 
    self.total_time = float(this_t)
    self.extrapolate_type = ""
    self.extrapolate_function = None
    self.a = None
    self.b = None
    self.extrapolation = -1.0

  def __str__(self):
    s = ""
    s = self.func_name + ": "
    for i in range(len(self.x)):
      s += "(" + str(self.x[i]) + "," + str(self.n_calls[i]) + "," + str(self.t[i]) + ")"
      if i != len(self.x) - 1:
        s += " -> "
       
    if self.extrapolate_type != "":
      s += "[" + str(self.extrapolate_type) + "]"
    if self.extrapolation >= 0.0: s += " - Extra: " + str(self.extrapolation)
    s += "\n"
    return s

  def append(self, this_x, this_n_calls, this_t):
    self.x.append(float(this_x))
    self.n_calls.append(int(this_n_calls))
    self.t.append(float(this_t))
    self.total_time += float(this_t)

  def extrapolate_constant(self, x):
    self.extrapolation = self.a
    return self.extrapolation

  def extrapolate_linear(self, x):
    self.extrapolation = self.a * x + self.b
    return self.extrapolation

  def extrapolate_log (self, x):
    self.extrapolation = self.a * x * np.log(x) + self.b
    return self.extrapolation

  def extrapolate_amdahl (self, x):
    self.extrapolation =  self.a / x + self.b
    return self.extrapolation

  def test_models(self):
    extrapolate_functions = [self.extrapolate_linear, self.extrapolate_constant, self.extrapolate_amdahl, self.extrapolate_log]
    extrapolate_types = ["linear", "constant", "amdahl", "log"]
    residuals = [0.0 for i in range(4)]
    a = [0.0 for i in range(4)]
    b = [0.0 for i in range(4)]
    if len(self.x) > 1:
      popt, _ = scipy.optimize.curve_fit(test_linear, self.x, self.t)
      a[0], b[0] = popt
      for x, y in zip(self.x, self.t):
        residuals[0] += (test_linear(x, a[0], b[0]) - y)**2
    popt, _ = scipy.optimize.curve_fit(test_constant, self.x, self.t)
    a[1] = popt[0] # Why?
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
    self.extrapolate_type = extrapolate_types[min_res]
    self.extrapolate_function = extrapolate_functions[min_res]
    self.a = a[min_res]
    self.b = b[min_res]
     

class extrapolate_linear:
  def __init__(self, m, n):
    self.m = m
    self.n = n
    pass

  def __str__(self):
    return "PROG: Linear (" + str(self.m) + "," + str(self.n) + ")"

  def predict(self, x):
    return self.m * x + self.n

class extrapolate_constant:
  def __init__(self, value):
    self.value = value
    pass

  def __str__(self):
    return "PROG: Constant (" + str(self.value) + ")"

  def predict(self, x):
    return self.value

class extrapolate_undefined:
  def __init__(self):
    pass

  def __str__(self):
    return "PROG: Undefined"

