#!/usr/bin/env python
import sys

import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np

import read_functions
import progression

def do_prediction():

  if len(sys.argv) > 1:
    all_dicts = []
    for filename in sys.argv[1:]:
      top_5_stack_ids, func_dict = read_functions.create_dictionary(filename)
      all_dicts.append(func_dict) 
      
      print ("top_5 in " + filename + ": ")
      for i in top_5_stack_ids:
        print (func_dict[i])
      print ("**************************************************")
  
    global_dict = read_functions.synchronize_dictionaries(all_dicts) 

# Do the prediction
    x_predict = float(entry_entry.get())
    t_predict = 0
    for this_hash, prog_entry in global_dict.items():
      progression.determine_progression_type(prog_entry, len(sys.argv) - 1)
      if not isinstance(prog_entry.progression_type, progression.prog_undefined):
         t_predict += prog_entry.progression_type.predict(x_predict)
      print (prog_entry)

    result_label["text"] = "Prediction: " + str(t_predict) + "s"

  else:
    print ("Require at least one log file as argument!")



window = tk.Tk()
window.title ("The crystal ball performance predictor")

frame_entry = tk.Frame (master = window)
entry_label = tk.Label (master = frame_entry, text = "Runtime:")
entry_entry = tk.Entry (master = frame_entry, width=5)
entry_button = tk.Button (master = frame_entry, text = "Predict", command = do_prediction)
entry_label.grid (row=0, column=0)
entry_entry.grid (row=0, column=1)
entry_button.grid (row=0, column=2)
frame_entry.grid (row=0, column=0)

result_frame = tk.Frame (master = window)
result_label = tk.Label (master = result_frame, text = "Prediction: ")
result_label.grid(row=0, column=0)
result_frame.grid(row=1, column=0)

all_dicts = []
for filename in sys.argv[1:]:
  top_5_stack_ids, func_dict = read_functions.create_dictionary(filename)
  all_dicts.append(func_dict) 
  
  print ("top_5 in " + filename + ": ")
  for i in top_5_stack_ids:
    print (func_dict[i])
  print ("**************************************************")

global_dict = read_functions.synchronize_dictionaries(all_dicts) 


frame_plot = tk.Frame (master = window)
frame_plot.grid(row=0, column=1)

#fig = Figure(figsize=(5,4), dpi=100) 
fig, ax = plt.subplots()

prog_entry = list(global_dict.values())[0]
print (prog_entry.x)
print (prog_entry.t)
#t = np.arange(0, 3, 0.01)
#fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
axes = []
func_names = []
for i, prog_entry in enumerate(global_dict.values()):
  l, = ax.plot(prog_entry.x, prog_entry.t)
  #ax = fig.add_subplot(111).plot(prog_entry.x, prog_entry.t)
  axes.append(l)
  func_names.append(prog_entry.func_name)
  if i == 5: break

print ("func_names: ", func_names)
ax.legend(axes, func_names)
ax.set_ylabel ("t[s]")

canvas = FigureCanvasTkAgg(fig, master = frame_plot)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, frame_plot)
toolbar.update()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


window.mainloop()
#tk.mainloop()


