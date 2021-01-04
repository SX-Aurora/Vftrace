#!/usr/bin/env python
import sys

import tkinter as tk
from tkinter import filedialog
#from tkinter import splitlist
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter import ttk

import numpy as np

import control_window
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

#open_files = []

def update_crystalball(open_files):
  print ("Update crystal ball: ", open_files)
  #for i_row, f in enumerate(open_files):
  #  print ("Append: ", f)
  #  list_names.append(tk.Label(list_frame, text = f).grid(row=i_row + 1, column=0))
  for f in open_files:
    ttk.Label(scrollable_frame, text=f + "\nHUHU").pack()
  for f in open_files:
    top5, func_dict = read_functions.create_dictionary(f)
    all_dicts.append(func_dict)
  global_dict = read_functions.synchronize_dictionaries(all_dicts)
  axes = []
  func_names = []
  for i, prog_entry in enumerate(global_dict.values()):
    l, = ax.plot(prog_entry.x, prog_entry.t)
    axes.append(l)
    func_names.append(prog_entry.func_name)
    if i == 5: break
  canvas.draw()
    
def create_sample():
  wcontrol.open_window()
  print ("HUHU: ", wcontrol.open_files, wcontrol.valid)
  if wcontrol.valid:
    print ("wcontrol.open_files:", wcontrol.open_files)
    open_files = wcontrol.open_files.copy()
    print ("open_files:", open_files)
    update_crystalball(open_files)

window = tk.Tk()
window.protocol ("WM_DELETE_WINDOW", window.destroy)
window.wm_title ("The crystal ball performance predictor")

wcontrol = control_window.control_window(window)
menubar = tk.Menu(window)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Create Sample", command=create_sample)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=file_menu)
window.config(menu=menubar)

#list_names = []
list_frame = tk.Frame (master = window)
list_header = tk.Label (list_frame, text = "Open files: ")
#list_header.grid(row=0, column=0)
list_header.pack()

container = ttk.Frame(list_frame)
canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
container.pack()
canvas.pack(side="left", fill="both", expand=True)
#canvas.grid(row=1, column=0)
scrollbar.pack(side="right", fill="y")
#scrollbar.grid(row=1, column=1)


#list_label = tk.Label (master = list_frame, text = "Loaded files: ")
#list_label.grid(row=0, column=0)
list_frame.grid(row=0, column=0)

all_dicts = []
#for filename in sys.argv[1:]:
#  top_5_stack_ids, func_dict = read_functions.create_dictionary(filename)
#  all_dicts.append(func_dict) 
#  
#  print ("top_5 in " + filename + ": ")
#  for i in top_5_stack_ids:
#    print (func_dict[i])
#  print ("**************************************************")

global_dict = {}
#global_dict = read_functions.synchronize_dictionaries(all_dicts) 

frame_plot = tk.Frame (master = window)
frame_plot.grid(row=0, column=1)

#fig = Figure(figsize=(5,4), dpi=100) 
fig, ax = plt.subplots()

#prog_entry = list(global_dict.values())[0]
#print (prog_entry.x)
#print (prog_entry.t)
#t = np.arange(0, 3, 0.01)
#fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
axes = []
func_names = []
# for i, prog_entry in enumerate(global_dict.values()):
#   l, = ax.plot(prog_entry.x, prog_entry.t)
#   #ax = fig.add_subplot(111).plot(prog_entry.x, prog_entry.t)
#   axes.append(l)
#   func_names.append(prog_entry.func_name)
#   if i == 5: break

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


