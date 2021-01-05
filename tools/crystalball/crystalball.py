#!/usr/bin/env python
import sys

import tkinter as tk
from tkinter import filedialog
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
      func_dict, overview = read_functions.create_dictionary(filename)
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

def update_crystalball(wcontrol):
  global global_dict_created
  global global_dict
  global n_calls_checked
  print  ("Update crystalball: ", n_calls_checked.get())
  if not global_dict_created:
    for f in wcontrol.open_files:
      overview, func_dict = read_functions.create_dictionary(f)
      ttk.Label(scrollable_frame, text = f + ":").pack()
      ttk.Label(scrollable_frame, text = str(overview)).pack()
      ttk.Label(scrollable_frame, text = "-------------------------").pack()
      all_dicts.append(func_dict)

    global_dict = read_functions.synchronize_dictionaries(wcontrol.x_values, all_dicts)
    global_dict_created = True

  axes = []
  func_names = []
  for i, prog_entry in enumerate(global_dict.values()):
    print ("HUHU: ", n_calls_checked.get())
    if n_calls_checked.get() == 1:
      l, = ax.plot(prog_entry.x, prog_entry.n_calls)
      axes.append(l)
    else:
      l, = ax.plot(prog_entry.x, prog_entry.t)
      axes.append(l)
    func_names.append(prog_entry.func_name)
    if i == 5: break
  ax.legend(axes, func_names)
  ax.set_xlabel (wcontrol.xlabel)
  if wcontrol.ylabel != '':
    ax.set_ylabel (wcnotrol.ylabel)
  elif n_calls_checked.get() == 1:
    ax.set_ylabel ("#Calls")
  else:
    ax.set_ylabel ("t[s]")
  plot_canvas.draw()

def switch_to_ncalls():
  if wcontrol.valid: update_crystalball(wcontrol)
    
def create_sample():
  wcontrol.open_window()
  if wcontrol.valid:
    update_crystalball(wcontrol)

window = tk.Tk()
window.protocol ("WM_DELETE_WINDOW", window.destroy)
window.wm_title ("The crystal ball performance predictor")

global_dict_created = False
global_dict = {}

wcontrol = control_window.control_window(window)
menubar = tk.Menu(window)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Create Sample", command=create_sample)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=file_menu)
window.config(menu=menubar)

list_frame = tk.Frame (master = window)
list_header = tk.Label (list_frame, text = "Open files: ")
list_header.pack()

container = ttk.Frame(list_frame)
scroll_canvas = tk.Canvas(container)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=scroll_canvas.yview)
scrollable_frame = ttk.Frame(scroll_canvas)

scrollable_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
scroll_canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
scroll_canvas.configure(yscrollcommand=scrollbar.set)
container.pack()
scroll_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

list_frame.grid(row=0, column=0)

all_dicts = []
global_dict = {}

frame_plot = tk.Frame (master = window)
frame_plot.grid(row=0, column=1)

fig, ax = plt.subplots()
axes = []
func_names = []


plot_canvas = FigureCanvasTkAgg(fig, master = frame_plot)
#plot_canvas = FigureCanvasTk(fig, master = frame_plot)
plot_canvas.draw()
plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

#toolbar = NavigationToolbar2Tk(plot_canvas, frame_plot)
#toolbar.update()
plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
checkbox_frame = tk.Frame(frame_plot)
#check_calls_label = tk.Label(checkbox_frame, text = "Show n_calls")
#check_calls_label.grid(row=0, column=0)
n_calls_checked = tk.IntVar()
check_calls_box = tk.Checkbutton(checkbox_frame, text = "Show n_calls",
                                 variable = n_calls_checked, onvalue = 1, offvalue = 0,
                                 command = switch_to_ncalls).grid(row=0, column=1)
checkbox_frame.pack()



window.mainloop()

