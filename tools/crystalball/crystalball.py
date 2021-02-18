#!/usr/bin/env python

# Crystalball visualizes Vftrace log files and allows for the extrapolation
# of total runtimes using the progression of the runtimes of the individual functions.
# 
import sys
import os
from collections import OrderedDict

import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter import ttk

import numpy as np

import control_window
import filter_window
import read_functions

def do_prediction():

  global prediction_entry, prediction_result
  if global_dict_created:
    # For each list of runtimes, we fit various models. Out of these, the best fit is chosen.
    for p in global_dict.values():
      p.test_models ()

    # User-defined target value given in the input field.
    x_predict = float(prediction_entry.get())
    t_predict = 0.0
    # Loop over all function entries and add up the predicted time.
    for key, p in global_dict.items():  
      # There are the additional entries for total runtime, sampling overhead and MPI overhead.
      # We add the prediction for the sampling overhead, but not the MPI overhead, as it is already included in the individual MPI functions.
      if key == "total" or key == "mpi_overhead": continue
      t_predict += p.extrapolate_function(x_predict)
    prediction_result["text"] = str("%.2f"%t_predict) + " seconds"

    # Detailed information about the extrapolation of each function is written to this file.
    with open("extrapolate.out", "w") as f:
      for p in global_dict.values():
        f.write(str(p))  
  else:
    prediction_result["text"] = "No input fives given!"

def plot_function(plot_data, x, y, normal_value=None):
  plot_data.set_xdata(x)
  if normal_value is not None:
    plot_data.set_ydata([round(y_val / n_val * 100, 2) for y_val, n_val in zip(y, normal_value)])
  else:
    plot_data.set_ydata(y)
  plot_data.set_linestyle("-")

def update_crystalball(wcontrol, wfilter):
  global global_dict_created
  global global_dict
  global var_calls_checked
  global var_total_checked
  global var_normal_checked
  global var_sampling_checked
  global var_mpi_checked
  global plot_data
  global n_current_plots
  global n_max_plots
  global n_functions_entry
  n_max_plots = int(n_functions_entry.get())
  if not global_dict_created:
    overviews = []
    for f in wcontrol.open_files:
      overview, func_dict = read_functions.create_dictionary(f)
      ttk.Label(scrollable_frame, text = f + ":").pack()
      ttk.Label(scrollable_frame, text = str(overview)).pack()
      ttk.Label(scrollable_frame, text = "-------------------------").pack()
      all_dicts.append(func_dict)
      overviews.append(overview)

    global_dict = read_functions.synchronize_dictionaries(wcontrol.x_values, overviews, all_dicts)
    global_dict_created = True

  axes = []
  func_names = []
  n_previous_plots = n_current_plots
  i_plot = 0
  for i, extr_entry in enumerate(global_dict.values()):
    if wfilter.skip(extr_entry.func_name): continue
    plot_this = True
    if i_plot == n_max_plots:
      ee = global_dict["total"]
      plot_this = var_total_checked.get()
    elif i_plot == n_max_plots + 1:
      ee = global_dict["sampling_overhead"]
      plot_this = var_sampling_checked.get()
    elif i_plot == n_max_plots + 2:
      ee = global_dict["mpi_overhead"]
      plot_this = var_mpi_checked.get()
    elif i_plot > n_max_plots + 2 and i_plot < n_previous_plots:
      plot_this = False
    elif i_plot < n_max_plots:
      ee = extr_entry
    else:
      break

    normal_value = None
    if var_normal_checked.get():
      if var_calls_checked.get():
        normal_value = global_dict["total"].n_calls
      else:
        normal_value = global_dict["total"].t

    if i_plot == len(plot_data):
      p, = ax.plot([],[], "o")
      plot_data.append(p)
    if plot_this:
      n_current_plots += 1
      if var_calls_checked.get():
        plot_function(plot_data[i_plot], ee.x, ee.n_calls, normal_value)
      else:
        plot_function(plot_data[i_plot], ee.x, ee.t, normal_value)
      if var_stackid_checked.get() and ee.stack_id >= 0:
        func_names.append(ee.func_name + " [" + str(ee.stack_id) + "]")
      else:
        func_names.append(ee.func_name)
    else:
      plot_data[i_plot].set_xdata([])
      plot_data[i_plot].set_ydata([])
    i_plot += 1
  ax.relim()
  ax.autoscale_view()
  ax.legend(plot_data, func_names)
  ax.set_xlabel (wcontrol.xlabel)
  if wcontrol.ylabel != '':
    ax.set_ylabel (wcontrol.ylabel)
  elif var_calls_checked.get():
    if var_normal_checked.get():
      ax.set_ylabel("%Calls")
    else:
      ax.set_ylabel ("#Calls")
  else:
    if var_normal_checked.get():
      ax.set_ylabel ("%t")
    else:
      ax.set_ylabel ("t[s]")
  plot_canvas.draw()
  plot_canvas.flush_events()

def update_if_valid():
  if wcontrol.valid: update_crystalball(wcontrol, wfilter)
    
def create_sample():
  wcontrol.open_window()
  if wcontrol.valid:
    update_crystalball(wcontrol, wfilter)

def filter_functions():
  wfilter.open_window(global_dict)
  if wfilter.needs_update: update_crystalball(wcontrol, wfilter) 

def save_plot():
  global fig
  filename = filedialog.asksaveasfilename(initialdir=os.getcwd(), title = "Save as",
                                          filetypes = (("jpeg", "*.jpg"),("png", "*.png"),("all files", "*")))
  fig.savefig(filename)

# Create the main window
window = tk.Tk()
window.protocol ("WM_DELETE_WINDOW", window.destroy)
window.wm_title ("The crystal ball performance predictor")

global_dict_created = False
# For each input file, a dictionary is created. Its keys are the stack IDs and its values
# are objects containing the other information found in a line of the Vftrace profile
# (e.g. n_calls, exclusive time, the function name). all_dicts is a list of these dictionaries.
all_dicts = []
# All "local" dictionaries are merged into a global dictionary. The keys are hashes of the
# function stack strings, the values are lists of objects containing runtime data (extrapolation entries).
# On these lists, extrapolations can be made.
# We use an OrderedDict, where the items keep their original position.
# More importantly, this dictionary type allows for sorting. This is required
# to sort the global dictionary with respect to the total time spent on a function entry.
global_dict = OrderedDict()

# These windows open when the corresponding button is clicked. We already create empty
# objects for them here.
wcontrol = control_window.control_window(window)
wfilter = filter_window.filter_window(window)

# We create the menubar. It consists of the tab "Action", which allows to control the in- and output,
# and the tab "Display", which controls details of the plot.
menubar = tk.Menu(window)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Create Sample", command = create_sample)
file_menu.add_command(label="Save plot", command = save_plot)
file_menu.add_separator()
file_menu.add_command(label="Exit", command = window.quit)
menubar.add_cascade(label="Actions", menu=file_menu)

# The "Display" tab contains check boxes for plot options. These require global variables which
# are defined below. Each option is disabled at startup (value=0), except for var_stackid_checked,
# which triggers the display of stack ids in the plot legend.
var_calls_checked = tk.IntVar(value=0)
var_total_checked = tk.IntVar(value=0)
var_normal_checked = tk.IntVar(value=0)
var_sampling_checked = tk.IntVar(value=0)
var_mpi_checked = tk.IntVar(value=0)
var_stackid_checked = tk.IntVar(value=1)

display_menu = tk.Menu(menubar, tearoff=0)
# All buttons update the plot if clicked.
display_menu.add_checkbutton(label="Show n_calls", variable = var_calls_checked, onvalue = 1, offvalue = 0, command = update_if_valid)
display_menu.add_checkbutton(label="Show total", variable = var_total_checked, onvalue = 1, offvalue = 0, command = update_if_valid)
display_menu.add_checkbutton(label="Show sampling overhead", variable = var_sampling_checked, onvalue = 1, offvalue = 0, command = update_if_valid)
display_menu.add_checkbutton(label="Show MPI overhead", variable = var_mpi_checked, onvalue = 1, offvalue = 0, command = update_if_valid)
display_menu.add_checkbutton(label="Normalize", variable = var_normal_checked, onvalue = 1, offvalue = 0, command = update_if_valid)
display_menu.add_checkbutton(label="Show stack IDs", variable = var_stackid_checked, onvalue = 1, offvalue = 0, command = update_if_valid)                  
display_menu.add_separator()
# Open the window which allows to filter functions with regular expressions.
display_menu.add_command(label="Filter", command = filter_functions)

menubar.add_cascade(label="Display", menu=display_menu)
window.config(menu=menubar)

# The left half of the window is reserved for a list of the open files with the corresponding overview info.
# It has a scrollbar, for which the below container is required.
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

# The right half of the window contains the plot, as well as a few buttons.
frame_plot = tk.Frame (master = window)
frame_plot.grid(row=0, column=1)

fig, ax = plt.subplots()
plot_data = []
n_current_plots = 0
# The maximum number of plots to be shown. A global variable, whose value is determined by the niput of n_functions_entry.
n_max_plots = 5

plot_canvas = FigureCanvasTkAgg(fig, master = frame_plot)
plot_canvas.draw()
plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

checkbox_frame = tk.Frame(frame_plot)
# Input the number of functions you wish to display
n_functions_button = tk.Button (checkbox_frame, text = "Set n_functions: ",
                                command = update_if_valid).grid(row=0, column=0)
n_functions_entry = tk.Entry (checkbox_frame, width=2)
n_functions_entry.insert(tk.END, n_max_plots)
n_functions_entry.grid(row=0, column=1)
checkbox_frame.pack()
# Do an extrapolation
prediction_frame = tk.Frame(frame_plot)
prediction_button = tk.Button (prediction_frame, text = "Extrapolate for x = ", command = do_prediction).grid(row=0, column=0)
prediction_entry = tk.Entry (prediction_frame, width=3)
prediction_entry.grid(row=0, column=1) # If gridded above, prediction_entry will be None-type...
prediction_result = tk.Label (prediction_frame, text = "")
prediction_result.grid(row=0, column=2)
prediction_frame.pack()



window.mainloop()

