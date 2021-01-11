#!/usr/bin/env python

# Crystalball visualizes Vftrace log files and allows for the extrapolation
# of total runtimes using the progression of the runtimes of the individual functions.
# 
import sys
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
    for p in global_dict.values():
      p.test_models ()

    x_predict = float(prediction_entry.get())
    t_predict = 0.0
    for key, p in global_dict.items():  
      if key == "total" or key == "mpi_overhead": continue
      t_predict += p.extrapolate_function(x_predict)
    prediction_result["text"] = str("%.2f"%t_predict) + " seconds"

    with open("extrapolate.out", "w") as f:
      for p in global_dict.values():
        f.write(str(p))  

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
    ax.set_ylabel (wcnotrol.ylabel)
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

def switch_button():
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
  filename = filedialog.asksaveasfilename(initialdir="/home/christian/Vftrace/tool/crystalball", title = "Save as",
                                          filetypes = (("jpeg", "*.jpg"),("png", "*.png"),("all files", "*")))
  fig.savefig(filename)

window = tk.Tk()
window.protocol ("WM_DELETE_WINDOW", window.destroy)
window.wm_title ("The crystal ball performance predictor")

global_dict_created = False
global_dict = OrderedDict()

var_calls_checked = tk.IntVar(value=0)
var_total_checked = tk.IntVar(value=0)
var_normal_checked = tk.IntVar(value=0)
var_sampling_checked = tk.IntVar(value=0)
var_mpi_checked = tk.IntVar(value=0)
var_stackid_checked = tk.IntVar(value=1)

wcontrol = control_window.control_window(window)
wfilter = filter_window.filter_window(window)
menubar = tk.Menu(window)
file_menu = tk.Menu(menubar, tearoff=0)
file_menu.add_command(label="Create Sample", command = create_sample)
file_menu.add_command(label="Save plot", command = save_plot)
file_menu.add_separator()
file_menu.add_command(label="Exit", command = window.quit)
menubar.add_cascade(label="Actions", menu=file_menu)
menubar.add_command (label="Filter", command = filter_functions)

display_menu = tk.Menu(menubar, tearoff=0)
display_menu.add_checkbutton(label="Show n_calls", variable = var_calls_checked, onvalue = 1, offvalue = 0, command = switch_button)
display_menu.add_checkbutton(label="Show total", variable = var_total_checked, onvalue = 1, offvalue = 0, command = switch_button)
display_menu.add_checkbutton(label="Show sampling overhead", variable = var_sampling_checked, onvalue = 1, offvalue = 0, command = switch_button)
display_menu.add_checkbutton(label="Show MPI overhead", variable = var_mpi_checked, onvalue = 1, offvalue = 0, command = switch_button)
display_menu.add_checkbutton(label="Normalize", variable = var_normal_checked, onvalue = 1, offvalue = 0, command = switch_button)
display_menu.add_checkbutton(label="Show stack IDs", variable = var_stackid_checked, onvalue = 1, offvalue = 0, command = switch_button)                  


menubar.add_cascade(label="Display", menu=display_menu)
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

frame_plot = tk.Frame (master = window)
frame_plot.grid(row=0, column=1)

fig, ax = plt.subplots()
plot_data = []
n_current_plots = 0
n_max_plots = 5

axes = []
func_names = []


plot_canvas = FigureCanvasTkAgg(fig, master = frame_plot)
plot_canvas.draw()
plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

checkbox_frame = tk.Frame(frame_plot)
n_functions_button = tk.Button (checkbox_frame, text = "Set n_functions: ",
                                command = switch_button).grid(row=0, column=0)
n_functions_entry = tk.Entry (checkbox_frame, width=2)
n_functions_entry.insert(tk.END, n_max_plots)
n_functions_entry.grid(row=0, column=1)
checkbox_frame.pack()
prediction_frame = tk.Frame(frame_plot)
prediction_button = tk.Button (prediction_frame, text = "Extrapolate for x = ", command = do_prediction).grid(row=0, column=0)
prediction_entry = tk.Entry (prediction_frame, width=3)
prediction_entry.grid(row=0, column=1) # If gridded above, prediction_entry will be None-type...
prediction_result = tk.Label (prediction_frame, text = "")
prediction_result.grid(row=0, column=2)
prediction_frame.pack()



window.mainloop()

