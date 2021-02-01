import os

import tkinter as tk
from tkinter import filedialog

# The window which allows to choose the input files.
class control_window:

  def __init__(self, root):
    self.open_files = []
    self.x_values = []
    self.root = root
    self.window = None
    self.open_files_frame = None
    self.open_files_header = None
    self.open_files_button = None
    self.open_files_list = []
    self.valid = False
  
  def update(self):
    for i_row, f in enumerate(self.open_files):
      self.open_files_list.append(tk.Label(self.open_files_frame, text = f).grid(row = i_row + 1, column=0))

  def open_file_dialog(self): 
    for f in list(filedialog.askopenfilenames(initialdir=os.getcwd(), title = "Choose log files")):
      self.open_files.append(f)
    self.update()

  def open_window(self):
    self.window = tk.Toplevel (master = self.root)
    self.window.protocol ("WM_DELETE_WINDOW", self.exit)
    self.open_files_frame = tk.Frame (master = self.window)
    self.open_files_header = tk.Label (master = self.open_files_frame, text = "Open Vftrace log files: ")
    self.open_files_button = tk.Button (master = self.open_files_frame, text = "Choose", command = self.open_file_dialog)
    self.open_files_header.grid (row=0, column=0)
    self.open_files_button.grid (row=0, column=1)
    self.open_files_frame.grid (row=0, column=0)

    self.entry_x_frame = tk.Frame (master = self.window)
    self.entry_x_header = tk.Label (master = self.entry_x_frame, text = "x-values: ")
    self.entry_x = tk.Entry (master = self.entry_x_frame, width=20)
    self.entry_x_header.grid(row=0, column=0)
    self.entry_x.grid(row=0, column=1)
    self.entry_x_frame.grid(row=1, column=0)

    self.entry_label_frame = tk.Frame (master = self.window)
    self.entry_xlabel_header = tk.Label (master = self.entry_label_frame, text = "x label: ")
    self.entry_xlabel = tk.Entry (master = self.entry_label_frame, width=10)
    self.entry_ylabel_header = tk.Label (master = self.entry_label_frame, text = "y label: ")
    self.entry_ylabel = tk.Entry (master = self.entry_label_frame, width=10)
    self.entry_xlabel_header.grid(row=0, column=0)
    self.entry_xlabel.grid(row=0, column=1)
    self.entry_ylabel_header.grid(row=0, column=2)
    self.entry_ylabel.grid(row=0, column=3)
    self.entry_label_frame.grid(row=2, column=0)

    self.validation_message = tk.Label (self.window)
    self.validation_message.grid(row=3, column=0)
    
    self.okay_button = tk.Button (master = self.window, text = "Okay", command = self.accept)
    self.cancel_button = tk.Button (master = self.window, text = "Cancel", command = self.exit)
    self.okay_button.grid(row=4, column=0)
    self.cancel_button.grid(row=4, column=1)
    self.window.mainloop() 

  def exit(self):
    if not self.valid:
      self.open_files = []
      self.x_values = []
    self.window.quit()
    self.window.destroy()

  def accept(self):
    if len(self.open_files) == 0:
      self.validation_message["text"] = "No log files given!"
    else:
      x_values = self.entry_x.get()
      for x in x_values.split(","):
        if x.isdigit():
          self.x_values.append(int(x))
      if self.x_values == []:
        self.validation_message["text"] = "Need comma separated list of x-values (x1, x2,...)"
      elif len(self.x_values) != len(self.open_files):
        self.validation_message["text"] = "Nr. of x-values does not match the number of log files!" 
      else:
        self.xlabel = self.entry_xlabel.get()
        self.ylabel = self.entry_ylabel.get()
        self.valid = True
        self.exit()
