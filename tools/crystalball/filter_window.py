import re

import tkinter as tk

class filter_window:

  def __init__(self, root):
    self.filter_tmp = []
    self.filter_functions = []
    self.root = root
    self.filter_entry_frame = None
    self.filter_button = None
    self.fiter_entry = None
    self.filter_show = None
    self.button_frame = None
    self.accept_button = None
    self.reset_button = None
    self.cancel_button = None
    self.window = None
    self.needs_update = False

  def exit(self):
    self.window.quit()
    self.window.destroy()

  def evaluate_filter(self, extr_entries):
    self.filter_tmp = []
    expr = self.filter_entry.get() 
    for ee in extr_entries:
      m = re.search(expr, ee.func_name) 
      if m is not None:
        if not ee.func_name in self.filter_tmp: self.filter_tmp.append(ee.func_name)
    s = "Functions: "
    for i, f in enumerate(self.filter_tmp):
      if i > 0: s += ", "
      s += f
    self.filter_show["text"] = s

  def accept(self):
    self.filter_functions = self.filter_tmp
    self.needs_update = self.filter_functions != []
    self.exit()

  def reset (self):
    self.filter_functions = []
    self.needs_update = True
    self.exit()

  def skip(self, func_name):
    if self.filter_functions != []:
      return not func_name in self.filter_functions 
    else:
      return False

  def open_window(self, global_dict):
    self.window = tk.Toplevel (master = self.root)
    self.window.protocol ("WM_DELETE_WINDOW", self.exit) 
    self.filter_entry_frame = tk.Frame (master = self.window)
    self.filter_button = tk.Button (master = self.filter_entry_frame, text = "Filter: ",
                                    command = lambda p = global_dict.values(): self.evaluate_filter(p)).grid(row=0, column=0)
    self.filter_entry = tk.Entry (master = self.filter_entry_frame, width=50)
    self.filter_entry.grid(row=0, column=1)
    self.filter_entry_frame.grid(row=0, column=0)

    # Must be filled 
    self.filter_show = tk.Label (master = self.window, text = "Functions: ")
    self.filter_show.grid(row=1, column=0)

    self.button_frame = tk.Frame (master = self.window) 
    self.accept_button = tk.Button (master = self.button_frame, text = "Accept", command = self.accept).grid(row=0, column=0)
    self.reset_button = tk.Button (master = self.button_frame, text = "Reset", command = self.reset).grid(row=0, column=1)
    self.cancel_button = tk.Button (master = self.button_frame, text = "Cancel", command = self.exit).grid(row=0, column=2) 
    self.button_frame.grid(row=2, column=0)

    self.window.mainloop()
