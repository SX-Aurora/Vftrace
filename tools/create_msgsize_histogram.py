#/usr/bin/env python

# This script takes a Vftrace logfile and parses the
# "Communication profile" section to create a histogram
# of send and receive message sizes.
#
# Note that we are only able to display the average message size.
# The histogram entries are weighted with the number of calls on the
# given stack branch.

import sys
import numpy as np
import matplotlib.pyplot as plt

def read_comm_table (filename):
    weights = []
    send = []
    recv = []

    countdown = -1

    with open (filename, "r") as f:
        for line in f.readlines():
            if 'Communication profile' in line:
                countdown = 3
            elif countdown > 0:
                countdown -= 1
            elif countdown == 0:
                #if '+-' in line: break
                try:
                   tmp = line.split("|")
                   weights.append(int(tmp[1]))
                   send.append(float(tmp[2]))
                   recv.append(float(tmp[3]))
                except:
                   break

    return weights, send, recv

if len(sys.argv) < 2:
    print ("Need a Vftrace logfile to process!")
    sys.exit(-1)

weights, send, recv = read_comm_table (sys.argv[1])

min_send = min(send)
max_send = max(send)
min_recv = min(recv)
max_recv = max(recv)

max_exp_send = int(np.log10(max_send)) + 1
max_exp_recv = int(np.log10(max_recv)) + 1 

fig, ax = plt.subplots(2, 1)
nbins = 15
bins = range(nbins)
ax[0].hist(send, weights=weights, bins=bins)
ax[0].set_title("Send")
ax[1].hist(recv, weights=weights, bins=bins)
ax[1].set_title("Receive")

ticks_send = [np.power(10, i * float(max_exp_send) / nbins) for i in range(nbins)]
ticks_recv = [np.power(10, i * float(max_exp_recv) / nbins) for i in range(nbins)]

def format_tick(tick):
    units = ["B", "kiB", "MiB", "GiB"]
    i_unit = 0
    while tick / 1024 > 1.0:
        tick /= 1024
        i_unit += 1
    if i_unit == 0: # No decimal points below kilobytes
        return "%s %s" % (str(round(tick,0)), units[0])
    else:
       return "%s %s" % (str(round(tick, 2)), units[i_unit])

def format_fn_send(tick_val, tick_pos):
    if int(tick_val) in range(nbins):
       return format_tick(ticks_send[int(tick_val)])
    else:
       return ''

def format_fn_recv(tick_val, tick_pos):
    if int(tick_val) in range(nbins):
       return format_tick(ticks_recv[int(tick_val)])
    else:
       return ''

ax[0].xaxis.set_major_formatter(format_fn_send)
ax[1].xaxis.set_major_formatter(format_fn_recv)

plt.show()
