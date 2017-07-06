import os
import psutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

pid = os.getpid()
py = psutil.Process(pid)

def memory():
    memoryUse = py.memory_info()[0]/2**30  # memory use in GB...I think
    return memoryUse


ms = []
for i in range(10000):
    m = memory()
    print(m)
    ms.append(m)

    x_ph = tf.placeholder(tf.float32, shape=(None, 200, 200, 4))
    l = tf.layers.conv2d(x_ph, 64, 3)

    del x_ph, l
    tf.reset_default_graph()

fig, ax = plt.subplots(1, 1)
ax.plot(ms)
fig.savefig('memory_use.png')
