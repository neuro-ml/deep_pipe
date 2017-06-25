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

graph = tf.get_default_graph()
x_ph = tf.placeholder(tf.float32, shape=(None, 200, 200, 4))
l = tf.layers.conv2d(x_ph, 64, 3)
init = tf.global_variables_initializer()
graph.finalize()

ms = []
for i in range(10000):
    m = memory()
    print(m)
    ms.append(m)

    session = tf.Session()

    session.run(init)
    session.close()

fig, ax = plt.subplots(1, 1)
ax.plot(ms)
fig.savefig('memory_use.png')
