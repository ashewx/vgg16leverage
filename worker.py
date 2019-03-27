# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["172.23.10.2:2222", "172.23.10.6:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)

print("Starting server #{}".format(task_number))

server.start()
server.join()
