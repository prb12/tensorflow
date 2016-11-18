
import tensorflow as tf
import numpy as np

from tensorflow.python.client import timeline

nccl_ops = tf.load_op_library('nccl_ops.so')

print(dir(nccl_ops))

NUM=4

inits = []
ops = []

sess_config = tf.ConfigProto(
      allow_soft_placement=False,
      device_count={'CPU': 1,
                    'GPU': 4},
      log_device_placement=False)


with tf.Session('', config=sess_config) as sess:
  # Manual Nccl all_reduce
  unique_id = nccl_ops.nccl_unique_id()
  for i in xrange(NUM):
    with tf.device('/gpu:%d' % i):
      handle = nccl_ops.nccl_comm_resource_handle_op(shared_name="comm%d" % i)
      print(handle)
      comm = nccl_ops.nccl_init_comm(handle, unique_id, rank=i, N=NUM)
      #print(comm)
      inits.append(comm)

      #bcast = nccl_ops.nccl_broadcast(handle, val)
      #ops.append(bcast)
      val = tf.constant(np.arange(1<<20, dtype=np.float32))
      reduce = nccl_ops.nccl_all_reduce(handle, val)
      ops.append(reduce.op)

  # Broadcast pattern.
  #var = tf.Variable([1,2,3,4])
  var = tf.get_variable('var', shape=(1024,1024))
  towers = []
  for i in xrange(NUM):
    with tf.device('/gpu:%d' % i):
      op = var + 1
      towers.append(op)

  # Reduction pattern.
  addn = tf.add_n(towers)

  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
  tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/tmp', 'nccl.pbtxt')

  print('Init...')
  sess.run(tf.initialize_all_variables())

  print('AddN')
  runoptions = tf.RunOptions()
  run_metadata = tf.RunMetadata()
  sess.run(addn, options=run_options, run_metadata=run_metadata)
  tl = timeline.Timeline(run_metadata.step_stats)
  trace = tl.generate_chrome_trace_format()
  with open('addn.rmd.proto', 'w') as f:
    f.write(run_metadata.SerializeToString())
  with open('addn.timeline.json', 'w') as f:
    f.write(trace)
  #stop

  print('Nccl Init')
  sess.run(inits)

  print('Broadcast')
  run_metadata = tf.RunMetadata()
  sess.run(towers, options=run_options, run_metadata=run_metadata)
  tl = timeline.Timeline(run_metadata.step_stats)
  with open('broadcast.rmd.proto', 'w') as f:
    f.write(run_metadata.SerializeToString())
  with open('braodcast.timeline.json', 'w') as f:
    f.write(trace)

  print('AllReduce...')
  run_metadata = tf.RunMetadata()
  vals = sess.run(ops, options=run_options, run_metadata=run_metadata)
  vals = sess.run(ops)
  vals = sess.run(ops)
  vals = sess.run(ops)
  tl = timeline.Timeline(run_metadata.step_stats)
  with open('allreduce.rmd.proto', 'w') as f:
    f.write(run_metadata.SerializeToString())
  with open('allreduce.timeline.json', 'w') as f:
    f.write(trace)
  print(vals)
