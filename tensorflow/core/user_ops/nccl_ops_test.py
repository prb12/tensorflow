
import tensorflow as tf
import numpy as np

nccl_ops = tf.load_op_library('nccl_ops.so')

print(dir(nccl_ops))

NUM=4

inits = []
ops = []

sess_config = tf.ConfigProto(
      allow_soft_placement=False,
      device_count={'CPU': 1,
                    'GPU': 4},
      log_device_placement=True)


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
      val = tf.constant(np.arange(1024, dtype=np.float32))
      reduce = nccl_ops.nccl_all_reduce(handle, val)
      ops.append(reduce)

  # Broadcast pattern.
  var = tf.Variable([1,2,3,4])
  towers = []
  for i in xrange(NUM):
    with tf.device('/gpu:%d' % i):
      op = var + 1
      towers.append(op)

  # Reduction pattern.
  addn = tf.add_n(towers)

  tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), '/tmp', 'nccl.pbtxt')
  print('Init...')
  sess.run(inits)

  print('AddN')
  sess.run(addn)

  print('Broadcast')
  sess.run(towers)

  print('AllReduce...')
  vals = sess.run(ops)
  vals = sess.run(ops)
  vals = sess.run(ops)
  vals = sess.run(ops)
  print(vals)
