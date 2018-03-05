
---
layout: post
title: tensor:get_operatopn_by_name Vs get_tensor_by_name
---


tf.Operation,When you call

op = graph.get_operation_by_name('logits')
... it returns an instance of type tf.Operation, which is a node in the computational graph, which performs some op on its inputs and produces one or more outputs. In this case, it's a plus op.

One can always evaluate an op in a session, and if this op needs some placehoder values to be fed in, the engine will force you to provide them. Some ops, e.g. reading a variable, don't have any dependencies and can be executed without placeholders.

In your case, (I assume) logits are computed from the input placeholder x, so logits doesn't have any value without a particular x.



tf.Tensor,On the other hand, calling

tensor = graph.get_tensor_by_name('logits:0')
... returns an object tensor, which has the type tf.Tensor:

Represents one of the outputs of an Operation.

A Tensor is a symbolic handle to one of the outputs of an Operation. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow tf.Session.


#### https://stackoverflow.com/questions/48022794/tensorflow-difference-get-tensor-by-name-vs-get-operation-by-name?noredirect=1&lq=1

#### http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

You could add_collection(name, operation/tensor) to the graph, and get_collection(name) returns a list of values in the collection with the given name.
get_tensor_by_name(name) returns a tensor with the given name.
get_operation_by_name(name) returns a operation with the given name.


example 1:

sess=tf.Session()

saver = tf.train.import_meta_graph('/home/rakesh/WORK/CNN_Lookout/runs/1519022246/checkpoints/model-200.meta') #load graph

saver.restore(sess,tf.train.latest_checkpoint('/home/rakesh/WORK/CNN_Lookout/runs/1519022246/checkpoints/./')) #load weigt

graph = tf.get_default_graph()

embedding_W= graph.get_tensor_by_name("embedding/W:0") #runs ok

embedding_W= graph.get_tensor_by_name("embedding/W") #value error:The name 'embedding/W' refers to an Operation,Tensor names must be of the form "<op_name>:<output_index>"

embedding_W= graph.get_operation_by_name("embedding/W") #TypeError: Can't convert Operation 'embedding/W' to Tensor (target dtype=None, name='params_0', as_ref=False)

embedding_W= graph.get_operation_by_name("embedding/W:0") #ValueError: Name 'embedding/W:0' appears to refer to a Tensor, not a Operation.
##### conclusion W is a tensor as it contain values , not an operation

Example 2:

The tf.Graph.get_operation_by_name() method always returns a tf.Operation object. When you pass a tf.Operation object to tf.Session.run(), TensorFlow will execute that operation (and everything on which it depends) and discard its outputs (if any).

If you are interested in the value of a particular output, you have to tell TensorFlow which output (a tf.Tensor) you are interested in. There are two main options:

Get a tf.Operation from the graph and then select one of its outputs:

op = my_graph.get_operation_by_name("op")
output = op.outputs[0]
print(sess.run(output))
Get a tf.Tensor from the graph by calling tf.Graph.get_tensor_by_name(), and appending ":<output index>" to the operation's name:

output = my_graph.get_tensor_by_name("op:0")
print(sess.run(output))

Why does TensorFlow draw this distinction? For one thing, a operation can have multiple outputs, so it is sometimes necessary to be specific about which output you want to fetch. For another, an operation may have a side effect and produce a large output—see tf.assign() for an example—and it is often more efficient to pass the tf.Operation to sess.run() so that the value is not copied back into the Python program.

meta = 'path/to/file.meta'

my_graph = tf.Graph()
with my_graph.as_default():
        with tf.Session() as sess:
                saver = tf.train.import_meta_graph(meta)
                saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(meta)))
                op = my_graph.get_operation_by_name("op")
                print(sess.run(op))
                
                
