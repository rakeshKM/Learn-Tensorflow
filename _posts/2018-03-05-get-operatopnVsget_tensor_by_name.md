---
layout: post
title: tensor constant,placeholder and variable
---

#tensor
in the graph, every node is an operation, which can have Tensors as inputs or outputs. 
In a TensorFlow graph, each node has zero or more inputs and zero or more outputs, and represents the instantiation of an operation.
TensorFlow doesn't have first-class Tensor objects(object named tensor), meaning that there are no notion of Tensor in the underlying graph that's executed by the runtime.
Instead the graph consists of op nodes connected to each other, representing operations. An operation allocates memory for its outputs,
which are available on endpoints :0, :1, etc, and you can  think of each of these endpoints as a Tensor.
Since most ops have only 1 endpoint, :0 is dropped. 
If you have tensorX corresponding to nodename:0 you can fetch its value as sess.run(tensorX) or sess.run('nodename:0').
Execution granularity happens at operation level, so the run method will execute op which will compute all of the endpoints, not just the :0 endpoint. 
It's possible to have an Op node with no outputs (like tf.group) in which case there are no tensors associated with it. 
It is not possible to have tensors without an underlying Op node.



Variables, constant and placeholders are nodes, aka, instantiation of OPERATIONS just like tf.mul or tf.add . 
I think they produce tensors as output, but they themselves are not tensors

#tf.constant

value = tf.constant(1)

So with tf.constant you get a single operation node, and you can fetch it using sess.run("Const:0") or sess.run(value)
just the most basic operation node, which contains a fixed value given when you create it



#tf.placeholder

TensorFlow provides a placeholder operation that must be fed with data on execution,
placeholders are operation to which you can feed a value (with the feed_dict argument in sess.run())

example 1:

w1 = tf.placeholder("float", name="w1")

w2 = tf.placeholder("float", name="w2")

b1= tf.Variable(2.0,name="bias")

feed_dict ={w1:4,w2:8}

example 2:

value=tf.placeholder(tf.int32) # creates a regular node with name Placeholder

feed_dict={"Placeholder:0":2} or feed_dict={value:2}  # both feeding is correct




#tf.Variable 

Variables are operation which you can update (with var.assign()). 

value = tf.Variable(tf.ones_initializer()(()))

value2 = value+3

it creates two nodes Variable and Variable/read, the :0 endpoint is a valid value to fetch on both of these nodes. 
However Variable:0 has a special ref type meaning it can be used as an input to mutating operations. 
The result of Python call tf.Variable is a Python Variable object and
there's some Python magic to substitute Variable/read:0 or Variable:0 depending on whether mutation is necessary. 
Since most ops have only 1 endpoint, :0 is dropped. 


For ops like tf.split or tf.nn.top_k which create nodes with multiple endpoints,
Python's session.run call automatically wraps output in tuple or collections.namedtuple of Tensor objects which can be fetched individually.
