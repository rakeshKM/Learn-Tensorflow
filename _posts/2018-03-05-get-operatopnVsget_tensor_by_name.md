
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
