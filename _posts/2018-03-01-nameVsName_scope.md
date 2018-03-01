---
layout: post
title: tensorflow name Vs Name_scope
---

Let's begin by a short introduction to variable sharing. It is a mechanism in TensorFlow that allows for sharing variables accessed in different parts of the code without passing references to the variable around. The method tf.get_variable can be used with the name of the variable as the argument to either create a new variable with such name or retrieve the one that was created before. This is different from using the tf.Variable constructor which will create a new variable every time it is called (and potentially add a suffix to the variable name if a variable with such name already exists). It is for the purpose of the variable sharing mechanism that a separate type of scope (variable scope) was introduced.

As a result, we end up having two different types of scopes:

name scope, created using tf.name_scope
variable scope, created using tf.variable_scope
Both scopes have the same effect on all operations as well as variables created using tf.Variable, i.e., the scope will be added as a prefix to the operation or variable name.

However, name scope is ignored by tf.get_variable. We can see that in the following example:

with tf.name_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0
The only way to place a variable accessed using tf.get_variable in a scope is to use a variable scope, as in the following example:

with tf.variable_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # my_scope/var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0
This allows us to easily share variables across different parts of the program, even within different name scopes:

with tf.name_scope("foo"):
    with tf.variable_scope("var_scope"):
        v = tf.get_variable("var", [1])
with tf.name_scope("bar"):
    with tf.variable_scope("var_scope", reuse=True):
        v1 = tf.get_variable("var", [1])
assert v1 == v
print(v.name)   # var_scope/var:0
print(v1.name)  # var_scope/var:0
UPDATE

link:

https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
