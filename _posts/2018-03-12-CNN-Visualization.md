
---
layout: post
title: CNN Visualizaion
---

## visualization through tensorboard
https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial

https://www.youtube.com/watch?list=PLOU2XLYxmsIKGc_NBoIhTn2Qhraji53cv&v=eBbEDRsCmv4

#### finding gradient of layer
tf.gradients(layer_output_tensor, input_x). This will give you direct original data from any given layer output tensor

### https://github.com/tensorflow/tensorflow/issues/2169
#### unpooling

    def unravel_argmax(argmax, shape):
        output_list = [argmax // (shape[2]*shape[3]),
                       argmax % (shape[2]*shape[3]) // shape[3]]
        return tf.pack(output_list)

      def unpool_layer2x2_batch(bottom, argmax):
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
    
##### much faster

    def unpool_layer2x2_batch(bottom, argmax):
        bottom_shape = tf.shape(bottom)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])
        return tf.scatter_nd(indices, values, tf.to_int64(top_shape))
    
#### new 
       def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
        """
           Unpooling layer after max_pool_with_argmax.
           Args:
               pool:   max pooled output tensor
               ind:      argmax indices
               ksize:     ksize is the same as for the pool
           Return:
               unpool:    unpooling tensor
        """
        with tf.variable_scope(scope):
            input_shape = tf.shape(pool)
            output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                              shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape)

            set_input_shape = pool.get_shape()
            set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
            ret.set_shape(set_output_shape)
            return ret

#### pre-1
    def unravel_argmax(argmax, shape):
        argmax_shape = argmax.get_shape()
        new_1dim_shape = tf.shape(tf.constant(0, shape=[tf.Dimension(4), argmax_shape[0]*argmax_shape[1]*argmax_shape[2]*argmax_shape[3]]))
        batch_shape = tf.constant(0, dtype=tf.int64, shape=[argmax_shape[0], 1, 1, 1]).get_shape()
        b = tf.multiply(tf.ones_like(argmax), tf.reshape(tf.range(shape[0]), batch_shape))
        y = argmax // (shape[2] * shape[3])
        x = argmax % (shape[2] * shape[3]) // shape[3]
        c = tf.ones_like(argmax) * tf.range(shape[3])
        pack = tf.stack([b, y, x, c])
        pack = tf.reshape(pack, new_1dim_shape)
        pack = tf.transpose(pack)
        return pack


    def unpool(updates, mask, ksize=[1, 2, 2, 1]):
        input_shape = updates.get_shape()
        new_dim_y = input_shape[1] * ksize[1]
        new_dim_x = input_shape[2] * ksize[2]
        output_shape = tf.to_int64((tf.constant(0, dtype=tf.int64, shape=[input_shape[0], new_dim_y, new_dim_x, input_shape[3]]).get_shape()))
        indices = unravel_argmax(mask, output_shape)
        new_1dim_shape = tf.shape(tf.constant(0, shape=[input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]]))
        values = tf.reshape(updates, new_1dim_shape)
        ret = tf.scatter_nd(indices, values, output_shape)
    return ret
    
    here:
          'mask' - is the result of the tf.nn.max_pool_with_argmax(input=image, ksize=ksize, strides=[1, 2, 2, 1], padding='SAME') operation.
    mask is the tensor of indices of max values of input_tensor.
    input_tensor is the input tensor of the maxpool operation.
    This operation ( unpool ) is inverted to the maxpool.
    
#### pre-2
   
       I think I found bug/issue with tf.nn.max_pool_with_argmax and the unpool workaround as presented here.

    tf.nn.max_pool_with_argmax indices are calculated as (y * w + x) * channels + c, but the "w" the width of the input tensor, not the width of the input tensor + padding, if any padding (padding='SAME' and width of tensor being odd) is applied.

    Using the unpool method, the width is calculated by dividing/modulo that output with input_shape[2] * ksize[2], with padding this will be 1 pixel bigger than the width that tf.nn.max_pool_with_argmax uses for its argmax output. So if a padding is applied, every row of the output image of the unpool() op will be slightly offset, leading to the image being slightly tilted.

    I'm currently implementing SegNet, which has several unpool operations one after the other, each making the tilting worse if there was any padding for it, which is really noticeable when looking at the final output.

    My workaround was to change the proposed unpool operation by simply adding an input-argument for the output shape as follows:

    def unpool(updates, mask, ksize=[1, 2, 2, 1], output_shape=None, name=''):
        with tf.variable_scope(name):
            mask = tf.cast(mask, tf.int32)
            input_shape = tf.shape(updates, out_type=tf.int32)
            #  calculation new shape
            if output_shape is None:
                output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask, dtype=tf.int32)
            batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
            feature_range = tf.range(output_shape[3], dtype=tf.int32)
            f = one_like_mask * feature_range
            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
            values = tf.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret
    then when calling the op, I supply the shape of the convolution in the encoder part of segnet as output_shape, so the code will use the correct (well, incorrect...) width when transforming the tf.nn.max_pool_with_argmax indices.

    Arguably, this is a bug with tf.nn.max_pool_with_argmax, since it should calculate the argmax indices by taking potential padding into account
#### in documentation 

       def unpool_2d(pool, 
                  ind, 
                  stride=[1, 2, 2, 1], 
                  scope='unpool_2d'):
      """Adds a 2D unpooling op.
      https://arxiv.org/abs/1505.04366
      Unpooling layer after max_pool_with_argmax.
           Args:
               pool:        max pooled output tensor
               ind:         argmax indices
               stride:      stride is the same as for the pool
           Return:
               unpool:    unpooling tensor
      """
      with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret
