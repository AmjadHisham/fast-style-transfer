import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1

def net(image):
    conv1 = _conv_layer(image, 32, 9, 1, name='conv1')
    conv2 = _conv_layer(conv1, 64, 3, 2, name='conv2')
    conv3 = _conv_layer(conv2, 128, 3, 2, name='conv3')
    resid1 = _residual_block(conv3, 3, name='resid1')
    resid2 = _residual_block(resid1, 3, name='resid2')
    resid3 = _residual_block(resid2, 3, name='resid3')
    resid4 = _residual_block(resid3, 3, name='resid4')
    resid5 = _residual_block(resid4, 3, name='resid5')
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2, name='conv_t1')
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2, name='conv_t2')
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False, name='conv_t3')
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def _conv_layer(net, num_filters, filter_size, strides, name, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size, name=name)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net, name=('%s-norm' % name))
    if relu:
        net = tf.nn.relu(net)

    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides, name):
    weights_init = _conv_init_vars(net, num_filters, filter_size, name=name, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.pack(new_shape)
    strides_shape = [1,strides,strides,1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net, name=('%s-norm' % name))
    return tf.nn.relu(net)

def _residual_block(net, filter_size, name):
    tmp = _conv_layer(net, 128, filter_size, 1, name=('%s-conv1' % name))
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False, name=('%s-conv2' % name))

def _instance_norm(net, name, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape), name=('%s-shift' % name))
    scale = tf.Variable(tf.ones(var_shape), name=('%s-scale' % name))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, name, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32, name=name)
    return weights_init
