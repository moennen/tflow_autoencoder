'''
Convolutionnal Encoder
'''


def conv_layer(x, filter_size, step):
    layer_w = tf.Variable(tf.random_normal(filter_size))
    layer_b = tf.Variable(tf.random_normal(filter_size[3]))
    layer = tf.nn.conv2d(x, layer_w, strides=[1, step, step, 1], padding='VALID')
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer)
    return layer


def deconv_layer(x, filter_size, output_size, step):
    # nb : conv2d_transpose need the outsize firt and then the insize
    w_shape = [filer_size[0], filter_size[1], filter_size[3], filter_size[2]]
    layer_w = tf.Variable(tf.random_normal(w_shape))
    layer_b = tf.Variable(tf.random_normal(filter_size[3]))
    out_shape = [tf.shape(x)[0]], output_size[0], output_size[1], filter_size[3]]
    layer = tf.nn.conv2d_transpose(x, layer_w, output_shape= out_shape,
                                   strides = [1, step, step, 1], padding = 'VALID')
    layer=tf.nn.bias_add(layer, layer_b)
    layer=tf.nn.relu(layer)
    return layer


def create_model(layer0, params):
    # Create the encoder flow of the network
    layer1=conv_layer(layer0, [7, 7, 3, 32], 2)
    layer2=conv_layer(layer1, [5, 5, 32, 64], 2)
    layer3=conv_layer(layer2, [5, 5, 64, 128], 2)
    layer4=conv_layer(layer3, [3, 3, 128, 128], 1)
    layer5=conv_layer(layer4, [3, 3, 128, 256], 2)
    layer6=conv_layer(layer5, [3, 3, 256, 256], 1)
    layer7=conv_layer(layer6, [3, 3, 256, 256], 2)
    layer8=conv_layer(layer7, [3, 3, 256, 256], 1)
    layer9=conv_layer(layer8, [3, 3, 256, 512], 2)

    # Create the decoder flow of the network
    layer10=deconv_layer(layer9_8, layer8,)

    return
