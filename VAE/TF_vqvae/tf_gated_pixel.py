from logging import getLogger
import logging

import tensorflow as tf
import numpy as np

logger = getLogger(__name__)


class GatedPixel:

    def __init__(self, conf, height, width, num_channels):
        logger.info("Building gated_pixel_cnn starts")

        # self.sess = sess
        self.data = conf.data
        self.height, self.width, self.channel = height, width, num_channels
        self.pixel_depth = 256
        self.q_levels = q_levels = conf.q_levels

        self.inputs = tf.placeholder(tf.float32, [None, height, width, num_channels])  # [N,H,W,C]
        self.target_pixels = tf.placeholder(tf.int64, [None, height, width,
                                                       num_channels])  # [N,H,W,C] (the index of a one-hot representation of D)

        # input conv layer
        logger.info("Building CONV_IN")
        net = conv(self.inputs, conf.gated_conv_num_feature_maps, [7, 7], "A", num_channels, scope="CONV_IN")

        # main gated layers
        for idx in range(conf.gated_conv_num_layers):
            scope = 'GATED_CONV%d' % idx
            net = gated_conv(net, [3, 3], num_channels, scope=scope)
            logger.info("Building %s" % scope)

        # output conv layers
        net = tf.nn.relu(conv(net, conf.output_conv_num_feature_maps, [1, 1], "B", num_channels, scope='CONV_OUT0'))
        logger.info("Building CONV_OUT0")
        self.logits = tf.nn.relu(
            conv(net, q_levels * num_channels, [1, 1], "B", num_channels, scope='CONV_OUT1'))  # shape [N,H,W,DC]
        logger.info("Building CONV_OUT1")

        if (num_channels > 1):
            self.logits = tf.reshape(self.logits, [-1, height, width, q_levels,
                                                   num_channels])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            self.logits = tf.transpose(self.logits,
                                       perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

        flattened_logits = tf.reshape(self.logits, [-1, q_levels])  # [N,H,W,C,D] -> [NHWC,D]
        target_pixels_loss = tf.reshape(self.target_pixels, [-1])  # [N,H,W,C] -> [NHWC]

        logger.info("Building loss and optims")
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            flattened_logits, target_pixels_loss))

        flattened_output = tf.nn.softmax(flattened_logits)  # shape [NHWC,D], values [probability distribution]
        self.output = tf.reshape(flattened_output, [-1, height, width, num_channels,
                                                    q_levels])  # shape [N,H,W,C,D], values [probability distribution]

        optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)

        new_grads_and_vars = \
            [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
        self.optim = optimizer.apply_gradients(new_grads_and_vars)

        # show_all_variables()

        logger.info("Building gated_pixel_cnn finished")

    def predict(self, images):
        '''
        images # shape [N,H,W,C]
        returns predicted image # shape [N,H,W,C]
        '''
        # self.output shape [NHWC,D]
        pixel_value_probabilities = self.sess.run(self.output, {
            self.inputs: images})  # shape [N,H,W,C,D], values [probability distribution]

        # argmax or random draw # [NHWC,1]  quantized index - convert back to pixel value
        pixel_value_indices = np.argmax(pixel_value_probabilities,
                                        4)  # shape [N,H,W,C], values [index of most likely pixel value]
        pixel_values = np.multiply(pixel_value_indices,
                                   ((self.pixel_depth - 1) / (self.q_levels - 1)))  # shape [N,H,W,C]

        return pixel_values

    def test(self, images, with_update=False):
        if with_update:
            _, cost = self.sess.run([self.optim, self.loss],
                                    {self.inputs: images[0], self.target_pixels: images[1]})
        else:
            cost = self.sess.run(self.loss, {self.inputs: images[0], self.target_pixels: images[1]})
        return cost

    def generate_from_occluded(self, images, num_generated_images, occlude_start_row):
        samples = np.copy(images[0:num_generated_images, :, :, :])
        samples[:, occlude_start_row:, :, :] = 0.

        for i in range(occlude_start_row, self.height):
            for j in range(self.width):
                for k in range(self.channel):
                    next_sample = self.predict(samples) / (self.pixel_depth - 1.)  # argmax or random draw here
                    samples[:, i, j, k] = next_sample[:, i, j, k]

        return samples

    def generate(self, images):
        samples = images[0:9, :, :, :]
        occlude_start_row = 18
        samples[:, occlude_start_row:, :, :] = 0.

        for i in range(occlude_start_row, self.height):
            for j in range(self.width):
                for k in range(self.channel):
                    next_sample = self.predict(samples) / (self.pixel_depth - 1.)  # argmax or random draw here
                    samples[:, i, j, k] = next_sample[:, i, j, k]

        return samples


WEIGHT_INITIALIZER = tf.contrib.layers.xavier_initializer()

logger = logging.getLogger(__name__)


def get_shape(layer):
    return layer.get_shape().as_list()


def conv(
        inputs,
        num_outputs,
        kernel_shape,  # [kernel_height, kernel_width]
        mask_type,  # None, 'A', 'B' or 'V'
        data_num_channels,
        strides=[1, 1],  # [column_wise_stride, row_wise_stride]
        padding="SAME",
        activation_fn=None,
        weights_initializer=WEIGHT_INITIALIZER,
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer,
        biases_regularizer=None,
        scope="conv2d"):
    with tf.variable_scope(scope):
        mask_type = mask_type.lower()
        if mask_type == 'v' and kernel_shape == [1, 1]:
            # No mask required for Vertical 1x1 convolution
            mask_type = None
        num_inputs = get_shape(inputs)[-1]

        kernel_h, kernel_w = kernel_shape
        stride_h, stride_w = strides

        assert kernel_h % 2 == 1 and kernel_w % 2 == 1, \
            "kernel height and width should be an odd number"

        weights_shape = [kernel_h, kernel_w, num_inputs, num_outputs]
        weights = tf.get_variable("weights", weights_shape,
                                  tf.float32, weights_initializer, weights_regularizer)

        if mask_type is not None:
            mask = _create_mask(num_inputs, num_outputs, kernel_shape, data_num_channels, mask_type)
            weights *= tf.constant(mask, dtype=tf.float32)
            tf.add_to_collection('conv2d_weights_%s' % mask_type, weights)

        outputs = tf.nn.conv2d(inputs,
                               weights, [1, stride_h, stride_w, 1], padding=padding, name='outputs')
        tf.add_to_collection('conv2d_outputs', outputs)

        if biases_initializer != None:
            biases = tf.get_variable("biases", [num_outputs, ],
                                     tf.float32, biases_initializer, biases_regularizer)
            outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

        if activation_fn:
            outputs = activation_fn(outputs, name='outputs_with_fn')

        logger.debug('[conv2d_%s] %s : %s %s -> %s %s' \
                     % (mask_type, scope, inputs.name, inputs.get_shape(), outputs.name, outputs.get_shape()))

        return outputs


# for this type layer: num_outputs = num_inputs = number of channels in input layer
def gated_conv(inputs, kernel_shape, data_num_channels, scope="gated_conv"):
    with tf.variable_scope(scope):
        # Horiz inputs/outputs on left in case num_inputs not multiple of 6, because Horiz is RGB gated and Vert is not.
        # inputs shape [N,H,W,C]
        horiz_inputs, vert_inputs = tf.split(3, 2, inputs)
        p = get_shape(horiz_inputs)[-1]
        p2 = 2 * p

        # vertical n x n conv
        # p in channels, 2p out channels, vertical mask, same padding, stride 1
        vert_nxn = conv(vert_inputs, p2, kernel_shape, 'V', data_num_channels, scope="vertical_nxn")

        # vertical blue diamond
        # 2p in channels, p out channels, vertical mask
        vert_gated_out = _gated_activation_unit(vert_nxn, kernel_shape, 'V', data_num_channels,
                                                scope="vertical_gated_activation_unit")

        # vertical 1 x 1 conv
        # 2p in channels, 2p out channels, no mask?, same padding, stride 1
        vert_1x1 = conv(vert_nxn, p2, [1, 1], 'V', data_num_channels, scope="vertical_1x1")

        # horizontal 1 x n conv
        # p in channels, 2p out channels, horizontal mask B, same padding, stride 1
        horiz_1xn = conv(horiz_inputs, p2, kernel_shape, 'B', data_num_channels, scope="horizontal_1xn")
        horiz_gated_in = vert_1x1 + horiz_1xn

        # horizontal blue diamond
        # 2p in channels, p out channels, horizontal mask B
        horiz_gated_out = _gated_activation_unit(horiz_gated_in, kernel_shape, 'B', data_num_channels,
                                                 scope="horizontal_gated_activation_unit")

        # horizontal 1 x 1 conv
        # p in channels, p out channels, mask B, same padding, stride 1
        horiz_1x1 = conv(horiz_gated_out, p, kernel_shape, 'B', data_num_channels, scope="horizontal_1x1")

        horiz_outputs = horiz_1x1 + horiz_inputs

        return tf.concat(3, [horiz_outputs, vert_gated_out])


def _create_mask(
        num_inputs,
        num_outputs,
        kernel_shape,
        data_num_channels,
        mask_type,  # 'A', 'B' or 'V'
):
    '''
    Produces a causal mask of the given type and shape
    '''
    mask_type = mask_type.lower()
    kernel_h, kernel_w = kernel_shape

    center_h = kernel_h // 2
    center_w = kernel_w // 2

    mask = np.ones(
        (kernel_h, kernel_w, num_inputs, num_outputs),
        dtype=np.float32)  # shape [KERNEL_H, KERNEL_W, NUM_INPUTS, NUM_OUTPUTS]

    if mask_type == 'v':
        mask[center_h:, :, :, :] = 0.
    else:
        mask[center_h, center_w + 1:, :, :] = 0.
        mask[center_h + 1:, :, :, :] = 0.

        if mask_type == 'b':
            mask_pixel = lambda i, j: i > j
        else:
            mask_pixel = lambda i, j: i >= j

        for i in range(num_inputs):
            for j in range(num_outputs):
                if mask_pixel(i % data_num_channels, j % data_num_channels):
                    mask[center_h, center_w, i, j] = 0.

    return mask


# implements equation (2) of the paper
# returns 1/2 number of channels as input
def _gated_activation_unit(inputs, kernel_shape, mask_type, data_num_channels, scope="gated_activation_unit"):
    with tf.variable_scope(scope):
        p2 = get_shape(inputs)[-1]

        # blue diamond
        # 2p in channels, 2p out channels, mask, same padding, stride 1
        # split 2p out channels into p going to tanh and p going to sigmoid
        bd_out = conv(inputs, p2, kernel_shape, mask_type, data_num_channels, scope="blue_diamond")  # [N,H,W,C[,D]]
        bd_out_0, bd_out_1 = tf.split(3, 2, bd_out)
        tanh_out = tf.tanh(bd_out_0)
        sigmoid_out = tf.sigmoid(bd_out_1)

    return tanh_out * sigmoid_out