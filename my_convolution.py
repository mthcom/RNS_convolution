from sc_convertor import *
import tensorflow as tf
import numpy as np
from math import ceil

def my_pad(arr, pad_top, pad_bottom, pad_left, pad_right):
    pads = [pad_top, pad_bottom, pad_left, pad_right]
    if(len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]):
        return arr
    result = list()
    for i in range(pads[0]):
        result.append(np.zeros(arr.shape[0] + pads[2] + pads[3]))
    for row in arr:
        new_row = np.concatenate((np.zeros(pads[2]),row,np.zeros(pads[3])))
        result.append(new_row)
    for i in range(pads[1]):
        result.append(np.zeros(arr.shape[0] + pads[2] + pads[3]))
    return np.array(result)

def my_conv(inputs, filters, strides, padding, return_tf=False, use_skippy=False, bits=8): #(NHWC for image and HWCN for kernel)
    with tf.Session() as sess:
        if(type(inputs) != np.ndarray):
            inputs = sess.run(inputs)
        if(type(filters) != np.ndarray):
            filters = sess.run(filters)
    if(padding == "VALID"):
        output_shape = (inputs.shape[0], ceil((inputs.shape[1] - filters.shape[0] + 1) / strides[1]), ceil((inputs.shape[2] - filters.shape[1] + 1) / strides[2]), filters.shape[3])
    else: # SAME
        if (inputs.shape[1] % strides[1] == 0):
            pad_along_height = max(filters.shape[0] - strides[1], 0)
        else:
            pad_along_height = max(filters.shape[0] - (inputs.shape[1] % strides[1]), 0)
        if (inputs.shape[2] % strides[2] == 0):
            pad_along_width = max(filters.shape[1] - strides[2], 0)
        else:
            pad_along_width = max(filters.shape[1] - (inputs.shape[2] % strides[2]), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        trans_inputs = inputs.transpose((0, 3, 1, 2))
        results = list()
        for i in range(trans_inputs.shape[0]):
            for j in range(trans_inputs.shape[1]):
                results.append(my_pad(trans_inputs[i][j], pad_top, pad_bottom, pad_left, pad_right))
        trans_inputs = np.zeros((trans_inputs.shape[0], trans_inputs.shape[1], results[0].shape[0], results[0].shape[1]))
        for i in range(trans_inputs.shape[0]):
            for j in range(trans_inputs.shape[1]):
                trans_inputs[i][j] = results[i + j]
        untrans_inputs = trans_inputs.transpose((0, 2, 3, 1))
        output_shape = (int(inputs.shape[0]), int(ceil((inputs.shape[1]) / strides[1])), int(ceil((inputs.shape[2]) / strides[2])), int(filters.shape[3]))
        inputs = untrans_inputs
#     print(output_shape)
    output = np.zeros(output_shape)
#     print(output_shape[0]*output_shape[3] * output_shape[1] * output_shape[2] * filters.shape[1] * filters.shape[0] * inputs.shape[3])
    for n_image in range(output_shape[0]):
        for n_kernel in range(output_shape[3]):
            for h_base in range(output_shape[1]):
                for w_base in range(output_shape[2]):
                    for h in range(filters.shape[0]):
                        for w in range(filters.shape[1]):
                            for c in range(inputs.shape[3]):
                                this_input = inputs[n_image][h_base * strides[1] + h][w_base * strides[2] + w][c]
                                this_filter = filters[h][w][c][n_kernel]
                                if(use_skippy):
                                    this_mult = sc_convertor.new_mult(this_input, this_filter, bits)
#                                     this_mult = 0
                                else:
                                    this_mult = this_filter * this_input
                                output[n_image][h_base][w_base][n_kernel] += this_mult
    if return_tf:
        return tf.constant(output, dtype=tf.float32, name="allah")
    else:
        return output