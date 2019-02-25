from my_convolution import my_conv
import numpy as np
from functools import reduce
import operator
import functools

def iterative_egcd(a, b):
    x,y, u,v = 0,1, 1,0
    while a != 0:
        q,r = b//a,b%a; m,n = x-u*q,y-v*q # use x//y for floor "floor division"
        b,a, x,y, u,v = a,r, u,v, m,n
    return b, x, y

def modinv(a, m):
    g, x, y = iterative_egcd(a, m) 
    if g != 1:
        return None
    else:
        return x % m

def quantize_i_k(i, k, maximum_output):
    this_image = i.copy()
    this_kernel = k.copy()
    all_max = max(this_image.max(), this_kernel.max())
    mult_factor = (maximum_output / all_max)
    this_image = this_image * mult_factor
    this_image = this_image.astype(int)
    this_kernel = this_kernel * mult_factor
    this_kernel = this_kernel.astype(int)
    return this_image, this_kernel, mult_factor

def chinese_theorem(inputs, shape, upper_limit, peymane):
    flat_inputs = list()
    for i in inputs:
        flat_inputs.append(i.flatten())
    result = np.zeros(flat_inputs[0].shape)
    B = list()
    x = list()
    for p in peymane:
        peymanes_copy = peymane.copy()
        peymanes_copy.remove(p)
        B.append(functools.reduce(operator.mul, peymanes_copy, 1))
        x.append(modinv(B[-1], p))
    for i in range(len(flat_inputs[0])):
        this_result = 0
        for j in range(len(peymane)):
            this_result += B[j] * x[j] * flat_inputs[j][i]
        peymane_mult = functools.reduce(operator.mul, peymane, 1)
        while(this_result > upper_limit):
            this_result -= peymane_mult
        result[i] = this_result
    result = np.reshape(result, shape)
    return result

def my_np_quan_conv(a, b, stride, padding, max_range):
    a_max = a.max()
    b_max = b.max()
    both_max = max([a_max, b_max])
    a_quan = ((a * (max_range / both_max)).astype(int)).astype(float)
    b_quan = ((b * (max_range / both_max)).astype(int)).astype(float)
    conv = my_conv(a_quan, b_quan, stride, padding)
    conv_dequan = conv * ((both_max / max_range)**2)
    
    return conv_dequan

def my_tensor_quan_conv(a, b, stride, padding, max_range):
    a_max = tf.reduce_max(a)
    b_max = tf.reduce_max(b)
    max_concat = tf.stack([a_max, b_max])
    both_max = tf.reduce_max(max_concat)
    a_muled = tf.multiply(a, max_range / both_max)
    b_muled = tf.multiply(b, max_range / both_max)
    a_quan = tf.cast(a_muled, tf.int32)
    b_quan = tf.cast(b_muled, tf.int32)
    float_a_quan = tf.cast(a_quan, tf.float32)
    float_b_quan = tf.cast(b_quan, tf.float32)
    conv = tf.nn.conv2d(float_a_quan, float_b_quan, stride, padding=padding)    
    conv_dequan_one = tf.multiply(conv, both_max / max_range)
    conv_dequan = tf.multiply(conv_dequan_one, both_max / max_range)
    
    return conv_dequan

def quantized_conv(image, kernel, s_h, s_w, padding):
    image_min = tf.reduce_min(image)
    image_max = tf.reduce_max(image)
    kernel_min = tf.reduce_min(kernel)
    kernel_max = tf.reduce_max(kernel)
    quantized_image, _,_ = tf.quantize(image, image_min, image_max, tf.quint8)
    quantized_kernel, _, _= tf.quantize(kernel, kernel_min, kernel_max, tf.quint8)
    quan_conv = tf.nn.quantized_conv2d(quantized_image, quantized_kernel, image_min, image_max, 
                                   kernel_min, kernel_max, [1, s_h, s_w, 1], padding)
    shit = tf.cast(quan_conv.output, dtype=tf.qint32)
    deq_out = tf.quantization.dequantize(shit, quan_conv.min_output, quan_conv.max_output)
    return deq_out

