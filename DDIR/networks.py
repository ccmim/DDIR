"""
Networks for DDIR
Some basic functions are borrowed from Voxelmorph
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
from keras import layers, models
import keras.layers as KL
from keras.layers import Layer, merge, multiply,Activation,Flatten,Dense,GlobalAveragePooling3D
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, Multiply
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

#from keras.layers import Input

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.models as nrn_models
import neuron.utils as nrn_utils

# other vm functions
import losses

def unet_core(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt], outputs=[x])




def multi_channel_unet(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_seg=None, tgt_seg=None, src_feats=1, tgt_feats=1,flag=0):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    
    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    if src_seg is None:
        src_seg = Input(shape=[*vol_size, src_feats])
    if tgt_seg is None:
        tgt_seg = Input(shape=[*vol_size, tgt_feats])
    
    
    
    seg_src = Lambda(mask_gen1)(src_seg)
    seg_tgt = Lambda(mask_gen1)(tgt_seg)
    src0 = multiply([src,seg_src[flag]])
    tgt0 = multiply([tgt,seg_tgt[flag]])
    x_in = concatenate([src0, tgt0])
    
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt,src_seg, tgt_seg], outputs=[x])


def flow_sum(x):
    return (x[1]+x[2]+x[3]+x[0]);

def mask_gen(tgt_seg):
    tgt_seg1 = trf_resize(tgt_seg, 2)
    one = tf.ones_like(tgt_seg1)
    zero = tf.zeros_like(tgt_seg1)
    mask0 = tf.where(tf.equal(tgt_seg1,0), x=one, y=zero)
    mask1 = tf.where(tf.equal(tgt_seg1,1), x=one, y=zero)
    mask2 = tf.where(tf.equal(tgt_seg1,2), x=one, y=zero)
    mask3 = tf.where(tf.equal(tgt_seg1,3), x=one, y=zero)
    return [mask0,mask1,mask2,mask3];


def DDIR(vol_size, enc_nf, dec_nf, int_steps=7, use_miccai_int=False, indexing='ij', bidir=False, vel_resize=1/2):
    """
    architecture for probabilistic diffeomoprhic VoxelMorph presented in the MICCAI 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    The stationary velocity field operates in a space (0.5)^3 of vol_size for computational reasons.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6, see unet function.
    :param use_miccai_int: whether to use the manual miccai implementation of scaling and squaring integration
            note that the 'velocity' field outputted in that case was 
            since then we've updated the code to be part of a flexible layer. see neuron.layers.VecInt
            **This param will be phased out (set to False behavior)**
    :param int_steps: the number of integration steps
    :param indexing: xy or ij indexing. we recommend ij indexing if training from scratch. 
            miccai 2018 runs were done with xy indexing.
            **This param will be phased out (set to 'ij' behavior)**
    :return: the keras model
    """    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims


    src = Input(shape=[*vol_size, 1])
    tgt = Input(shape=[*vol_size, 1])
    src_segment = Input(shape=[*vol_size, 1])
    tgt_segment = Input(shape=[*vol_size, 1])

    mask0,mask1,mask2,mask3 = Lambda(mask_gen1)(src_segment)
    mask4,mask5,mask6,mask7 = Lambda(mask_gen1)(tgt_segment)

    # get unet
    unet_model0 = multi_channel_unet(vol_size, enc_nf, dec_nf,src = src,tgt = tgt,src_seg= src_segment,tgt_seg = tgt_segment, full_size=False,flag = 0)
    unet_model1 = multi_channel_unet(vol_size, enc_nf, dec_nf,src = src,tgt = tgt,src_seg= src_segment,tgt_seg = tgt_segment, full_size=False,flag = 1)
    unet_model2 = multi_channel_unet(vol_size, enc_nf, dec_nf,src = src,tgt = tgt,src_seg= src_segment,tgt_seg = tgt_segment, full_size=False,flag = 2)
    unet_model3 = multi_channel_unet(vol_size, enc_nf, dec_nf,src = src,tgt = tgt,src_seg= src_segment,tgt_seg = tgt_segment, full_size=False,flag = 3)
    

    x_out0 = unet_model0.outputs[-1]
    x_out1 = unet_model1.outputs[-1]
    x_out2 = unet_model2.outputs[-1]
    x_out3 = unet_model3.outputs[-1]

    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    #z0
    flow_mean0 = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow0')(x_out0)
    flow_log_sigma0 = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10),
                            name='log_sigma0')(x_out0)
    flow_params0 = concatenate([flow_mean0, flow_log_sigma0])
    flow0 = Sample(name="z_sample0")([flow_mean0, flow_log_sigma0])
    flow0 = nrn_layers.VecInt(method='ss', name='flow-int0', int_steps=int_steps)(flow0)
    flow0 = trf_resize(flow0, vel_resize, name='diffflow0')
    #z1
    flow_mean1 = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow1')(x_out1)
    flow_log_sigma1 = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10),
                            name='log_sigma1')(x_out1)
    flow_params1 = concatenate([flow_mean1, flow_log_sigma1])
    flow1 = Sample(name="z_sample1")([flow_mean1, flow_log_sigma1])
    flow1 = nrn_layers.VecInt(method='ss', name='flow-int1', int_steps=int_steps)(flow1)
    flow1 = trf_resize(flow1, vel_resize, name='diffflow1')
    
    
    #z2
    flow_mean2 = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow2')(x_out2)
    flow_log_sigma2 = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10),
                            name='log_sigma2')(x_out2)
    flow_params2 = concatenate([flow_mean2, flow_log_sigma2])
    flow2 = Sample(name="z_sample2")([flow_mean2, flow_log_sigma2])
    flow2 = nrn_layers.VecInt(method='ss', name='flow-int2', int_steps=int_steps)(flow2)
    flow2 = trf_resize(flow2, vel_resize, name='diffflow2')
    
    #z3
    flow_mean3 = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow3')(x_out3)
    flow_log_sigma3 = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10),
                            name='log_sigma3')(x_out3)
    flow_params3 = concatenate([flow_mean3, flow_log_sigma3])
    flow3 = Sample(name="z_sample3")([flow_mean3, flow_log_sigma3])
    flow3 = nrn_layers.VecInt(method='ss', name='flow-int3', int_steps=int_steps)(flow3)
    flow3 = trf_resize(flow3, vel_resize, name='diffflow3')

    flow0 = multiply([flow0,mask0])
    flow1 = multiply([flow1,mask1])
    flow2 = multiply([flow2,mask2])
    flow3 = multiply([flow3,mask3])
    

    flow = Lambda(flow_sum, name='diffflow')([flow0, flow1, flow2, flow3])


    if bidir:
        neg_flow = trf_resize(neg_flow, vel_resize, name='neg_diffflow')

    # transform
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    y_seg = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing)([src_segment, flow])
    if bidir:
        y_tgt = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([tgt, neg_flow])

    # prepare outputs and losses
    outputs = [y, y_seg,flow_params0, flow_params1, flow_params2, flow_params3]
    if bidir:
        outputs = [y, y_tgt, flow_params]

    # build the model
    return Model(inputs=[src, tgt,src_segment,tgt_segment], outputs=outputs)



def mask_gen1(tgt_seg):
    tgt_seg1 = tgt_seg
    one = tf.ones_like(tgt_seg1)
    zero = tf.zeros_like(tgt_seg1)
    mask0 = tf.where(tf.equal(tgt_seg1,0), x=one, y=zero)
    mask1 = tf.where(tf.equal(tgt_seg1,1), x=one, y=zero)
    mask2 = tf.where(tf.equal(tgt_seg1,2), x=one, y=zero)
    mask3 = tf.where(tf.equal(tgt_seg1,3), x=one, y=zero)
    return [mask0,mask1,mask2,mask3];



def nn_trf(vol_size, indexing='xy'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    # note the nearest neighbour interpolation method
    # note xy indexing because Guha's original code switched x and y dimensions
    nn_output = nrn_layers.SpatialTransformer(interp_method='nearest', indexing=indexing) #linear  nearest
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)



########################################################
# Helper functions
########################################################

def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = keras.layers.LeakyReLU(alpha=0.3)(x_out)
    #x_out = keras.layers.ReLU(x_out)
    return x_out

def conv_block1(x_in, nf, strides=1, name=''):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides, name = name)(x_in)
    x_out = keras.layers.LeakyReLU(alpha=0.3)(x_out)
    #x_out = keras.layers.ReLU(x_out)
    return x_out

def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


def trf_resize(trf, vel_resize, name='flow'):
    if vel_resize > 1:
        trf = nrn_layers.Resize(1/vel_resize, name=name+'_tmp')(trf)
        return Rescale(1 / vel_resize, name=name)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = Rescale(1 / vel_resize, name=name+'_tmp')(trf)
        return  nrn_layers.Resize(1/vel_resize, name=name)(trf)


class Sample(Layer):
    """ 
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Rescale(Layer):
    """ 
    Keras layer: rescale data by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize 

    def compute_output_shape(self, input_shape):
        return input_shape

class RescaleDouble(Rescale):
    def __init__(self, **kwargs):
        self.resize = 2
        super(RescaleDouble, self).__init__(self.resize, **kwargs)

class ResizeDouble(nrn_layers.Resize):
    def __init__(self, **kwargs):
        self.zoom_factor = 2
        super(ResizeDouble, self).__init__(self.zoom_factor, **kwargs)


class LocalParamWithInput(Layer):
    """ 
    The neuron.layers.LocalParam has an issue where _keras_shape gets lost upon calling get_output :(
        tried using call() but this requires an input (or i don't know how to fix it)
        the fix was that after the return, for every time that tensor would be used i would need to do something like
        new_vec._keras_shape = old_vec._keras_shape

        which messed up the code. Instead, we'll do this quick version where we need an input, but we'll ignore it.

        this doesn't have the _keras_shape issue since we built on the input and use call()
    """

    def __init__(self, shape, my_initializer='RandomNormal', mult=1.0, **kwargs):
        self.shape=shape
        self.initializer = my_initializer
        self.biasmult = mult
        super(LocalParamWithInput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.shape,  # input_shape[1:]
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalParamWithInput, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # want the x variable for it's keras properties and the batch.
        b = 0*K.batch_flatten(x)[:,0:1] + 1
        params = K.expand_dims(K.flatten(self.kernel * self.biasmult), 0)
        z = K.reshape(K.dot(b, params), [-1, *self.shape])
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape)
