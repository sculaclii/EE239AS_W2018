import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

###############################################################################################

# New class created for optimization of the neural network to target 65% validation accuracy
class ThreeLayerConvNet(object):
  # Data minibatches (N, C, H, W)
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, num_convL=1, num_affineL=2,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim (C, H, W)
    - num_filters
    - filter_size
    - num_convL: number of conv-relu-pool in the layer stack. Max: (H||W)/2**# >= filter_size
    - num_affineL: total number of affine layers after conv layer stack
    - hidden_dim: hidden_dim of each affine layer
    - num_classes: dim(scores)
    - weight_scale: st.d. for weight initialization
    - reg: L2 reg coefficient
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_affineL= num_affineL
    self.num_convL= num_convL
    self.filter_size= filter_size

    # [conv-relu-pool]xN - [affine]xM - softmax

    C, H, W= input_dim
    Hdim= hidden_dim
    Cls= num_classes
    F= num_filters

    # assume Stride=1 and pad=(filter_size-1)//2, and thus dim(conv_out)=dim(conv_in)
    
    Fin=C # number of channels of input image is C(colors). After 1st conv layer images will have F channels
    for i in range(num_convL):
        self.params['W'+str(i+1)]= weight_scale*np.random.randn(F, Fin, filter_size, filter_size)
        self.params['b'+str(i+1)]= np.zeros(F)
        if self.use_batchnorm:
            self.params['gamma'+str(i+1)]= np.ones(F)
            self.params['beta'+str(i+1)]= np.zeros(F)
        Fin=F
    
    Hdim_in= F*H*W//2**(num_convL*2) #image resolution is disected by 2 with each conv max pool
    # after first affine layer, all remaining Hdim_in= Hdim
    for i in range(num_convL, num_convL+num_affineL-1):
        self.params['W'+str(i+1)]= weight_scale*np.random.randn(Hdim_in,Hdim)
        self.params['b'+str(i+1)]= np.zeros(Hdim)
        if self.use_batchnorm:
            self.params['gamma'+str(i+1)]= np.ones(Hdim)
            self.params['beta'+str(i+1)]= np.zeros(Hdim)
        Hdim_in=Hdim
    i= num_convL+num_affineL-1
    self.params['W'+str(i+1)]= weight_scale*np.random.randn(Hdim_in, Cls)
    self.params['b'+str(i+1)]= np.zeros(Cls)

    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_convL+self.num_affineL-1)]

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):

    # conv_param for the forward pass through the convolutional layer
    conv_param = {'stride': 1, 'pad': (self.filter_size - 1) // 2}
    # pool_param for forward pass through the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    mode = 'test' if y is None else 'train'
    if self.use_batchnorm:
        for bn_param in self.bn_params:
            bn_param[mode] = mode

    # [conv-relu-pool-batchnorm]xN - [affine-relu-batchnorm]x(M-1) - affine - softmax
    
    caches= [] # list of caches from each layer
    bcaches= [] # list of caches for each batch norm layer
    layer_in= X # input to 1st conv layer is the raw image, later: input= output of prev layers

    # Forward through conv-relu-pool stacks
    for i in range(self.num_convL):
        out_temp, cache_temp= conv_relu_pool_forward(layer_in, self.params['W'+str(i+1)], \
                                                     self.params['b'+str(i+1)], conv_param, pool_param)
        caches.append(cache_temp)
        if self.use_batchnorm:
            out_temp, bcache_temp= spatial_batchnorm_forward(out_temp, self.params['gamma'+str(i+1)], \
                                           self.params['beta'+str(i+1)], self.bn_params[i])
            bcaches.append(bcache_temp)
        layer_in=out_temp
    
    # Forward through affine-relu stacks
    for i in range(self.num_convL, self.num_convL+self.num_affineL-1):
        out_temp, cache_temp= affine_relu_forward(layer_in, self.params['W'+str(i+1)], \
                                                  self.params['b'+str(i+1)])
        caches.append(cache_temp)
        if self.use_batchnorm:
            out_temp, bcache_temp= batchnorm_forward(out_temp, self.params['gamma'+str(i+1)], \
                                       self.params['beta'+str(i+1)], self.bn_params[i])
            bcaches.append(bcache_temp)
        layer_in=out_temp
    
    # Forward through last affine layer (output= scores)
    i= self.num_affineL+self.num_convL-1
    out_temp, cache_temp= affine_forward(layer_in, self.params['W'+str(i+1)], \
                                     self.params['b'+str(i+1)])
    caches.append(cache_temp)
    scores= out_temp

    if y is None:
      return scores
    
    loss, grads = 0, {}
    douts= []
    # Backprop through top softmax layer & calc loss
    loss, dout_temp = softmax_loss(out_temp, y)
    for i in range(self.num_convL+self.num_affineL):
        loss+= 0.5*self.reg*np.sum(np.square(self.params['W'+str(i+1)]))

    # Backprop through top affine layer
    i= self.num_convL+self.num_affineL-1
    dout_temp, grads['W'+str(i+1)], grads['b'+str(i+1)]= \
                            affine_backward(dout_temp, caches.pop())
    
    # Backprop through the remaining affine-relu-batchnorm layers
    for i in reversed(range(self.num_convL, self.num_convL+self.num_affineL-1)):
        if self.use_batchnorm:
            dout_temp, grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = \
                                        batchnorm_backward(dout_temp, bcaches.pop())
        dout_temp, grads['W'+str(i+1)], grads['b'+str(i+1)]= \
                            affine_relu_backward(dout_temp, caches.pop())
        
    # Backprop through the conv-relu-pool layers
    for i in reversed(range(self.num_convL)):
        if self.use_batchnorm:
            dout_temp, grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = \
                                spatial_batchnorm_backward(dout_temp, bcaches.pop())
        dout_temp, grads['W'+str(i+1)], grads['b'+str(i+1)]= \
                            conv_relu_pool_backward(dout_temp, caches.pop())

    return loss, grads

##################################################################################


""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class OldStatic_ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    #---------->conv-relu-2x2 max pool--affine----relu-affine-softmax->
    #(N,C,H,W)->-----(N,F,H,W)--(N,F,H/2,W,2)-(N,H)--------(N,Cl)
    C, H, W= input_dim
    Hdim= hidden_dim
    Cls= num_classes
    F= num_filters
    
    self.params['W1']= weight_scale*np.random.randn(F, C, filter_size, filter_size)
    self.params['b1']= np.zeros(F)
    self.params['W2']= weight_scale*np.random.randn(F*H//2*W//2,Hdim)
    self.params['b2']= np.zeros(Hdim)
    self.params['W3']= weight_scale*np.random.randn(Hdim,Cls)
    self.params['b3']= np.zeros(Cls)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    #---------->conv-relu-2x2 max pool--affine----relu-affine-softmax->
    #(N,C,H,W)->-----(N,F,H,W)--(N,F,H/2,W,2)-(N,H)--------(N,Cl)-->(L)
    out_c1, cache_c1= conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out_a2, cache_a2= affine_relu_forward(out_c1, W2, b2)
    out_a3, cache_a3= affine_forward(out_a2, W3, b3)
    scores= out_a3
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    loss, dout_a3 = softmax_loss(out_a3, y)
    for w in [W1, W2, W3]:
        loss+= 0.5*self.reg*np.sum(w*w)
    ###### Consider if reshaping is needed below: #######
    dout_a2, dW3, db3= affine_backward(dout_a3, cache_a3)
    dout_c1, dW2, db2= affine_relu_backward(dout_a2, cache_a2)
    dX, dW1, db1= conv_relu_pool_backward(dout_c1, cache_c1)
    grads['W1']=dW1; grads['b1']=db1
    grads['W2']=dW2; grads['b2']=db2
    grads['W3']=dW3; grads['b3']=db3
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads

