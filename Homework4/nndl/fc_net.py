import numpy as np
import pdb

from .layers import *
from .layer_utils import *

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize all parameters of the network in the self.params dictionary.
    #   The weights and biases of layer 1 are W1 and b1; and in general the 
    #   weights and biases of layer i are Wi and bi. The
    #   biases are initialized to zero and the weights are initialized
    #   so that each parameter has mean 0 and standard deviation weight_scale.
    #
    #   BATCHNORM: Initialize the gammas of each layer to 1 and the beta
    #   parameters to zero.  The gamma and beta parameters for layer 1 should
    #   be self.params['gamma1'] and self.params['beta1'].  For layer 2, they
    #   should be gamma2 and beta2, etc. Only use batchnorm if self.use_batchnorm 
    #   is true and DO NOT batch normalize the output scores.
    # ================================================================ #
    inH=input_dim
    # outH: output dim of hidden layer, inH: input dim of hidden layer
    for i, outH in enumerate(hidden_dims+[num_classes]):
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(inH, outH)
        self.params['b'+str(i+1)] = np.zeros(outH)
        inH=outH
    if self.use_batchnorm:
        for i, outH in enumerate(hidden_dims):
            self.params['gamma'+str(i+1)]= np.ones(outH)
            self.params['beta'+str(i+1)]= np.zeros(outH)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the FC net and store the output
    #   scores as the variable "scores".
    #
    #   BATCHNORM: If self.use_batchnorm is true, insert a bathnorm layer
    #   between the affine_forward and relu_forward layers.  You may
    #   also write an affine_batchnorm_relu() function in layer_utils.py.
    #
    #   DROPOUT: If dropout is non-zero, insert a dropout layer after
    #   every ReLU layer.
    # ================================================================ #
    a_in=X; a_caches=[]; r_caches=[]; b_caches=[]; drop_caches=[]

    for i in range(self.num_layers-1):
        # affine layer [i] (index 0 count):
        a_out, a_cache = affine_forward(a_in,self.params['W'+str(i+1)],self.params['b'+str(i+1)])
        a_caches.append(a_cache)
        # batchnorm layer [i] (index 0 count):
        if self.use_batchnorm:
            b_out, b_cache = batchnorm_forward( a_out, self.params['gamma'+str(i+1)], \
                                               self.params['beta'+str(i+1)], self.bn_params[i] )
            b_caches.append(b_cache)
        else:
            b_out= a_out
        # relu layer [i] (index 0 count):
        r_out, r_cache = relu_forward(b_out)
        r_caches.append(r_cache)
        # dropout layer [i] (index 0 count)
        if self.use_dropout:
            drop_out, drop_cache= dropout_forward(r_out, self.dropout_param)
            drop_caches.append(drop_cache)
        else:
            drop_out= r_out
        a_in= drop_out
    # pass through top affine layer
    a_out, a_cache = affine_forward(a_in,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
    a_caches.append(a_cache)
    # scores = output of the last affine layer
    scores=a_out
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backwards pass of the FC net and store the gradients
    #   in the grads dict, so that grads[k] is the gradient of self.params[k]
    #   Be sure your L2 regularization includes a 0.5 factor.
    #
    #   BATCHNORM: Incorporate the backward pass of the batchnorm.
    #
    #   DROPOUT: Incorporate the backward pass of dropout.
    # ================================================================ #
    # calculate loss; input= output of the top affine layer
    loss, da_out = softmax_loss(a_out, y)
    # add derivative of the regularization terms in each affine layer to loss:
    for i in range(self.num_layers):
        loss+= self.reg*0.5*np.sum(np.square(self.params['W'+str(i+1)]))

    # calculate gradients
    
    # backprop through top affine layer:
    dd_out, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)]= \
        affine_backward(da_out, a_caches[self.num_layers-1])
    # add regularization term of this affine layer:
    grads['W'+str(self.num_layers)]+=self.reg*self.params['W'+str(self.num_layers)]
    
    for i in reversed( range(self.num_layers-1) ):
        # backprop i-th dropout layer (index 0 count) if enabled
        if self.use_dropout:
            dr_out= dropout_backward(dd_out, drop_caches[i])
        else:
            dr_out= dd_out
        # backprop i-th relu layer (index 0 count):
        db_out = relu_backward(dr_out, r_caches[i])
        # backprop batchnorm layer below the relu layer, if enabled:
        if self.use_batchnorm:
            da_out, grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = batchnorm_backward(db_out, b_caches[i])
        else:
            da_out= db_out
        # backprop through i-th affine layer (index 0 count):
        dd_out, grads['W'+str(i+1)], grads['b'+str(i+1)]= \
            affine_backward(da_out, a_caches[i])
        # add regularization term of this affine layer:    
        grads['W'+str(i+1)]+=self.reg*self.params['W'+str(i+1)]
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads
