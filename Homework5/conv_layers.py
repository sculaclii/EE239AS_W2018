import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W= x.shape
  F, _, HH, WW= w.shape
  # output dimensions:
  outH= 1+(H+2*pad-HH)//stride
  outW= 1+(W+2*pad-WW)//stride

  x_pad = np.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant')
  
  out= np.zeros((N, F, outH, outW))
  for n in range(N): # iterate over N image samples
      for f in range(F): # ... over F kernels of filter
          for outh in range(outH): # along image height
              for outw in range(outW): # along image width W
                  out[n, f, outh, outw]= \
                      np.sum( x_pad[n, :, outh*stride:outh*stride+HH, outw*stride:outw*stride+WW] \
                             * w[f, :, :, :] ) + b[f]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N, F, oH, oW= dout.shape
  F, C, wH, wW= w.shape
  N, C, xH, xW= x.shape
  _, _, pH, pW= xpad.shape
  S = stride

  # Calculate dw
  dw= np.zeros((F, C, wH, wW))
  for f in range(F):
      for c in range(C):
          for wh in range(wH):
              for ww in range(wW):
                  xpad_masked= xpad[:,c, wh:(wh+oH*S):S, ww:(ww+oW*S):S]
                  dw[f,c,wh,ww]=np.sum( xpad_masked*dout[:,f,:,:] )
  
  # Calculate db
  db= np.zeros(F)
  for f in range(F):
      db[f]= np.sum( dout[:,f,:,:] )
  
  # Calculate dx
  dxpad= np.zeros_like(xpad)
  for oh in range(oH):
      for ow in range(oW):
          for n in range(N):
              for f in range(F):
                  dxpad[n,:, oh:oh+wH, ow:ow+wW]+= w[f,:,:,:]*dout[n,f,oh,ow]
  dx= dxpad[:,:, pad:-pad, pad:-pad].reshape(x.shape)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, xH, xW= x.shape
  pH, pW= (pool_param['pool_height'], pool_param['pool_width'])
  S= pool_param['stride']
  # output dimensions:
  oH= 1+(xH-pH)//S
  oW= 1+(xW-pW)//S

  out= np.zeros((N, C, oH, oW))
  for n in range(N):
      for c in range(C):
          for oh in range(oH):
              for ow in range(oW):
                  out[n, c, oh, ow]= \
                      np.max( x[n, c, oh*S:oh*S+pH, ow*S:ow*S+pW] )
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, xH, xW= x.shape
  pH, pW= (pool_param['pool_height'], pool_param['pool_width'])
  S= pool_param['stride']
  N, C, oH, oW= dout.shape

  dx= np.zeros_like(x)
  for n in range(N):
      for c in range(C):
          for oh in range(oH):
              for ow in range(oW):
                  x_pool= x[n, c, oh*S:oh*S+pH, ow*S:ow*S+pW]
                  (maxh, maxw) = np.unravel_index(np.argmax(x_pool), x_pool.shape)
                  maxh+= oh*S; maxw+= ow*S
                  dx[n, c, maxh, maxw]= dout[n,c,oh,ow]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N,C,H,W= x.shape
  # first reshape x: (N, C, H, W)->(N*H*W, C) in 2 steps:
  xresh= np.moveaxis(x,1,-1) # (N, C, H, W)->(N, H, W, C)
  xresh= xresh.reshape(N*H*W, C)
  out, cache= batchnorm_forward(xresh, gamma, beta, bn_param)
  out= out.reshape(N,H,W,C)
  out= np.moveaxis(out, -1, 1)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N,C,H,W= dout.shape
  # first reshape dout: (N, C, H, W)->(N*H*W, C) in 2 steps:
  dout_resh= np.moveaxis(dout,1,-1) # (N, C, H, W)->(N, H, W, C)
  dout_resh= dout_resh.reshape(N*H*W, C)

  dx, dgamma, dbeta= batchnorm_backward(dout_resh, cache)
  
  dx= dx.reshape(N,H,W,C)
  dx= np.moveaxis(dx, -1, 1)
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta