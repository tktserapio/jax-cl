from typing import Callable, List, Optional, Type, Union

from flax import nnx
from functools import partial
import jax
from jax import numpy as jnp, Array

# TODO: 
# - Add SequentialKeywordArgs so that we can use it for CBP feature_list forward passing

class BasicBlock(nnx.Module):
  def __init__(
    self, in_planes: int, out_planes: int, do_downsample: bool = False, *, rngs: nnx.Rngs
  ):
    strides = (2, 2) if do_downsample else (1, 1)
    self.conv1_bn1 = nnx.Sequential(
      nnx.Conv(
        in_planes, out_planes, kernel_size=(3, 3), strides=strides,
        padding="SAME", use_bias=True, 
        kernel_init=nnx.initializers.kaiming_normal(), # kaiming for relu
        bias_init=nnx.initializers.constant(0.0), # zero bias
        rngs=rngs,
      ),
      nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, 
      scale_init=nnx.initializers.constant(1.0), # bn weight = 1
      bias_init=nnx.initializers.constant(0.0), # bn bias = 0
      rngs=rngs),
    )
    self.conv2_bn2 = nnx.Sequential(
      nnx.Conv(
        out_planes, out_planes, kernel_size=(3, 3), strides=(1, 1),
        padding="SAME", use_bias=True, 
        kernel_init=nnx.initializers.kaiming_normal(), # kaiming for relu
        bias_init=nnx.initializers.constant(0.0), # zero bias
        rngs=rngs,
      ),
      nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, 
      scale_init=nnx.initializers.constant(1.0), # bn weight = 1
      bias_init=nnx.initializers.constant(0.0), # bn bias = 0
      rngs=rngs),
    )

    if do_downsample:
      self.downsample = nnx.Sequential( 
        nnx.Conv(in_planes, out_planes, kernel_size=(1, 1), strides=strides, 
        padding="VALID", use_bias=True, 
        kernel_init=nnx.initializers.kaiming_normal(), # kaiming for relu
        bias_init=nnx.initializers.constant(0.0), # zero bias
        rngs=rngs, 
        ), 
        nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, 
        scale_init=nnx.initializers.constant(1.0), # bn weight = 1
        bias_init=nnx.initializers.constant(0.0), # bn bias = 0
        rngs=rngs),
      )
    else:
      self.downsample = None

  def __call__(self, x: jax.Array):

    identity = x

    out = self.conv1_bn1(x)
    out = nnx.relu(out)

    out = self.conv2_bn2(out)
    # out = nnx.relu(out) # pytorch doesn't relu here, so we remove it
    
    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = nnx.relu(out)

    return out


class ResNet18(nnx.Module):
  def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
    self.num_classes = num_classes
    self.conv1_bn1 = nnx.Sequential(
      nnx.Conv(
        3, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
        use_bias=True, 
        kernel_init=nnx.initializers.kaiming_normal(), 
        bias_init=nnx.initializers.constant(0.0), # zero bias
        rngs=rngs,
      ),
      nnx.BatchNorm(64, momentum=0.9, epsilon=1e-5, 
      scale_init=nnx.initializers.constant(1.0), # bn weight = 1
      bias_init=nnx.initializers.constant(0.0), # bn bias = 0
      rngs=rngs),
    )
    self.layer1 = nnx.Sequential(
      BasicBlock(64, 64, rngs=rngs), 
      BasicBlock(64, 64, rngs=rngs),
    )
    self.layer2 = nnx.Sequential(
      BasicBlock(64, 128, do_downsample=True, rngs=rngs), 
      BasicBlock(128, 128, rngs=rngs),
    )
    self.layer3 = nnx.Sequential(
      BasicBlock(128, 256, do_downsample=True, rngs=rngs), 
      BasicBlock(256, 256, rngs=rngs),
    )
    self.layer4 = nnx.Sequential(
      BasicBlock(256, 512, do_downsample=True, rngs=rngs), 
      BasicBlock(512, 512, rngs=rngs),
    )
    self.fc = nnx.Linear(512, self.num_classes, 
      kernel_init=nnx.initializers.kaiming_normal(), # kaiming for relu
      bias_init=nnx.initializers.constant(0.0), # zero bias
      rngs=rngs
    )

  def __call__(self, x: jax.Array):
    x = self.conv1_bn1(x)
    x = nnx.relu(x)
    # x = nnx.max_pool(x, (2, 2), strides=(2, 2))

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = nnx.avg_pool(x, (x.shape[1], x.shape[2]))
    x = x.reshape((x.shape[0], -1))
    x = self.fc(x)
    return x

def build_resnet18(num_classes: int, *, rngs: nnx.Rngs):
  return ResNet18(num_classes=num_classes, rngs=rngs)