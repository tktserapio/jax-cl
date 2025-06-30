from typing import Callable, List, Optional, Type, Union

from flax import nnx
from functools import partial
import jax
from jax import numpy as jnp, Array

# import torch
# import torch.nn as nn
# from torch import Tensor

# from torchvision.utils import _log_api_usage_once

""" 
This is a modified version of torchvision's code for instantiating resnets. Here's a list of the changes made to the 
source code:
    - All convolutional layers now have bias set to True, where they were original set to False.
    - Removed the first maxpool layers so that input stays somewhat large.
    - Layer conv1 in the ResNet class has kernel size set to 3 and stride set to 1, where they were originally 7 and 2,
      respectively. 
    - Forward calls have a feature list argument to store the features of the network. This is only used for continual 
      backprop and doesn't affect the output of the network.
To see the source code, go to: torchvision.models.resnet (for torchvision==0.15.1)
"""

class SequentialWithKeywordArguments(nnx.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def __call__(self, input, **kwargs):
        for layer in self.layers:
            if hasattr(layer, '__call__'):
                # Check if the layer accepts keyword arguments (like our custom blocks)
                # by checking if it has a training or feature_list parameter
                try:
                    import inspect
                    sig = inspect.signature(layer.__call__)
                    param_names = set(sig.parameters.keys())
                    if ('training' in param_names or 'feature_list' in param_names or 
                        any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())):
                        input = layer(input, **kwargs)
                    else:
                        input = layer(input)
                except Exception as e:
                    # If we can't inspect or other error, try without kwargs first
                    try:
                        input = layer(input)
                    except TypeError:
                        # If that fails, try with kwargs as last resort
                        try:
                            input = layer(input, **kwargs)
                        except Exception as final_e:
                            print(f"ERROR in SequentialWithKeywordArguments: Layer {layer} failed with {final_e}")
                            raise final_e
            else:
                input = layer(input)
        return input

# class SequentialWithKeywordArguments(torch.nn.Sequential):
#     """
#     Sequential module that allows the use of keyword arguments in the forward pass
#     """
#     def forward(self, input, **kwargs):
#         for module in self:
#             input = module(input, **kwargs)
#         return input


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, *, rngs: nnx.Rngs) -> nnx.Conv:
    """3x3 convolution with padding from torchvision.models.resnet but bias is set to True"""
    return nnx.Conv(
        in_features=in_planes,
        out_features=out_planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding=((dilation, dilation), (dilation, dilation)),
        feature_group_count=groups,
        use_bias=True,
        kernel_dilation=(dilation, dilation),
        kernel_init=nnx.initializers.kaiming_normal(), # use nnx initializers
        rngs=rngs
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, *, rngs: nnx.Rngs) -> nnx.Conv:
    """1x1 convolution"""
    return nnx.Conv(
        in_features=in_planes, 
        out_features=out_planes, 
        kernel_size=(1, 1), 
        strides=(stride, stride), 
        use_bias=True,
        kernel_init=nnx.initializers.kaiming_normal(), # use nnx initializers
        rngs=rngs
    )


class BasicBlock(nnx.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nnx.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nnx.Module]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Split RNG keys for each layer
        conv1_key, bn1_key, conv2_key, bn2_key = jax.random.split(rngs.default(), 4)
        
        # self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride=stride, rngs=nnx.Rngs(conv1_key))
        self.bn1 = norm_layer(planes, rngs=nnx.Rngs(bn1_key))
        self.relu = nnx.relu
        self.conv2 = conv3x3(planes, planes, rngs=nnx.Rngs(conv2_key))
        self.bn2 = norm_layer(planes, rngs=nnx.Rngs(bn2_key))
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: Array, feature_list: list = None, training: bool = True) -> Array:
        """
        Forward pass through the block
        :param x: input to the resnet block
        :param feature_list: list to store the features of the network
        :param training: whether to use training mode for BatchNorm
        :return: output of the block
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, use_running_average=not training)  # Fixed: explicit training mode
        out = self.relu(out)

        if feature_list is not None: feature_list.append(out)

        out = self.conv2(out)
        out = self.bn2(out, use_running_average=not training)  # Fixed: explicit training mode

        if self.downsample is not None:
            identity = self.downsample(x, training=training)  # Pass training mode to downsample

        out += identity
        out = self.relu(out)

        if feature_list is not None: feature_list.append(out)

        return out


class ResNet(nnx.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nnx.Module]] = None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        # _log_api_usage_once(self) 
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        # Split RNG keys for each layer in the main ResNet
        conv1_key, bn1_key, layer1_key, layer2_key, layer3_key, layer4_key, fc_key = jax.random.split(rngs.default(), 7)
        
        # Kaiming initialization for conv layers (note the function call is done by nnx)
        
        self.conv1 = nnx.Conv(
            in_features=3, 
            out_features=self.inplanes, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding=((1, 1), (1, 1)), 
            use_bias=True,
            kernel_init=nnx.initializers.kaiming_normal(),
            rngs=nnx.Rngs(conv1_key)
        )
        self.bn1 = norm_layer(self.inplanes, rngs=nnx.Rngs(bn1_key))
        self.relu = nnx.relu
        self.layer1 = self._make_layer(block, 64, layers[0], rngs=nnx.Rngs(layer1_key))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], rngs=nnx.Rngs(layer2_key))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], rngs=nnx.Rngs(layer3_key))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], rngs=nnx.Rngs(layer4_key))
        
        # JAX equivalent of AdaptiveAvgPool2d - global average pooling
        # self.output_pool will be implemented in __call__ method using jax.numpy
        
        # Linear layer with kaiming initialization for output layer
        self.fc = nnx.Linear(
            512 * block.expansion, 
            num_classes,
            kernel_init=nnx.initializers.kaiming_normal(),  # Use nnx initializers
            rngs=nnx.Rngs(fc_key)
        )

        # Zero-initialize the last BN in each residual branch in JAX
        # This will be handled differently - we'll need to manually set these after creation
        if zero_init_residual:
            self._zero_init_residual_weights()

    def _zero_init_residual_weights(self):
        """Zero-initialize the last BN in each residual branch for JAX/Flax"""
        # This needs to be called after all layers are created
        # We'll traverse the layers and zero out the final BN weights
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if hasattr(layer, 'layers'):  # Sequential-like structure
                for block in layer.layers:
                    if hasattr(block, 'bn2') and hasattr(block.bn2, 'scale'):
                        # In JAX/Flax BatchNorm, 'scale' is equivalent to PyTorch's 'weight'
                        block.bn2.scale.value = jnp.zeros_like(block.bn2.scale.value)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> SequentialWithKeywordArguments:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Split keys for downsample layers
            conv_key, bn_key = jax.random.split(rngs.default(), 2)
            downsample = SequentialWithKeywordArguments(
                conv1x1(self.inplanes, planes * block.expansion, stride, rngs=nnx.Rngs(conv_key)),
                norm_layer(planes * block.expansion, rngs=nnx.Rngs(bn_key)),
            )

        # Split keys for all blocks in this layer
        block_keys = jax.random.split(rngs.default(), blocks)
        
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, rngs=nnx.Rngs(block_keys[0])
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    rngs=nnx.Rngs(block_keys[i]),
                )
            )

        return SequentialWithKeywordArguments(*layers)

    def _forward_impl(self, x: Array, feature_list: list = None, training: bool = True) -> Array:
        """
        Forward pass for a resnet
        :param x: input to the network
        :param feature_list: optional list for storing the features of the network
        :param training: whether to use training mode for BatchNorm
        :return: output of the network
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not training)  # Fixed: explicit training mode
        x = self.relu(x)

        if feature_list is not None: feature_list.append(x)

        x = self.layer1(x, feature_list=feature_list, training=training)
        x = self.layer2(x, feature_list=feature_list, training=training)
        x = self.layer3(x, feature_list=feature_list, training=training)
        x = self.layer4(x, feature_list=feature_list, training=training)

        if feature_list is not None: feature_list.pop(-1)

        # Global average pooling (JAX equivalent of nn.AdaptiveAvgPool2d((1, 1)))
        # With NHWC format, spatial dimensions are (1, 2) for (H, W)
        x = jnp.mean(x, axis=(1, 2))  # Average over spatial dimensions (H, W)
        
        if feature_list is not None: feature_list.append(x)

        x = self.fc(x)

        return x

    def __call__(self, x: Array, feature_list: list = None, training: bool = True) -> Array:
        return self._forward_impl(x, feature_list, training)


def build_resnet18(num_classes: int, norm_layer, *, rngs: nnx.Rngs):
    """
    :param num_classes: number of classes for the classification problem
    :param norm_layer: type of normalization. Options: [nnx.BatchNorm, nnx.Identity]
    :param rngs: random number generators for JAX/Flax
    :return: an instance of ResNet with the correct number of layers for ResNet18
    """
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], norm_layer=norm_layer, num_classes=num_classes, rngs=rngs)


def kaiming_init_resnet_module(nn_module: nnx.Module):
    # This function is no longer used in JAX/Flax nnx approach
    # Initialization is handled at module creation time
    pass