from typing import Callable, List, Optional, Type, Union

from flax import nnx
from functools import partial
import jax
from jax import numpy as jnp, Array

# TODO (things that are in torchvision_modified_resnet): 
# CBP stuff:
# - Forward calls have a feature list argument to store the features of the network. This is only used for continual backprop and doesn't affect the output of the network.
# - SequentialKeywordArguments: A custom sequential module that allows passing keyword arguments to each layer in the sequence. This is used to pass the feature list to each block in the ResNet (for CBP)

class BasicBlock(nnx.Module):
    def __init__(self, in_planes: int, out_planes: int, do_downsample: bool = False, *, rngs: nnx.Rngs):
        stride = (2, 2) if do_downsample else (1, 1)
        self.conv1_bn1 = nnx.Sequential(
            nnx.Conv(
                in_planes, out_planes, kernel_size=(3, 3), strides=stride,
                padding="SAME", use_bias=True, rngs=rngs,
            ),
            nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, rngs=rngs),
        )

        self.conv2_bn2 = nnx.Sequential(
            nnx.Conv(
                out_planes, out_planes, kernel_size=(3, 3), strides=(1, 1),
                padding="SAME", use_bias=True, rngs=rngs,
            ),
            nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, rngs=rngs),
        )

        if do_downsample:
            self.conv3_bn3 = nnx.Sequential(
                nnx.Conv(
                    in_planes, out_planes, kernel_size=(1, 1), strides=(2, 2), 
                    padding="VALID", use_bias=True, rngs=rngs,
                ), 
                nnx.BatchNorm(out_planes, momentum=0.9, epsilon=1e-5, rngs=rngs),
            )
        else:
            self.conv3_bn3 = lambda x: x
    
    def __call__(self, x: jax.Array):
        out = self.conv1_bn1(x)
        out = nnx.relu(out)
        
        out = self.conv2_bn2(out)
        out = nnx.relu(out)

        shortcut = self.conv3_bn3(x)
        out += shortcut
        out = nnx.relu(out)
        return out


class ResNet18(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.num_classes = num_classes
        
        self.conv1_bn1 = nnx.Sequential(
            nnx.Conv(
                3, 64, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                use_bias=False, rngs=rngs,
            ),
            nnx.BatchNorm(64, momentum=0.9, epsilon=1e-5, rngs=rngs),
        )
        
        self.layer1 = nnx.Sequential(
            BasicBlock(64, 64, rngs=rngs), BasicBlock(64, 64, rngs=rngs),
        )
        
        self.layer2 = nnx.Sequential(
            BasicBlock(64, 128, do_downsample=True, rngs=rngs), BasicBlock(128, 128, rngs=rngs),
        )
        
        self.layer3 = nnx.Sequential(
            BasicBlock(128, 256, do_downsample=True, rngs=rngs), BasicBlock(256, 256, rngs=rngs),
        )
        
        self.layer4 = nnx.Sequential(
            BasicBlock(256, 512, do_downsample=True, rngs=rngs), BasicBlock(512, 512, rngs=rngs),
        )
        
        self.fc = nnx.Linear(512, self.num_classes, rngs=rngs)
    def __call__(self, x: jax.Array):
        x = self.conv1_bn1(x)
        x = nnx.relu(x)
        # x = nnx.max_pool(x, (2, 2), strides=(2, 2)) # removed max pooling 

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = nnx.avg_pool(x, (x.shape[1], x.shape[2]))
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x

def build_resnet18(num_classes: int, norm_layer, *, rngs: nnx.Rngs):
    """
    :param num_classes: number of classes for the classification problem
    :param norm_layer: type of normalization. Options: [nnx.BatchNorm, nnx.Identity]
    :param rngs: random number generators for JAX/Flax
    :return: an instance of ResNet with the correct number of layers for ResNet18
    """
    return ResNet18(num_classes=num_classes, rngs=rngs)


# class BasicBlock(nnx.Module):
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nnx.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nnx.Module]] = None,
#         *,
#         rngs: nnx.Rngs,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nnx.BatchNorm
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
#         # Split RNG keys for each layer
#         conv1_key, bn1_key, conv2_key, bn2_key = jax.random.split(rngs.default(), 4)
        
#         # self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride=stride, rngs=nnx.Rngs(conv1_key))
#         self.bn1 = norm_layer(planes, rngs=nnx.Rngs(bn1_key))
#         self.relu = nnx.relu
#         self.conv2 = conv3x3(planes, planes, rngs=nnx.Rngs(conv2_key))
#         self.bn2 = norm_layer(planes, rngs=nnx.Rngs(bn2_key))
#         self.downsample = downsample
#         self.stride = stride

#     def __call__(self, x: Array, feature_list: list = None, training: bool = True) -> Array:
#         """
#         Forward pass through the block
#         :param x: input to the resnet block
#         :param feature_list: list to store the features of the network
#         :param training: whether to use training mode for BatchNorm
#         :return: output of the block
#         """
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out, use_running_average=not training)  # Fixed: explicit training mode
#         out = self.relu(out)

#         if feature_list is not None: feature_list.append(out)

#         out = self.conv2(out)
#         out = self.bn2(out, use_running_average=not training)  # Fixed: explicit training mode

#         if self.downsample is not None:
#             identity = self.downsample(x, training=training)  # Pass training mode to downsample

#         out += identity
#         out = self.relu(out)

#         if feature_list is not None: feature_list.append(out)

#         return out


# class ResNet(nnx.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock]],
#         layers: List[int],
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nnx.Module]] = None,
#         *,
#         rngs: nnx.Rngs,
#     ) -> None:
#         # _log_api_usage_once(self) 
#         if norm_layer is None:
#             norm_layer = nnx.BatchNorm
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
        
#         # Split RNG keys for each layer in the main ResNet
#         conv1_key, bn1_key, layer1_key, layer2_key, layer3_key, layer4_key, fc_key = jax.random.split(rngs.default(), 7)
        
#         # Kaiming initialization for conv layers (note the function call is done by nnx)
        
#         self.conv1 = nnx.Conv(
#             in_features=3, 
#             out_features=self.inplanes, 
#             kernel_size=(3, 3), 
#             strides=(1, 1), 
#             padding=((1, 1), (1, 1)), 
#             use_bias=True,
#             kernel_init=nnx.initializers.kaiming_normal(),
#             rngs=nnx.Rngs(conv1_key)
#         )
#         self.bn1 = norm_layer(self.inplanes, rngs=nnx.Rngs(bn1_key))
#         self.relu = nnx.relu
#         self.layer1 = self._make_layer(block, 64, layers[0], rngs=nnx.Rngs(layer1_key))
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], rngs=nnx.Rngs(layer2_key))
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], rngs=nnx.Rngs(layer3_key))
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], rngs=nnx.Rngs(layer4_key))
        
#         # JAX equivalent of AdaptiveAvgPool2d - global average pooling
#         # self.output_pool will be implemented in __call__ method using jax.numpy
        
#         # Linear layer with kaiming initialization for output layer
#         self.fc = nnx.Linear(
#             512 * block.expansion, 
#             num_classes,
#             kernel_init=nnx.initializers.kaiming_normal(),  # Use nnx initializers
#             rngs=nnx.Rngs(fc_key)
#         )

#         # Zero-initialize the last BN in each residual branch in JAX
#         # This will be handled differently - we'll need to manually set these after creation
#         if zero_init_residual:
#             self._zero_init_residual_weights()

#     def _zero_init_residual_weights(self):
#         """Zero-initialize the last BN in each residual branch for JAX/Flax"""
#         # This needs to be called after all layers are created
#         # We'll traverse the layers and zero out the final BN weights
#         for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
#             if hasattr(layer, 'layers'):  # Sequential-like structure
#                 for block in layer.layers:
#                     if hasattr(block, 'bn2') and hasattr(block.bn2, 'scale'):
#                         # In JAX/Flax BatchNorm, 'scale' is equivalent to PyTorch's 'weight'
#                         block.bn2.scale.value = jnp.zeros_like(block.bn2.scale.value)

#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#         *,
#         rngs: nnx.Rngs,
#     ) -> SequentialWithKeywordArguments:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             # Split keys for downsample layers
#             conv_key, bn_key = jax.random.split(rngs.default(), 2)
#             downsample = SequentialWithKeywordArguments(
#                 conv1x1(self.inplanes, planes * block.expansion, stride, rngs=nnx.Rngs(conv_key)),
#                 norm_layer(planes * block.expansion, rngs=nnx.Rngs(bn_key)),
#             )

#         # Split keys for all blocks in this layer
#         block_keys = jax.random.split(rngs.default(), blocks)
        
#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, rngs=nnx.Rngs(block_keys[0])
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                     rngs=nnx.Rngs(block_keys[i]),
#                 )
#             )

#         return SequentialWithKeywordArguments(*layers)

#     def _forward_impl(self, x: Array, feature_list: list = None, training: bool = True) -> Array:
#         """
#         Forward pass for a resnet
#         :param x: input to the network
#         :param feature_list: optional list for storing the features of the network
#         :param training: whether to use training mode for BatchNorm
#         :return: output of the network
#         """
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x, use_running_average=not training)  # Fixed: explicit training mode
#         x = self.relu(x)

#         if feature_list is not None: feature_list.append(x)

#         x = self.layer1(x, feature_list=feature_list, training=training)
#         x = self.layer2(x, feature_list=feature_list, training=training)
#         x = self.layer3(x, feature_list=feature_list, training=training)
#         x = self.layer4(x, feature_list=feature_list, training=training)

#         if feature_list is not None: feature_list.pop(-1)

#         # Global average pooling (JAX equivalent of nn.AdaptiveAvgPool2d((1, 1)))
#         # With NHWC format, spatial dimensions are (1, 2) for (H, W)
#         x = jnp.mean(x, axis=(1, 2))  # Average over spatial dimensions (H, W)
        
#         if feature_list is not None: feature_list.append(x)

#         x = self.fc(x)

#         return x

#     def __call__(self, x: Array, feature_list: list = None, training: bool = True) -> Array:
#         return self._forward_impl(x, feature_list, training)


# def build_resnet18(num_classes: int, norm_layer, *, rngs: nnx.Rngs):
#     """
#     :param num_classes: number of classes for the classification problem
#     :param norm_layer: type of normalization. Options: [nnx.BatchNorm, nnx.Identity]
#     :param rngs: random number generators for JAX/Flax
#     :return: an instance of ResNet with the correct number of layers for ResNet18
#     """
#     return ResNet(BasicBlock, layers=[2, 2, 2, 2], norm_layer=norm_layer, num_classes=num_classes, rngs=rngs)


# def kaiming_init_resnet_module(nn_module: nnx.Module):
#     # This function is no longer used in JAX/Flax nnx approach
#     # Initialization is handled at module creation time
#     pass