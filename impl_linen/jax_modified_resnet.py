from typing import Callable, List, Optional, Type, Union, Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

def conv3x3(planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv(
        features=planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding=((1, 1), (1, 1)),
        feature_group_count=groups,
        use_bias=True,
        kernel_dilation=(dilation, dilation),
        kernel_init=nn.initializers.kaiming_normal()
    )

def conv1x1(planes : int, stride : int = 1):
    return nn.Conv(
        features=planes, 
        kernel_size=(1, 1),
        strides=(stride, stride),
        use_bias=True, 
        kernel_init=nn.initializers.kaiming_normal()
    )

class BasicBlock(nn.Module):
    expansion: int = 1
    planes: int
    stride: int = 1
    groups: int = 1
    base_width: int = 64
    dilation: int = 1
    norm_layer: Optional[Callable[..., nn.Module]] = None, 
    downsample: Optional[nn.Module] = None

    def setup(self):
        if norm_layer is None:
            norm_layer = nnx.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(planes=self.planes, stride=self.stride, groups=self.groups, dilation=self.dilation)
        self.bn1 = self.norm_layer()
        self.conv2 = conv3x3(planes=self.planes, stride=self.stride, groups=self.groups, dilation=self.dilation)
        self.bn2 = self.norm_layer()

    @nn.compact
    def __call__(self, x: Array, feature_list: List = None, training: bool = True):
        identity = x

        out = conv3x3(planes=self.planes, stride=self.stride, groups=self.groups, dilation=self.dilation)(x)
        out = self.bn1(out, use_running_average=not training)
        out = nn.relu(out)

        if feature_list is not None:
            feature_list.append(out)

        out = conv3x3(planes=self.planes, stride=self.stride, groups=self.groups, dilation=self.dilation)(out)
        out = self.bn2(out, use_running_average=not training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity 
        out = nn.relu(out)

        if feature_list is not None: feature_list.append(out)

        return out

class ResNet(nn.Module):

    block: Type[Union[BasicBlock]],
    layers: List[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: Optional[List[bool]] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,


    def setup(self):
        if norm_layer is None:
            norm_layer = nn.BatchNorm 
        self._norm_layer = norm_layer 

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)), use_bias=True, kernel_init=nn.initializers.kaiming_normal(), dilation=)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> SequentialWithKeywordArguments:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation 

        if dilate:
            self.dilation *= stride 
            stride = 1
        
        # TODO: implement downsampling 

        layers = []
    
    def __call__()