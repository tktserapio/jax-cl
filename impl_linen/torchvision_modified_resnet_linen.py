from typing import Callable, List, Optional, Type, Union, Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

class Sequential(nn.Module):
    """Simple Sequential container for a list of submodules."""
    layers: List[Any]

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for i, layer in enumerate(self.layers):
            # name each layer for parameter collection
            x = layer(x, *args, **kwargs)
        return x

def conv3x3(out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv:
    """3x3 convolution with padding from torchvision.models.resnet but bias is set to True"""
    return nn.Conv(
        features=out_planes,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding=[(dilation, dilation), (dilation, dilation)],
        feature_group_count=groups,
        use_bias=True,
        kernel_dilation=(dilation, dilation),
        kernel_init=nn.initializers.kaiming_normal()
    )


def conv1x1(out_planes: int, stride: int = 1) -> nn.Conv:
    """1x1 convolution"""
    return nn.Conv(
        features=out_planes, 
        kernel_size=(1, 1), 
        strides=(stride, stride), 
        use_bias=True,
        kernel_init=nn.initializers.kaiming_normal()
    )


class BasicBlock(nn.Module):
    inplanes: int
    planes: int
    stride: int = 1
    downsample: Optional[nn.Module] = None
    groups: int = 1
    base_width: int = 64
    dilation: int = 1
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm
    expansion: int = 1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        feature_list: Optional[List[jnp.ndarray]] = None,
        train: bool = True
    ) -> jnp.ndarray:
        # -- sanity checks (same as PyTorch) --
        if self.groups != 1 or self.base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if self.dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported")

        # --- first conv–BN–ReLU ---
        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=[(self.dilation, self.dilation),
                     (self.dilation, self.dilation)],
            feature_group_count=self.groups,
            use_bias=True,
            kernel_dilation=(self.dilation, self.dilation),
            name="conv1"
        )(x)  # :contentReference[oaicite:0]{index=0}
        out = self.norm_layer(
            use_running_average=not train,
            name="bn1"
        )(out)  # :contentReference[oaicite:1]{index=1}
        out = nn.relu(out)  # :contentReference[oaicite:2]{index=2}

        if feature_list is not None:
            feature_list.append(out)

        # --- second conv–BN ---
        out = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            feature_group_count=self.groups,
            use_bias=True,
            kernel_dilation=(1, 1),
            name="conv2"
        )(out)  # :contentReference[oaicite:3]{index=3}
        out = self.norm_layer(
            use_running_average=not train,
            name="bn2"
        )(out)  # :contentReference[oaicite:4]{index=4}

        # --- residual / downsample path ---
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = nn.relu(out)  # :contentReference[oaicite:5]{index=5}

        if feature_list is not None:
            feature_list.append(out)

        return out


class ResNet(nn.Module):
    block: Callable[..., nn.Module]
    layers: List[int]
    num_classes: int = 1000
    zero_init_residual: bool = False
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: Optional[List[bool]] = None
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        feature_list: Optional[List[jnp.ndarray]] = None,
        train: bool = True
    ) -> jnp.ndarray:
        # initial conv + BN + ReLU
        inplanes = 64
        dilation = 1
        rsd = self.replace_stride_with_dilation or [False, False, False]

        x = nn.Conv(
            features=inplanes,
            kernel_size=(3,3),
            strides=(1,1),
            padding=[(1,1),(1,1)],
            use_bias=True,
            name="conv1"
        )(x)
        x = self.norm_layer(
            use_running_average=not train,
            momentum=0.1,
            epsilon=1e-5,
            axis=1,  # For NHWC format, channel axis is -1
            name="bn1"
        )(x)
        x = nn.relu(x)
        if feature_list is not None:
            feature_list.append(x)

        def _make_layer(
            x: jnp.ndarray,
            inplanes: int,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False
        ) -> Tuple[jnp.ndarray, int, int]:
            prev_dilation = dilation
            curr_stride = stride
            nonlocal_dilation = dilation  # capture outer
            # handle dilation vs stride
            if dilate:
                nonlocal_dilation *= curr_stride
                curr_stride = 1

            # prepare downsample if needed
            downsample = None
            if curr_stride != 1 or inplanes != planes * self.block.expansion:
                down_layers = []
                down_layers.append(
                    nn.Conv(
                        features=planes * self.block.expansion,
                        kernel_size=(1,1),
                        strides=(curr_stride,curr_stride),
                        use_bias=False,
                    )
                )
                down_layers.append(
                    self.norm_layer(
                        use_running_average=not train,
                        momentum=0.1,
                        epsilon=1e-5,
                        axis=1,  # For NHWC format, channel axis is -1
                    )
                )
                downsample = Sequential(layers=down_layers)

            # first block
            x = self.block(
                inplanes,
                planes,
                curr_stride,
                downsample,
                self.groups,
                self.width_per_group,
                prev_dilation,
                self.norm_layer,
                expansion=self.block.expansion
            )(x, feature_list=feature_list, train=train)
            inplanes = planes * self.block.expansion

            # remaining blocks
            for _ in range(1, blocks):
                x = self.block(
                    inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.width_per_group,
                    dilation=nonlocal_dilation,
                    norm_layer=self.norm_layer,
                    expansion=self.block.expansion
                )(x, feature_list=feature_list, train=train)

            return x, inplanes, nonlocal_dilation

        # layer1
        x, inplanes, dilation = _make_layer(x, inplanes, 64, self.layers[0], stride=1, dilate=rsd[0])
        # layer2
        x, inplanes, dilation = _make_layer(x, inplanes, 128, self.layers[1], stride=2, dilate=rsd[1])
        # layer3
        x, inplanes, dilation = _make_layer(x, inplanes, 256, self.layers[2], stride=2, dilate=rsd[2])
        # layer4
        x, inplanes, dilation = _make_layer(x, inplanes, 512, self.layers[3], stride=2, dilate=rsd[2])

        if feature_list is not None:
            feature_list.pop(-1)

        # adaptive avg pool to (1,1)
        x = jnp.mean(x, axis=(2,3), keepdims=False)
        if feature_list is not None:
            feature_list.append(x)

        # final linear layer
        x = nn.Dense(
            features=self.num_classes,
            name="fc"
        )(x)

        return x


def build_resnet18(num_classes: int = 1000, norm_layer=nn.BatchNorm, **kwargs) -> ResNet:
    """Build ResNet-18 with specified parameters"""
    return ResNet(BasicBlock, layers=[2, 2, 2, 2], norm_layer=norm_layer, num_classes=num_classes, **kwargs)
