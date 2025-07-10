from typing import Callable, List, Optional, Type, Union

from flax import nnx
from functools import partial
import jax
from jax import numpy as jnp, Array

class SequentialWithKeywordArguments(nnx.Module):
    """Sequential module that passes keyword arguments through all layers"""
    
    def __init__(self, *layers):
        self.layers = list(layers)
    
    def __call__(self, x: jax.Array, **kwargs):
        for layer in self.layers:
            # Check if layer accepts kwargs
            try:
                # Most nnx modules don't accept extra kwargs, so we filter
                if hasattr(layer, '__call__') and 'feature_list' in str(layer.__call__.__code__.co_varnames):
                    x = layer(x, **kwargs)
                else:
                    x = layer(x)
            except:
                x = layer(x)
        return x


class BasicBlock(nnx.Module):
    expansion: int = 1
    
    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1, 
        downsample: Optional[nnx.Module] = None,
        *, 
        rngs: nnx.Rngs
    ):
        # Match PyTorch: conv3x3 with bias=True
        self.conv1 = nnx.Conv(
            inplanes, planes, kernel_size=(3, 3), strides=(stride, stride),
            padding="SAME", use_bias=True,
            kernel_init=nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(
            planes, momentum=0.9, epsilon=1e-5,
            scale_init=nnx.initializers.constant(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        
        self.conv2 = nnx.Conv(
            planes, planes, kernel_size=(3, 3), strides=(1, 1),
            padding="SAME", use_bias=True,
            kernel_init=nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(
            planes, momentum=0.9, epsilon=1e-5,
            scale_init=nnx.initializers.constant(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        
        self.downsample = downsample

    def __call__(self, x: jax.Array, feature_list: Optional[List] = None) -> jax.Array:
        """Forward pass matching PyTorch BasicBlock exactly"""
        identity = x

        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        if feature_list is not None:
            feature_list.append(out)

        # Second conv-bn (no relu yet)
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample identity if needed
        if self.downsample is not None:
            identity = self.downsample(x, feature_list=feature_list)

        # Residual connection + final relu
        out = out + identity
        out = nnx.relu(out)

        if feature_list is not None:
            feature_list.append(out)

        return out


class ResNet18(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            3, 64, kernel_size=(3, 3), strides=(1, 1),
            padding="SAME", use_bias=True,
            kernel_init=nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(
            64, momentum=0.9, epsilon=1e-5,
            scale_init=nnx.initializers.constant(1.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )
        
        # Build layers using helper method
        self.inplanes = 64
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, rngs=rngs)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, rngs=rngs)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, rngs=rngs)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, rngs=rngs)
        
        # Final layers
        self.fc = nnx.Linear(
            512 * BasicBlock.expansion, num_classes,
            kernel_init=nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs
        )

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, *, rngs: nnx.Rngs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Create downsample layer
            downsample = SequentialWithKeywordArguments(
                nnx.Conv(
                    self.inplanes, planes * block.expansion, 
                    kernel_size=(1, 1), strides=(stride, stride),
                    use_bias=True,
                    kernel_init=nnx.initializers.kaiming_normal(),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs
                ),
                nnx.BatchNorm(
                    planes * block.expansion, momentum=0.9, epsilon=1e-5,
                    scale_init=nnx.initializers.constant(1.0),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, rngs=rngs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rngs=rngs))

        return SequentialWithKeywordArguments(*layers)

    def __call__(self, x: jax.Array, feature_list: Optional[List] = None) -> jax.Array:
        """Forward pass matching PyTorch ResNet exactly"""
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)

        if feature_list is not None:
            feature_list.append(x)

        x = self.layer1(x, feature_list=feature_list)
        x = self.layer2(x, feature_list=feature_list)
        x = self.layer3(x, feature_list=feature_list)
        x = self.layer4(x, feature_list=feature_list)

        if feature_list is not None:
            feature_list.pop(-1) # remove last layer4 feature?

        # Global average pooling
        x = nnx.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
        x = x.reshape((x.shape[0], -1))  # Flatten

        # ✅ Add final pooled features (matches PyTorch)
        if feature_list is not None:
            feature_list.append(x)

        # Final classification
        x = self.fc(x)
        return x


def build_resnet18(num_classes: int, *, rngs: nnx.Rngs):
    """Build ResNet18 matching the PyTorch version exactly"""
    return ResNet18(num_classes=num_classes, rngs=rngs)

# model = build_resnet18(num_classes=100, rngs=nnx.Rngs(0))
# nnx.display(model)