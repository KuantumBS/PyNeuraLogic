import json
from typing import Callable

import torch
from torch import nn
from torch.autograd import Function

from neuralogic.core import Template, Settings
from neuralogic.dataset import Dataset


class NeuraLogicFunction(Function):
    @staticmethod
    def forward(ctx, model, mapping, number_format, dtype, *inputs):
        ctx.mapping = mapping
        ctx.model = model
        ctx.number_format = number_format
        ctx.inputs = inputs
        ctx.dtype = dtype

        example_map = mapping[0]
        query_map = mapping[1]

        dataset = Dataset(
            [[example[value.detach().numpy()] for example, value in example_map.items()]],
            [query_map],
        )

        built_dataset = model.build_dataset(dataset)
        sample = built_dataset.samples[0]

        ctx.sample = sample

        return torch.tensor(model(sample, train=False), dtype=dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        backproper, weight_updater = ctx.model.backprop(ctx.sample, -grad_output.detach().numpy())
        state_index = backproper.stateIndex

        mapping = ctx.mapping
        sample = ctx.sample
        number_format = ctx.number_format
        dtype = ctx.dtype

        gradients = tuple(
            -torch.tensor(
                json.loads(
                    str(sample.get_fact(fact).getComputationView(state_index).getGradient().toString(number_format))
                ),
                dtype=dtype,
            ).reshape(input.shape)
            for fact, input in zip(mapping[0], ctx.inputs)
        )

        ctx.model.strategy.trainer.updateWeights(ctx.model.strategy.currentModel, weight_updater)
        ctx.model.strategy.trainer.invalidateSample(ctx.model.strategy.trainer.getInvalidation(), sample.java_sample)

        return None, None, None, None, *gradients


class NeuraLogic(nn.Module):
    def __init__(self, template: Template, to_logic: Callable, settings: Settings, dtype=torch.float32):
        super(NeuraLogic, self).__init__()

        self.model = template.build(settings)
        self.number_format = self.model.settings.settings_class.superDetailedNumberFormat
        self.dtype = dtype

        self.internal_weights = nn.Parameter(torch.empty((0,)))
        self.to_logic = to_logic

    def forward(self, *inputs):
        mapping = self.to_logic(*inputs)
        return NeuraLogicFunction.apply(
            self.model, mapping, self.number_format, self.dtype, *(value for value in mapping[0].values())
        )
