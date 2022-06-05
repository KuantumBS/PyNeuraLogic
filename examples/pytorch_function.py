import torch

torch.manual_seed(1)

import neuralogic

neuralogic.manual_seed(1)

from torch.nn import Sequential

from neuralogic.nn.torch_function import NeuraLogic
from neuralogic.core import Relation, Template, Settings, Activation

template = Template()
template += (Relation.xor[1, 8] <= Relation.xy) | [Activation.IDENTITY]
template += Relation.xor / 0 | [Activation.IDENTITY]


def to_logic(input):
    """
    Returns tuple -> (mapping to examples, output predicate)
    Mapping is dict of - key = predicate, value = value (tensor) for the predicate
    """
    return (
        {Relation.xy: input},
        Relation.xor,
    )


torch_train_set = [
    torch.tensor([0.0, 0.0]),
    torch.tensor([0.0, 1.0]),
    torch.tensor([1.0, 0.0]),
    torch.tensor([1.0, 1.0]),
]

torch_train_labels = [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0)]

sequential = Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.Tanh(),
    NeuraLogic(template, to_logic, Settings()),
    torch.nn.Sigmoid(),
)

optimizer = torch.optim.SGD(sequential.parameters(), lr=0.1)
loss = torch.nn.MSELoss()

for _ in range(500):
    for input_data, label in zip(torch_train_set, torch_train_labels):

        output = sequential(input_data)
        loss_value = loss(output, label)

        print("Output", output.item(), "Label", label.item(), "Loss", loss_value.item())

        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        optimizer.step()
