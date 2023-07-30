"""
This module contains a basic PyTorch module that I can convert to TorchScript,
so I can follow the TorchScript tutorial.
"""

import torch


class MyModule(torch.nn.Module):
    def __init__(self, n_cols: int, n_rows: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(n_cols, n_rows))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.weight + input


if __name__ == "__main__":
    my_module = MyModule(3, 4)
    example = torch.rand(3, 4)

    traced_script_module = torch.jit.trace(my_module, example)
    traced_script_module.save("example_script_model.pt")
