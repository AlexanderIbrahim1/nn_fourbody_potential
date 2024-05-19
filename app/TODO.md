# TODO

- after I create and train the shifted softplus model, I need to modify the interface to change which activation function the users put into the MLP
  - right now, it defaults to ReLU
  - maybe add a "kind" label to the `load_potential()` function?
