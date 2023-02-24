# PLAN

## 2023-02-24
Right now, I'm able to train the MLP
- I'm still far from a model that I'm happy with

TODO:
- plot the NN PES and the analytic PES along different paths in coordinate space
  - right now, all I have is the test error to go off of
  - maybe the NN PES is complete nonsense?

- feed the "classical EOS geometries" into the training data
  - how does it affect the test error when used with the dummy PES?
  - because I'll probably end up doing that with the *ab initio* NN PES

- save the loss at each epoch into a file, to be able to see trends
- save the model at every n^th epoch
  - the .pth files don't seem to take up very much memory, so this is reasonable for now

- try out some weight regularization methods; see if they improve the situation

- normalize the starting 1/r values so they are all between [0, 1]
  - though I doubt this will actually improve things?

- increasing the model's size does seem to help
  - keep trying deeper and wider models, and see how far the test error can go down!

- the Adam optimizer is probably fine
