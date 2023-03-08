# PLAN

## 2023-02-24
Right now, I'm able to train the MLP
- I'm still far from a model that I'm happy with

TODO:
- [x] plot the NN PES and the analytic PES along different paths in coordinate space
  - right now, all I have is the test error to go off of
  - maybe the NN PES is complete nonsense?
- I have done this!
  - the PES isn't that great at short distances, where the greatest energies are
  - I think I'm sampling too far out?
    - I have switched to sampling sidelengths between 2.2 and 5.0 (instead of 2.2 and 10.0)
    - I'll find out if it works soon

- [x] try out some weight regularization methods; see if they improve the situation
- I have done this!
  - good results come from using a weight decay value of 0.0001
  - I have tried 5e-4 and 5e-5, but they don't work as well

- [x] the Adam optimizer is probably fine
  - I have settled on this optimizer

- [] clean up the NN training pipeline
  - [x] create a separate subdirectory for each trained model
    - there should be a README.md file inside that describes the model
  - [x] save the loss at each epoch into a file, so the trends are easier to see
    - I also need this information for early stopping
  - [x] save the model at every n^th epoch
    - the .pth files don't seem to take up very much memory, so this is reasonable for now
  - I need the training pipeline to be a bit more stable
    - right now, there are errors with poor descriptions, and some lines have to be commented out, etc.
    - there should be some more consistency

- feed the "classical EOS geometries" into the training data
  - how does it affect the test error when used with the dummy PES?
  - because I'll probably end up doing that with the *ab initio* NN PES

- normalize the starting 1/r values so they are all between [0, 1]
  - though I doubt this will actually improve things?

- increasing the model's size does seem to help
  - keep trying deeper and wider models, and see how far the test error can go down!

