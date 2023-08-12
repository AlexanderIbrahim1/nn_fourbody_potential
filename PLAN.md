# PLAN

## 2023-05-04

### Get PyTorch working on graham/cedar

- there are instructions for how to do this on computecanada's website (or whatever it is called now)
  - I should bring a few of the .xyz files locally first
    - this lets me work on them here, which is faster
    - then I can just git clone the potential on the remote server, and get it working right away

### Prune the amount of data involved in training [IMPLEMENTED]

- there are lots of sample energies in the training data that are very weak
  - this slows down training, and makes the NN adjust its weights for configurations that barely matter
  - I need to cut down the amount of training data, then retrain

### Try less weight decay [IMPLEMENTED]

- removing the weight decay increases the validation error
  - so I need to keep it in

### Get more training data for short-range interaction energies

- the short-range interaction energies are the most important ones for my purposes
  - I need to get some more, and then retrain

### Add short-range and long-range extrapolations [DONE]

- Long-range
- take a linear combination of the NNPES energy and the dispersion energy
  - also use the same "connecting function" that the SG potential used (with different parameters)
    - maybe the decay should be much faster, because the NNPES energy converges fairly quickly to the long-range energy anyways?

- Short-range
- do what was done in the 3BPES paper
  - assume that the four-body interaction energy varies exponentially with the scaling of all 6 side lengths
  - if a 6-tuple input has side lengths that are too small:
    - get three inputs with side lengths in the training domain
    - calculate energies for these three inputs
    - perform an exponential extrapolation to the small input
  - if the exponential increase is too drastic, switch to a linear extrapolation
- this is more complicated, and requires a buffering system
  - because the inputs to the NN, and the outputs, no longer match 1-to-1, if 3 inputs are needed for a single energy

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

- [x] clean up the NN training pipeline
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


## 2023-08-11

What kinds of samples should I use to check if the C++ version and the Python version give the same results?
- empty case
- one of each range group
- two of each range group
- generate 200 random samples, write them into a file
 - load them from C++ and Python
 - get the energies and write them out to a file
 - write another script to compare the energies of the two programs
