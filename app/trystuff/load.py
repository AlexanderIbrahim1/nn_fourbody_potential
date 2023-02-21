import numpy as np


if __name__ == "__main__":
    filename = 'file.txt'
    
    x = np.loadtxt(filename, usecols=(0, 1))
    y = np.loadtxt(filename, usecols=(2,))

    print(x)
    print(y)
