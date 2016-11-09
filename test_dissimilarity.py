import numpy as np
from numpy.random import uniform, randint
from dissimilarity import compute_dissimilarity

if __name__ == '__main__':

    n_streamlines = 10000
    len_min = 30
    len_max = 150
    print("Generating random tractography.")
    tracks = np.array([uniform(size=(randint(len_min, len_max), 3))
                       for i in range(n_streamlines)],
                      dtype=np.object)

    dissimilarity_matrix, prototype_idx = compute_dissimilarity(tracks,
                                                                verbose=True)
