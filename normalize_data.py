import numpy as np
def normalize(data):
    sq = 0
    for el in data:
        sq += np.square(el)

    sq = np.sqrt(sq)
    arr = data / sq
    return arr