import numpy as np
import matplotlib.pyplot as plt

def blend_pixel_matrices(matrix1:np.array, matrix2:np.array) -> np.array:
    """
    Blends two matrices pixel by pixel, returning a new matrix where each pixel is taken from matrix2 if it is non-zero,
    non-NaN, and not None; otherwise, it takes the pixel from matrix1.
    :param matrix1: First input matrix.
    :param matrix2: Second input matrix.
    :return: Blended matrix.
    """
    retour = np.where(np.logical_and(matrix2 != 0, matrix2 != None), matrix2, matrix1)
    return np.array(retour, dtype=float)


test1 = np.diag([1, 1, 1, 1, 1], k=0)  # Example usage of np.diag to create a diagonal matrix
test2 = np.array([[None, None, None, 5, None],
               [2, None, None, None, None],
               [None, None, None, None, 3],
               [None, None, None, 4, None],
               [5, None, None, None, 6]])
retour = blend_pixel_matrices(test1, test2)
print(retour)
plt.imshow(retour)
plt.colorbar()
plt.show()