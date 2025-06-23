def blend_pixel_matrices(matrix1, alpha1, matrix2, alpha2):
    """
    Blend two matrices of pixels with their respective alpha values.

    Parameters:
    matrix1 (numpy.ndarray): RGB values of the first matrix of pixels.
    alpha1 (float): Alpha value of the first matrix.
    matrix2 (numpy.ndarray): RGB values of the second matrix of pixels.
    alpha2 (float): Alpha value of the second matrix.

    Returns:
    numpy.ndarray: Blended matrix of RGB values.
    """
    # Ensure the matrices are of the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must be of the same shape")

    # Calculate the resulting alpha
    alpha = alpha1 + alpha2 * (1 - alpha1)

    # Blend the matrices
    blended_matrix = (matrix1 * alpha1 + matrix2 * alpha2 * (1 - alpha1)) / alpha

    return blended_matrix, alpha