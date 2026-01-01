import numpy as np
import scipy.io as sio
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


""" ###############################
    #### CREATE THEORETICAL MM ####
    ###############################
"""


def create_MM():
    """Simulate a theoretical MM.

    Parameters:
    None

    Returns:
        MM: Mueller matrix of all N pixels in a Array (4,4,N) shape
        M00: Intensity of the image
        Nx: Number of columns (of pixels) in the image
        Ny: Number of rows (of pixels) in the image
        Feasibility (np.array): Array with the physical feasibility of each pixel
                               (0 means feasible, 1 means M00=0 i.e. not feasible)
    """

    Nx = 3
    Ny = 3
    M = np.zeros((4, 4, Nx*Ny))
    M[0, 0, :] = 1
    M[1, 1, :] = 1
    M[2, 2, :] = 1
    M[3, 3, :] = 1

    M00 = M[0, 0, :]

    M = np.array(M)

    print(M)
    return M, M00, Nx, Ny



def rotator_MM(theta):
    """Theoretical MM of a rotator.
    J.J. Gil, Polarized light (pag.131, eq. 4.30)
    
    Parameters:
        theta (N,) array: Angle of the rotation

    Returns:
        M: Mueller matrix of all N pixels in a Array (4,4,N) shape
    """

    # Nx = 3
    # Ny = 3
    
    if  isinstance(theta, np.ndarray):
        N = len(theta)
    else:
        N = 1
    
    M = np.zeros((4, 4, N))
    M[0, 0, :] = 1
    M[1, 1, :] = np.cos(2*theta)
    M[1, 2, :] = np.sin(2*theta)
    M[2, 1, :] = -np.sin(2*theta)
    M[2, 2, :] = np.cos(2*theta)
    M[3, 3, :] = 1

    M = np.array(M)
    
    return M


def retarder_MM(delta):
    """Theoretical MM of a retarder.
    J.J. Gil, Polarized light (pag.132, eq. 4.31)
    
    Parameters:
        
        delta (N,) array: Retardance

    Returns:
        M: Mueller matrix of all N pixels in a Array (4,4,N) shape

        
    """

    # Nx = 3
    # Ny = 3
    
    if  isinstance(delta, np.ndarray):
        N = len(delta)
    else:
        N = 1
           
    M = np.zeros((4, 4,N))
    M[0, 0, :] = 1
    M[1, 1, :] = 1
    M[2, 2, :] = np.cos(delta)
    M[2, 3, :] = np.sin(delta)
    M[3, 2, :] = -np.sin(delta)    
    M[3, 3, :] = np.cos(delta)

    M = np.array(M)

    return M


def diattenuator_MM(p1, p2, norm=1):
    """Theoretical MM of a diattenuator.
    J.J. Gil, Polarized light (pag.143, eq. 4.79)
    
    Parameters:
        p1 (N,) array: Axis 1 intensity coefficient
        p2 (N,) array: Axis 2 intensity coefficient
        norm (int): 1 if we want the normalized MM, any other value for the non-normalized

    Returns:
        MM: Mueller matrix of all N pixels in a Array (4,4,N) shape
    """

    # Nx = 3
    # Ny = 3
        
    if  isinstance(p1, np.ndarray):
        N = len(p1)
    else:
        N = 1
    
    if norm==1:
        a = 1/2 * (p1**2+p2**2)
    else:
        a=1
        
    M = np.zeros((4, 4, N))
    M[0, 0, :] = 1/2 * (p1**2+p2**2) /a
    M[0, 1, :] = 1/2 * (p1**2-p2**2) /a
    M[1, 0, :] = 1/2 * (p1**2-p2**2) /a
    M[1, 1, :] = 1/2 * (p1**2+p2**2) /a
    M[2, 2, :] = p1*p2 /a
    M[3, 3, :] = p1*p2 /a
    

    M = np.array(M)

    return M

""" ####################
    #### READ FILES ####
    ####################
"""


def read_file(file_name, selected_pixels=None, crop_image=False):
    """Reads the polarimetric information contained in a .mat file.

    Parameters:
        file_name ('str'): Name of the file to read
        crop_image (bool): True or False. False = default
        selected_pixels (array): [(int, int), (int, int)]

    Returns:
        MM: Mueller matrix of all N pixels in a (4,4,N) shape
        M00: Intensity of the image
        Nx: Number of columns (of pixels) in the image
        Ny: Number of rows (of pixels) in the image
        Feasibility (np.array): Array with the physical feasibility of each pixel
                               (0 means feasible, 1 means M00=0 i.e. not feasible)
    """

    print('Reading .mat file...')
    mat_file = file_name  # Name of the file
    mat_contents = sio.loadmat(mat_file)
    sorted(mat_contents.keys())

    # Extract all the Mueller matrix elements:
    M00 = mat_contents['M00']
    M01 = mat_contents['M01']
    M02 = mat_contents['M02']
    M03 = mat_contents['M03']
    M10 = mat_contents['M10']
    M11 = mat_contents['M11']
    M12 = mat_contents['M12']
    M13 = mat_contents['M13']
    M20 = mat_contents['M20']
    M21 = mat_contents['M21']
    M22 = mat_contents['M22']
    M23 = mat_contents['M23']
    M30 = mat_contents['M30']
    M31 = mat_contents['M31']
    M32 = mat_contents['M32']
    M33 = mat_contents['M33']
    try:
        SatROI = mat_contents['SatROI']
    except:
        SatROI =0

    Nx = len(M00)
    Ny = len(M00[0])

    print('.mat file loaded')

    if crop_image is True:
        if selected_pixels is None:
            pixels, size_image = select_pixels_image(M30, Nx, Ny)
        else:
            pixels = selected_pixels
        # Cropped Mueller matrix:

        M00 = M00[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M01 = M01[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M02 = M02[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M03 = M03[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]

        M10 = M10[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M11 = M11[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M12 = M12[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M13 = M13[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]

        M20 = M20[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M21 = M21[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M22 = M22[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M23 = M23[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]

        M30 = M30[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M31 = M31[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M32 = M32[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        M33 = M33[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]
        SatROI = SatROI[pixels[0][0]:pixels[1][0], pixels[0][1]:pixels[1][1]]

        Nx = len(M00)
        Ny = len(M00[0])

    MM_img = np.array([[M00, M01, M02, M03],  # Create the Mueller matrix
                       [M10, M11, M12, M13],  # of the whole image
                       [M20, M21, M22, M23],
                       [M30, M31, M32, M33]])

    N = Nx * Ny  # Number of pixels
    MM = MM_img.reshape(4, 4, N)  # Reshape in order to have the MM of each pixel

    # Physical feasibility
    M00 = np.ravel(M00)  # Reshape M00 image into a 1D vector
    SatROI = np.ravel(SatROI)  # Reshape M00 image into a 1D vector
    Feasibility = np.zeros((N,), dtype=int)  # Initialize array assuming all pixels are feasible

    Feasibility = np.where(SatROI == 1, Feasibility, 1)  # Feasibility = 1 -> saturated pixels

    # Remove MM with very low M00:
    #MM, Feasibility = avoid_0_intensity_pixels(MM, N, M00, Feasibility)

    return MM, M00, Nx, Ny, Feasibility, SatROI




def save_MM_mat(file_name, MM, M00, Nx, Ny):
    """Save the polarimetric information in a .mat file.

    Parameters:
        file_name ('str'): Name of the file to read
        MM: Mueller matrix of all N pixels in a (4,4,N) shape
        M00: Intensity of the image
        Nx: Number of columns (of pixels) in the image
        Ny: Number of rows (of pixels) in the image

    Returns:
        MM: Mueller matrix of all N pixels in a (4,4,N) shape
        M00: Intensity of the image
        Nx: Number of columns (of pixels) in the image
        Ny: Number of rows (of pixels) in the image

    """

    # Save the file in .mat format
    sio.savemat(file_name, {'MM': MM, 'Nx': Nx, 'Ny': Ny, 'M00': M00})

    print(f"File saved as {file_name}")

    return MM, M00, Nx, Ny




""" ##################################
    ####  MM BASIC MANIPULATIONS  ####
    ##################################
"""


def avoid_0_intensity_pixels(M, N, M00, Feasibility, epsilon=10):
    """Avoid M00<epsilon pixels

        Parameters:
        M (4,4,N) array: Mueller Matrix
        M00 (N) array: M00
        epsilon = 10  # Condition of 0 intensity pixel (M00 is noise)

    Returns:
        M_filtered (4,4,N) array: Mueller Matrix
        Feasibility (np.array): Array with the physical feasibility of each pixel
             (0 means feasible, 1 means M00=0 i.e. not feasible)
    """
    M_filtered = copy.deepcopy(M)
    Feasibility = np.where(M00 > epsilon, Feasibility, 1)  # Feasibility = 1 -> 0 intensity pixels
    NotFeasible = np.where(M00 < epsilon)  # Array with the index of 0 intensity pixels

    if len(NotFeasible[0]) == 0:  # Give info to the user
        print('All pixels have appropriate intensity')

    else:
        print('Some pixels have 0 intensity')
        M_filtered[:, :, np.where(M00 < epsilon)] = 0
        M_filtered[0, 0, np.where(M00 < epsilon)] = 1
        M_filtered[1, 1, np.where(M00 < epsilon)] = 1
        M_filtered[2, 2, np.where(M00 < epsilon)] = 1
        M_filtered[3, 3, np.where(M00 < epsilon)] = 1

    return M_filtered, Feasibility


def PhysicalFeasibility_filter(M, Feasibility):
    """Converts a given Mueller matrix into a physically feasible one.
    Before filtering, MM should be normalized with .normalize().

    Parameters:
        M (4,4,N) array: Mueller Matrix
        Feasibility (1D array): Outputted when using .normalize() function.

    Returns:
        MM_filt (4,4,N) array: Physically feasible Mueller matrix
        H_filt (4,4,N) array: Filtered Covariance matrix
        Feasibility (1D array): Updated with not feasible pixels (0 -> feasible
                                                                  1 -> M00=0 / saturated pixels
                                                                  2 -> eigenvalue<0)
    """

    # Filter not physically feasible pixels (negative eigenvalues)
    H = covariance_matrix(M)
    H = np.moveaxis(H, -1, 0)  # Reshape to (N,4,4)
    eigenvalues_H, U = np.linalg.eigh(H)  # Eigenvalues & Eigenvectors not square matrix
    # print(eigenvalues_H)

    U_dag = np.conjugate(np.transpose(U, axes=[0, 2, 1]))  # Dagger (transpose & conjugate) of eigenvectors matrix U

    for i in range(0, 3):  # Study pixels where lambda is < 0
        Feasibility = np.where(np.transpose(eigenvalues_H, axes=[1, 0])[i] >= 0, Feasibility, 2)
        # Feasibility = 2 -> filtered pixels (lambdai<0) // pixels with <0 eigenvalues, mark them with number 2

    # Filter the lambda values:
    # eigenvalues_H = np.where(eigenvalues_H >= 0, eigenvalues_H, 0)  # If eigenvalue is negative, make it 0

    # Recompute H and M with the filtered eigenvalues
    N = np.size(H, axis=0)  # nº of pixels
    diag_lambdas, H_filt, lambdas, = np.zeros([N, 4, 4]), np.zeros_like(H), np.zeros([N, 4])  # Initialize

    for i in range(0, N):  # Puts the eigenvalues of each pixel in a diagonal matrix
        diag_lambdas[i, :, :] = np.diag(eigenvalues_H[i, :])
        lambdas[i, :] = eigenvalues_H[i, :][::-1]

    H_filt = U @ diag_lambdas @ U_dag  # Recompose H with the filtered eigenvalues

    H = copy.deepcopy(H_filt)  # Update H with filtered values
    H = np.moveaxis(H, 0, -1)  # (4,4,N) shape

    MM_filt = MM_from_covariance(H)
    
    # print(lambdas)

    return MM_filt, Feasibility, lambdas, H_filt


def normalize(M, Feasibility=None):
    """Normalizes a given MM object dividing by its total intensity, M00.
    Shape of MM object should be (4,4,N).

    Parameters: 
        M (4,4,N) array: Mueller matrix
        Feasibility (1D array): Array outputted in read_file() function

    Returns:
        M (4,4,N) array: Normalized Mueller matrix
        Feasibility (1D array)
    """

    if Feasibility is None:
        M00 = M[0, 0, :]
        N = len(M00)

        Feasibility = np.zeros((N,), dtype=int)  # Initialize array assuming all pixels are feasible
        Feasible = np.where(Feasibility == 0)  # Search for feasible (and not) pixels
        epsilon = 1 / 4096  # Condition of 0 intensity pixel
        NotFeasible = np.where(M00 < epsilon)  # Array with the index of NOT feasible pixels
        Feasibility = np.where(M00 > epsilon, Feasibility, 1)  # Feasibility = 1 -> 0 intensity pixels

        M[:, :, Feasible] = M[:, :, Feasible] / M00  # Normalizes all physically feasible pixels
        M[:, :, NotFeasible] = 0  # Anulates the whole MM for a not feasible pixel

        return M, Feasibility

    Feasible = np.where(Feasibility == 0)  # Search for feasible (and not) pixels
    NotFeasible = np.where(Feasibility != 0)

    M00 = M[0, 0, :]
    M[:, :, Feasible] = M[:, :, Feasible] / M00[Feasible]  # Normalizes all physically feasible pixels
    M[:, :, NotFeasible] = 0  # Anulates the whole MM for a not feasible pixel

    for i in range(0, 3):  # NotFeasible MM pixels in a diagonal Mueller matrix
        M[i, i, NotFeasible] = 1  # Mii = 1 for a not feasible pixel

    return M, Feasibility


def covariance_matrix(M):
    """Computes the covariance matrix H of a given Mueller matrix.

    Parameters:
        M (4,4,N) array: Mueller matrix

    Returns:
        H (4,4,N) array: Covariance matrix of M

    """

    H = 0.25*np.array([[M[0, 0, :] + M[0, 1, :] + M[1, 0, :] + M[1, 1, :],  # J.J. Gil, Polarized Light (pag. 171, eq 5.16)
                   M[0, 2, :] + M[1, 2, :] + 1j * (M[0, 3, :] + M[1, 3, :]),
                   M[2, 0, :] + M[2, 1, :] - 1j * (M[3, 0, :] + M[3, 1, :]),
                   M[2, 2, :] + M[3, 3, :] + 1j * (M[2, 3, :] - M[3, 2, :])],

                  [M[0, 2, :] + M[1, 2, :] - 1j * (M[0, 3, :] + M[1, 3, :]),
                   M[0, 0, :] - M[0, 1, :] + M[1, 0, :] - M[1, 1, :],
                   M[2, 2, :] - M[3, 3, :] - 1j * (M[2, 3, :] + M[3, 2, :]),
                   M[2, 0, :] - M[2, 1, :] - 1j * (M[3, 0, :] - M[3, 1, :])],

                  [M[2, 0, :] + M[2, 1, :] + 1j * (M[3, 0, :] + M[3, 1, :]),
                   M[2, 2, :] - M[3, 3, :] + 1j * (M[2, 3, :] + M[3, 2, :]),
                   M[0, 0, :] + M[0, 1, :] - M[1, 0, :] - M[1, 1, :],
                   M[0, 2, :] - M[1, 2, :] + 1j * (M[0, 3, :] - M[1, 3, :])],

                  [M[2, 2, :] + M[3, 3, :] - 1j * (M[2, 3, :] - M[3, 2, :]),
                   M[2, 0, :] - M[2, 1, :] + 1j * (M[3, 0, :] - M[3, 1, :]),
                   M[0, 2, :] - M[1, 2, :] - 1j * (M[0, 3, :] - M[1, 3, :]),
                   M[0, 0, :] - M[0, 1, :] - M[1, 0, :] + M[1, 1, :]]])
    return H



def MM_from_covariance(H):
    """Computes the Mueller matrix from the covariance matrix

    Parameters:
        H (4,4,N) array: Covariance matrix

    Returns:
        M (4,4,N) array: Mueller matrix

    """
    M = np.array(
        [[H[0, 0, :] + H[1, 1, :] + H[2, 2, :] + H[3, 3, :],  # J.J. Gil, Polarized Light (pag. 172, eq 5.18)
          H[0, 0, :] - H[1, 1, :] + H[2, 2, :] - H[3, 3, :],
          H[0, 1, :] + H[1, 0, :] + H[2, 3, :] + H[3, 2, :],
          -1j * (H[0, 1, :] - H[1, 0, :] + H[2, 3, :] - H[3, 2, :])],

         [H[0, 0, :] + H[1, 1, :] - H[2, 2, :] - H[3, 3, :],
          H[0, 0, :] - H[1, 1, :] - H[2, 2, :] + H[3, 3, :],
          H[0, 1, :] + H[1, 0, :] - H[2, 3, :] - H[3, 2, :],
          -1j * (H[0, 1, :] - H[1, 0, :] - H[2, 3, :] + H[3, 2, :])],

         [H[0, 2, :] + H[2, 0, :] + H[1, 3, :] + H[3, 1, :],
          H[0, 2, :] + H[2, 0, :] - H[1, 3, :] - H[3, 1, :],
          H[0, 3, :] + H[3, 0, :] + H[1, 2, :] + H[2, 1, :],
          -1j * (H[0, 3, :] - H[3, 0, :] - H[1, 2, :] + H[2, 1, :])],

         [1j * (H[0, 2, :] - H[2, 0, :] + H[1, 3, :] - H[3, 1, :]),
          1j * (H[0, 2, :] - H[2, 0, :] - H[1, 3, :] + H[3, 1, :]),
          1j * (H[0, 3, :] - H[3, 0, :] + H[1, 2, :] - H[2, 1, :]),
          H[0, 3, :] + H[3, 0, :] - H[1, 2, :] - H[2, 1, :]]])

    M = M.real  # Get rid of imaginary part in Mueller matrix
    
    return M
    



def coherency_matrix(M):
    """Computes the coherency matrix C of a given Mueller matrix.

    Parameters:
        M (4,4,N) array: Mueller matrix

    Returns:
        C (4,4,N) array: Coherency matrix of M

    """

    C = np.array([[M[0, 0, :] + M[1, 1, :] + M[2, 2, :] + M[3, 3, :],  # J.J. Gil, Polarized Light (pag. 174, eq 5.23)
                   M[0, 1, :] + M[1, 0, :] - 1j * (M[2, 3, :] - M[3, 2, :]),
                   M[0, 2, :] + M[2, 0, :] + 1j * (M[1, 3, :] - M[3, 1, :]),
                   M[0, 3, :] + M[3, 0, :] - 1j * (M[1, 2, :] - M[2, 1, :])],

                  [M[0, 1, :] + M[1, 0, :] + 1j * (M[2, 3, :] - M[3, 2, :]),
                   M[0, 0, :] + M[1, 1, :] - M[2, 2, :] - M[3, 3, :],
                   M[1, 2, :] + M[2, 1, :] + 1j * (M[0, 3, :] - M[3, 0, :]),
                   M[1, 3, :] + M[3, 1, :] - 1j * (M[0, 2, :] - M[2, 0, :])],

                  [M[0, 2, :] + M[2, 0, :] - 1j * (M[1, 3, :] - M[3, 1, :]),
                   M[1, 2, :] + M[2, 1, :] - 1j * (M[0, 3, :] - M[3, 0, :]),
                   M[0, 0, :] - M[1, 1, :] + M[2, 2, :] - M[3, 3, :],
                   M[2, 3, :] + M[3, 2, :] + 1j * (M[0, 1, :] - M[1, 0, :])],

                  [M[0, 3, :] + M[3, 0, :] + 1j * (M[1, 2, :] - M[2, 1, :]),
                   M[1, 3, :] + M[3, 1, :] + 1j * (M[0, 2, :] - M[2, 0, :]),
                   M[2, 3, :] + M[3, 2, :] - 1j * (M[0, 1, :] - M[1, 0, :]),
                   M[0, 0, :] - M[1, 1, :] - M[2, 2, :] + M[3, 3, :]]])

    N = len(M[0, 0])  # Number of pixels
    a = np.repeat(0.25, N)  # Make an array of 0.25's
    C = np.moveaxis(C, -1, 0)  # (N,4,4) shape
    C = np.einsum('i,ijk->ijk', a, C)  # Multiply 0.25*C

    C = np.moveaxis(C, 0, -1)  # (4,4,N) reshape back to original

    return C


def identity(Mdim, N):
    """Creates an array of N elements, where each of them is a MxM identity matrix.

    Parameters:
        Mdim (int): Dimensions of the identity matrix.
        N (int): Number of matrices in the array.

    Returns:
        I_M (NxMxM array): N identity matrices of size MxM.
    """
    shape = (Mdim, Mdim, N)
    I_M = np.zeros(shape)
    idx = np.arange(shape[0])
    I_M[idx, idx, :] = 1

    return I_M


def flatten(M):
    """Reshapes a (4,4,N) MM object into shape (16,N).

    Returns:
        Same MM object reshaped into (16, N).
    """

    N = len(M[0, 0, :])
    MM = M.reshape(16, N)

    return MM


"""
###################################
####  POLARIMETRIC PARAMETERS  ####
###################################
"""


def Polarizance(M):
    """Calculates all parameters related to polarizance from a (4x4,N) normalized MM.

    Returns:
        P_vect (3xN array): Polarizance vector
        P (1xN array): Polarizance
        Pc (1xN array): Circular polarizance
        Pl (1xN array): Linear polarizance
        Pazi (1xN array): Azimuth angle of the polarizance vector
        Pelip (1xN array): Elipticity angle of the polarizance vector
    """

    P_vect = np.array([M[1, 0, :], M[2, 0, :], M[3, 0, :]])
    P = np.sqrt(P_vect[0, :] ** 2 + P_vect[1, :] ** 2 + P_vect[2, :] ** 2)

    Pl = np.sqrt(P_vect[0, :] ** 2 + P_vect[1, :] ** 2)  # Pag 204 J.J. Gil Polarized Light
    Pc = P_vect[2, :]

    # Azimuth angle of the polarizance vector
    Pazi = np.zeros(P.size)  # Initialize Pazi
    Pazi_Feasible = np.where(np.abs(P_vect[0, :]) >= 0.00000000001)
    Pazi[Pazi_Feasible] = 0.5 * np.arctan(P_vect[1, Pazi_Feasible] / P_vect[0, Pazi_Feasible]) / np.pi * 180
    Pazi = np.where(Pazi != 0, Pazi, 90)
    # Pazi = 0.5 * np.arctan(P_vect[1, :] / P_vect[0, :]) / np.pi * 180

    # Elipticity angle of the polarizance vector
    Pelip = np.zeros(P.size)  # Initialize Pelip:
    Pelip_Feasible = np.where(np.abs(Pl) >= 0.00000000001)
    Pelip[Pelip_Feasible] = 0.5 * np.arctan(Pc[Pelip_Feasible] / Pl[Pelip_Feasible]) / np.pi * 180
    Pelip = np.where(Pl != 0, Pelip, 90)
    # Pelip = 0.5 * np.arctan(Pc / Pl) / np.pi * 180

    return P_vect, P, Pc, Pl, Pazi, Pelip


def Diattenuation(M):
    """Calculates all parameters related to diattenuation from a (4x4,N) normalized MM.

    Returns:
        D_vect (3xN array): Diattenuation vector
        D (1xN array): Diattenuation
        Dc (1xN array): Circular Diattenuation
        Dl (1xN array): Linear diattenuation
        Dazi (1xN array): Azimuth angle of the diattenuation vector
        Delip (1xN array): Elipticity angle of the diattenuation vector
    """

    D_vect = np.array([M[0, 1, :], M[0, 2, :], M[0, 3, :]])
    D = np.sqrt(D_vect[0, :] ** 2 + D_vect[1, :] ** 2 + D_vect[2, :] ** 2)

    Dl = np.sqrt(D_vect[0, :] ** 2 + D_vect[1, :] ** 2)  # Pag 204 J.J. Gil Polarized Light
    Dc = D_vect[2, :]

    # Azimuth angle of the diattenuation vector
    Dazi = np.zeros(D.size)  # Initialize Dazi
    Dazi_Feasible = np.where(np.abs(D_vect[0, :]) >= 0.00000000001)
    Dazi[Dazi_Feasible] = 0.5 * np.arctan(D_vect[1, Dazi_Feasible] / D_vect[0, Dazi_Feasible]) / np.pi * 180
    Dazi = np.where(Dazi != 0, Dazi, 90)
    # Dazi = 0.5 * np.arctan(D_vect[1, :] / D_vect[0, :]) / np.pi * 180

    # Elipticity angle of the diattenuation vector
    Delip = np.zeros(D.size)  # Initialize Pelip:
    Delip_Feasible = np.where(np.abs(Dl) >= 0.00000000001)
    Delip[Delip_Feasible] = 0.5 * np.arctan(Dc[Delip_Feasible] / Dl[Delip_Feasible]) / np.pi * 180
    Delip = np.where(Dl != 0, Delip, 90)
    # Delip = 0.5 * np.arctan(Dc / Dl) / np.pi * 180

    return D_vect, D, Dc, Dl, Dazi, Delip


def Retardance(M, Mr=None):
    """Calculates all parameters related to retardance,
    once the MM has been Lu-Chipman decomposed.

    Parameters:
        M (4x4xN array): Mueller Matrix Image
        Mr (4x4xN array): Retarder's MM outputted in .LuChipman_decomposition()

    Returns:
        R (1xN array): Retardance
        delta (1xN array): Linear retardance

        Psi (1xN array): Retarder's rotation
        TODO: Rotor
    """

    if Mr is None:
        Mr = LuChipman_decomposition(M)[1]  # If no Mr is give, make the Lu-Chipman decomposition

    mr = Mr[1:, 1:, :]  # Small 3x3 sub-matrix of Mr
    tr_mr = np.matrix.trace(mr)  # Trace of mr

    R = np.arccos(0.5 * (tr_mr - 1)) / np.pi * 180  # Goldstein 8-134 Global retardance
    # TODO: Cuidado con los índices de Mr
    delta = (np.arccos(np.sqrt((Mr[1, 1, :] + Mr[2, 2, :]) ** 2 + (Mr[2, 1, :] - Mr[1, 2, :]) ** 2) - 1)) / np.pi * 180  # Linear retardance

    Psi = (np.arctan((Mr[2, 1, :] - Mr[1, 2, :]) / (Mr[1, 1, :] + Mr[2, 2, :]))) / np.pi * 180  # Retarder's rotation

    # TODO: Another Retardance parameters:
    # sindr2 = 2 * np.sin(R) / np.pi * 180
    # r1 = real(squeeze((mr(2, 3,:) - mr(3, 2,:)))./ sindr2);
    # r2 = real(squeeze((mr(3, 1,:) - mr(1, 3,:)))./ sindr2);
    # Rc = real(squeeze((mr(3, 1,:) - mr(1, 3,:)))./ sindr2. * R); %Retardador circular
    # theta = 0.5 * atand(r2. / r1);
    # RL = real(sqrt(r1. ^ 2 + r2. ^ 2). * R); # Linear retarder

    return R, delta, Psi


def IPPs(M, H_filt=None, Ls=None, Ls_output=False):
    """Computes IPPs and H eigenvalues. M should be already filtered and normalized.

    Parameters:
        M (4,4,N array) : Mueller matrix
        H_filt (4x4xN array): Covariance matrix filtered. Outputted in
                              PhysicalFeasibility_filter() function.
        Ls (4xN array): Lambdas (lambda0 > lambda1 > lambda2 > lambda3)
        Ls_output (bool): If True, also returns the eigenvalues of H.

    Returns:
        (P1, P2, P3) tuple: Polarimetric purity indices
        (lambda1, lambda2, lambda3, lambda4) tuple: Eigenvalues of the filtered H
    """

    if Ls is None:  # If no covariance matrix is given, compute it
        H_filt = covariance_matrix(M)
        H_filt = np.moveaxis(H_filt, -1, 0)  # Reshape to (N,4,4)
        eigenvalues_H = np.linalg.eigh(H_filt)[0]  # np.real(np.linalg.eigh(H_filt)[0])  # Get rid of imaginary part
        Ls = -np.sort(-eigenvalues_H, axis=1)  # Order manually (decreasing order)

    trH = H_filt.trace(axis1=1, axis2=2).real  # Make elements of H real to calculate its trace

    lambda0 = Ls[:, 0]
    lambda1 = Ls[:, 1]
    lambda2 = Ls[:, 2]
    lambda3 = Ls[:, 3]

    P1 = (lambda0 - lambda1) / trH
    P2 = (lambda0 + lambda1 - 2 * lambda2) / trH
    P3 = (lambda0 + lambda1 + lambda2 - 3 * lambda3) / trH

    if Ls_output is False:
        return P1, P2, P3

    else:
        return P1, P2, P3, lambda0, lambda1, lambda2, lambda3


def Pdelta(M):
    """Computes the parameter Pdelta, defined as the Euclidean distance
    between the given MM and an ideal depolarizer.

    Returns:
        Pdelta: array with (4,4,N) shape
    """
    quadratic_sum = np.sum(M ** 2, axis=1)
    quadratic_sum = np.sum(quadratic_sum, axis=0)
    P_delta = np.sqrt((quadratic_sum - M[0, 0] ** 2) / 3) / M[0, 0]

    return P_delta


def Ps(M):
    """Computes the degree of spherical purity (Ps) as done in
    Polarized Light (JJ Gil), Pag 199, eq. (6.1).

    Returns:
        Ps (1xN) array: Degree of spherical purity
    """
    quadratic_sum = np.sum(M[1:, 1:, :] ** 2, axis=1)
    quadratic_sum = np.sum(quadratic_sum, axis=0)
    Ps = np.sqrt(quadratic_sum / 3)

    return Ps


"""
###########################
###  MM DECOMPOSITIONS  ###
###########################
"""


def LuChipman_decomposition(M, norm=False):
    """Returns the three pure matrices (Depolarizer, Retarder, Diattenuator)
    from the Lu-Chipman decomposition of M: M = Mp*Mr*Md
    Parameters:
        M (4,4,N) array: Mueller Matrix
        norm (boolean): If True the input M is normalized.

    Returns:
        (Mdelta, Mr, Md): Tuple where element [0] is Mdelta (depolarizer), [1] is Mr (retarder) and [2] is Md (diattenuator).
                      Each of them is an array of size (4x4xN).
    """
    if norm is True:
        M = normalize(M)[0]

    # Extract info from M:
    N = len(M[0, 0])

    # Initialize the output matrices:
    Mdelta, Mr, Md = np.zeros_like(M), np.zeros_like(M), np.zeros_like(M)
    Mdelta[0, 0, :], Mr[0, 0, :], Md[0, 0, :] = 1, 1, 1  # M00 = 1 since M should be normalized

    # Start by calculating the diattenuator
    D_vect = np.array([M[0, 1, :], M[0, 2, :], M[0, 3, :]])  # Diattenuation vector
    DT_vect = np.transpose(D_vect, axes=(1, 0))  # D^T, transpose vector of D
    D = np.sqrt(D_vect[0, :] ** 2 + D_vect[1, :] ** 2 + D_vect[2, :] ** 2)  # Diattenuation
    D[np.where(D > 1)] = 1

    I_3 = identity(3, N)  # Creates N identity 3x3 matrices
    I_3 = np.moveaxis(I_3, -1, 0)  # Shape like (N,3,3)

    D_tensor_DT = np.zeros_like(M[1:, 1:, :])  # Initialize tensor product of D*D^T
    D_tensor_DT = np.moveaxis(D_tensor_DT, -1, 0)  # Reshape to (N,3,3)
    for i in range(0, N):  # Tensor product D*D^T
        D_tensor_DT[i, :, :] = np.tensordot(D_vect[0:3, i], DT_vect[i, 0:3], axes=0)

    # a and b parameters are computed:
    a = np.real(np.sqrt(1 - D ** 2))  # Goldstein 9-125 -> Parameter used later/ real part of a
    a[np.where(D == 1)] = 0.000001  # Condition to avoid that 1/a = inf
    b = np.zeros(N)  # Goldstein 9-126 -> Initialize b parameter
    b[np.where(D > 0.000001)] = (1 - a[np.where(D > 0.000001)]) / D[np.where(D > 0.000001)] ** 2  # Goldstein 9-126 -> Parameter used later
    b[np.where(D <= 0.000001)] = 1/2  # When D=0, b tends to 1/2 (through limit calculation)

    # The Diattenuation matrix is computed:
    aI_3 = np.einsum('i,ijk->ijk', a, I_3)  # a*Identity(3). Multiply i element of a with i matrix of I_3
    bD_tensor_DT = np.einsum('i,ijk->ijk', b, D_tensor_DT)  # Same with b and D*D^T
    mD = aI_3 + bD_tensor_DT  # 3x3 sub-matrix

    if D.any() <= 0.000001:  # just if D is close to 0
        mD[np.where(D <= 0.000001)] = np.moveaxis(identity(3, N), -1, 0)[np.where(D <= 0.000001)] # Goldstein 9-123
    Md[1:, 0, :] = D_vect  # Fill with D and D^T vectors
    Md[0, 1:, :] = D_vect
    Md[1:, 1:, :] = np.moveaxis(mD, 0, -1)  # Fill 3x3 sub-matrix with mD

    # Calculate the inverse of the diattenuator: Goldstein 9-130
    M1 = np.zeros_like(Md)  # Create matrices to operate
    M2 = np.zeros_like(Md)
    M1[0, 0, :] = 1

    M1[1:, 0, :] = -1 * D_vect  # Fill with -D and -D^T vectors
    M1[0, 1:, :] = -1 * D_vect
    M1[1:, 1:, :] = np.moveaxis(I_3, 0, -1)  # Goldstein 9-130 -> Fill 3x3 sub-matrix with Identity(3)
    M2[1:, 1:, :] = np.moveaxis(D_tensor_DT, 0, -1)  # Goldstein 9-130 -> Fill 3x3 sub-matrix with D*D^T, rest is 0

    M1 = np.moveaxis(M1, -1, 0)
    M2 = np.moveaxis(M2, -1, 0)
    # TODO: Qué ocurre si a = 0 en Goldstein 9-130
    # TODO: corregir M_ para valores de a!=0
    M1 = np.einsum('i,ijk->ijk', 1 / (a ** 2), M1)  # 1/a^2 * M1
    M2 = np.einsum('i,ijk->ijk', 1 / (a ** 2 * (a + 1)), M2)

    Md_I = M1 + M2  # Goldstein 9-130 -> Inverse matrix of Md, Md^-1
    Md_I = np.moveaxis(Md_I, 0, -1)

    # Extract Md^I from the total matrix M
    M_ = np.moveaxis(np.zeros_like(M), -1, 0)  # M_ [Shape like (N,4,4)] -> (M' in the book Goldstein 9-144) is M with the information of Md substracted

    M = np.moveaxis(M, -1, 0)
    Md_I = np.moveaxis(Md_I, -1, 0)

    M_ = M @ Md_I # Goldstein 9-144

    # Extract useful info from M_ Goldstein 9-145 (m')
    m_ = M_[:, 1:, 1:]  # (Nx3x3) sub-matrix of M_ (m')
    m_T = np.transpose(m_, axes=(0, 2, 1))  # m^T, transposed matrix of m_

    m_mT = m_ @ m_T  # Matrix multiplication -> Goldstein 9-146

    #lambdas = np.linalg.eig(m_mT)[0]  # Eigenvalues of m_ * m_^T = eigenvalues of m_delta^2
    lambdas = np.linalg.svd(m_mT)[1]  # Eigenvalues of m_ * m_^T = eigenvalues of m_delta^2
    lambdas = -np.sort(-lambdas)  # Decreasing order

    l1 = lambdas[:, 0]  # l1>l2>l3
    l2 = lambdas[:, 1]
    l3 = lambdas[:, 2]

    k1 = np.real(np.sqrt(l1)) + np.real(np.sqrt(l2)) + np.real(np.sqrt(l3))  # Parameters used later
    k2 = np.real(np.sqrt(l1 * l2)) + np.real(np.sqrt(l2 * l3)) + np.real(np.sqrt(l3 * l1))
    k3 = np.real(np.sqrt(l1 * l2 * l3))

    # Goldstein 9-149:
    k2I_3 = np.einsum('i,ijk->ijk', k2, I_3)  # k2 * I_3 -> Goldstein 9-149 (first term)
    Mp1 = m_mT + k2I_3  # Goldstein 9-149 (second term)
    k1m_mT = np.einsum('i,ijk->ijk', k1, m_mT)  # -> Goldstein 9-149 (second term)
    k3I_3 = np.einsum('i,ijk->ijk', k3, I_3)    # -> Goldstein 9-149 (second term)
    Mp2 = k1m_mT + k3I_3  # -> Goldstein 9-149 (first term)
    sign = np.linalg.slogdet(m_)[0]  # sign of the determinant of m' -> Goldstein 9-149 (sign)
    Mp1_I = np.einsum('i,ijk->ijk', sign, np.linalg.inv(Mp1))  # Goldstein 9-149 (first term) -> Inverse of Mp1, multiplied by the sign of m'
    mdelta = Mp1_I @ Mp2  # Nx3x3 sub-matrix of Mp # -> Goldstein 9-149

    """ A different way to obtain Mdelta:
    P = np.moveaxis(M[:, 1:, 0], 0, -1)  # Polarizance vector
    m = np.moveaxis(M[:, 1:, 1:], 0, -1)  # 3x3 sub-matrix of M
    # Calculate the depolarizer, Mdelta
    mD_vect = np.zeros_like(P)  # Initialize multiplication of m * D_vect. It is an array of size (3,N)
    for i in range(0, N):
        mD_vect[:, i] = np.dot(m[:, :, i], D_vect[:, i])  # Multiplication
    Vd = P - mD_vect  # P - m * D_vect
    Pdelta = 1 / (a ** 2) * Vd[:3, :]  # 1/a^2 * (P-m*D) -> Goldstein 9-147
    Mdelta[1:, 0, :] = Pdelta  # Build Mdelta
    """
    Mdelta = np.transpose(M_, (1, 2, 0))  # Goldstein 9-142 is similar to Goldstein 9-145 (mdelta!=m')
    Mdelta[1:, 1:, :] = np.transpose(mdelta, (1, 2, 0))  # Goldstein 9-142 -> Reshape mdelta to 3x3xN and insert it as sub-matrix
    Mdelta = np.real(Mdelta)
    Mdelta = np.where(Mdelta <= 1, Mdelta, 1)
    Mdelta = np.where(Mdelta >= -1, Mdelta, -1)


    # Calculate the retarder, Mr
    # CN = np.max(np.linalg.cond(mdelta))  # Compute the CN in order to analyze Singular Matrices
    # print(np.shape(np.where(np.isinf(CN))))
    mdelta_I = np.linalg.pinv(mdelta)  # Inverse of mdelta
    mr = np.real(mdelta_I @ m_)  # -> Goldstein 9-153
    # -> Goldstein 9-153

    # Goldstein 9-153:
    k1m_mT = np.einsum('i,ijk->ijk', k1, m_mT)  # -> Goldstein 9-153 (first term)
    Mr1 = k1m_mT + k3I_3  # Goldstein 9-153 (first term)

    # TODO Conditional number para ver si es singular: si es singular Mr1_I, cuántos píxeles?
    # CN = np.max(np.linalg.cond(Mr1))  # Compute the CN in order to analyze Singular Matrices

    # print(np.shape(np.where(np.isinf(CN))))

    # Initialize Mr1_I
    Mr1_I = np.zeros_like(Mr1)  # Create matrix
    # Mr1_I = np.einsum('i,ijk->ijk', sign, np.linalg.inv(Mr1))  # Goldstein 9-153 (first term) -> Inverse of Mp1, multiplied by the sign of m'
    # print(np.shape(Mr1_I))
    # print(np.shape(Mr1))
    # k2m_ = np.einsum('i,ijk->ijk', k2, m_)  # -> Goldstein 9-153 (second term)
    # m_mTm_ = m_mT @ m_  # Matrix multiplication -> Goldstein 9-153
    # #m_mTm_ = np.einsum('i,ijk->ijk', m_mT, m_)  # -> Goldstein 9-153 (second term)
    # Mr2 = m_mTm_ + k2m_  # -> Goldstein 9-153 (first term)
    #
    # mr = Mr1_I @ Mr2  # Nx3x3 sub-matrix of Mp # -> Goldstein 9-153
    Mr[1:, 1:, :] = np.transpose(mr, (1, 2, 0))  # -> Goldstein 9-131 (Build Mr)
    Mr = np.where(Mr <= 1, Mr, 1)
    Mr = np.where(Mr >= -1, Mr, -1)

    return Mdelta, Mr, Md


def Arrow_decomposition(M):
    """Returns the small matrices (mro, mA,mri) from the Arrow decomposition: MM = M_RO*M_A*M_RI.

    Parameters:
        M (4x4xN array): Mueller matrix to decompose.

    Returns:
        mro (3x3xN array): small MM of retarders retarder
        mA (3x3xN array): diag(a1, a2, epsilon*a3)
        mri (3x3xN array): small MM of entrance retarder
    """

    m = M[1:, 1:, :]  # Get the small 3x3 (right-under) submatrix from the MM
    N = len(m[0, 0])  # nº of pixels
    m = np.moveaxis(m, -1, 0)  # Move N pixels to first dimension. Now shape of m is Nx3x3

    mro, mA, mri = np.linalg.svd(
        m)  # svd performs a singular value decomposition of the submatrix m, such that (A = U*S*V')

    mA_diag = np.zeros_like(mro)  # Initialize

    for i in range(0, N):  # Puts the values of mA in a diagonal matrix of shape Nx3x3
        mA_diag[i, :, :] = np.diag(mA[i, :])

    det_mri = np.linalg.det(mri)  # Determinants
    det_mro = np.linalg.det(mro)

    mA_diag[:, 2, 2] = det_mri * det_mro * mA_diag[:, 2, 2]  # The sign of the third component depends on the determinants of mri and mro
    for i in range(1, 3):
        mri[:, i, 2] = det_mri * mri[:, i, 2]
        mro[:, 2, i] = det_mro * mro[:, 2, i]

    mA = mA_diag

    mro = np.moveaxis(mro, 0, -1)  # Reshape back to 3x3xN
    mA = np.moveaxis(mA, 0, -1)
    mri = np.moveaxis(mri, 0, -1)

    return mro, mA, mri


def Characteristic_decomposition(M):
    """Returns an array of the matrices from the characteristic decomposition 
    (M0, M1, M2, M3), and an array with its coefficients, according to 
    J.J. Gil, Polarized Light (pag. 188, eq 5.81 and 5.82)

    Parameters:
        M (4x4xN array): Mueller matrix to decompose.

    Returns:
        coefficients (4xN array): Coefficients of the sum 
        matrices (4x4x4xN array): Matrices M0, M1, M2, M3
    """


    H = covariance_matrix(M)
    trH = H.trace(axis1=0, axis2=1).real # The trace must be real (it's an hermitic matrix)
    
    H = np.moveaxis(H, -1, 0)  # Reshape to (N,4,4)
    eigenvalues, eigenvectors = np.linalg.eigh(H) 
    
    # The function returns de eigenvalues in ascending order, so lambda0=eigenvalues[:,3], lambda1=eigenvalues[:,2], and so on
    
    # Coefficients
    lambda01 = (eigenvalues[:,3] - eigenvalues[:,2]) / trH
    lambda12 = 2 * (eigenvalues[:,2] - eigenvalues[:,1]) /trH
    lambda23 = 3 * (eigenvalues[:,1] - eigenvalues[:,0]) /trH
    lambda3 = 4 * eigenvalues[:,0] /trH
    
    coefficients = np.array([np.real(lambda01), np.real(lambda12), np.real(lambda23), np.real(lambda3)])
    
    # Matrices
    
    H0 = trH[:, np.newaxis, np.newaxis] * eigenvectors[:,:,3][:,:,None]@eigenvectors[:,:,3][:,None, :].conj()
    H1 = trH[:, np.newaxis, np.newaxis] / 2 * sum(eigenvectors[:,:,3-i][:,:,None]@eigenvectors[:,:,3-i][:,None, :].conj() for i in range (0,2))
    H2 = trH[:, np.newaxis, np.newaxis] / 3 * sum(eigenvectors[:,:,3-i][:,:,None]@eigenvectors[:,:,3-i][:,None, :].conj() for i in range (0,3))
    H3 = trH[:, np.newaxis, np.newaxis] / 4 * sum(eigenvectors[:,:,3-i][:,:,None]@eigenvectors[:,:,3-i][:,None, :].conj() for i in range (0,4))
    
    H0 = np.moveaxis(H0, 0, -1) #Reshape to obtain arrays of size (4x4xN)
    H1 = np.moveaxis(H1, 0, -1)
    H2 = np.moveaxis(H2, 0, -1)
    H3 = np.moveaxis(H3, 0, -1)
    
    M0 = MM_from_covariance(H0)
    M1 = MM_from_covariance(H1)
    M2 = MM_from_covariance(H2)
    M3 = MM_from_covariance(H3)
    
    matrices = np.array([M0, M1, M2, M3])

    return coefficients, matrices

"""
###################
###  P3 FILTER  ###
###################
"""

def P3_filter(M):
    """Returns the Mueller matrix with the P3 filter applied.

    Parameters:
        M (4x4xN array): Mueller matrix to filter.

    Returns:
        M_filt (4x4xN array): Mueller matrix P3-filtered.
    """

    N = len(M[0, 0])  # Number of pixels
    P3 = IPPs(M)[2]
    one = np.ones_like(P3)  # Array of 1's
    Mdiag = identity(4, N)

    one_P3 = one - P3
    Mdiag = np.einsum('i,ijk->ijk', one_P3, Mdiag)  # (1-P3) * Mdiag

    M_filt = M - Mdiag

    return M_filt



"""
#################
####  PLOTS  ####
#################
"""


def plot_Pobs(param, Nx=None, Ny=None, title=None, save=False, norm=False):
    """Plots a given polarimetric observable.

    Parameters:
        param (1D array): Polarimetric observable to plot
        Nx, Ny (int): Size of the image
        title (str): Title of the plot. Default no title
        save (bool): True if want to save the plot. Default False

    Returns:
        Plot of the polarimetric observable
    """

    plt.rcParams['figure.dpi'] = 300  # Change to get a good quality plot. Otherwise is very blurry
    plt.rcParams['savefig.dpi'] = 300

    if Nx and Ny is None:
        n = int(np.sqrt(len(param)))  # Size of the image. Only works for square images
        param = np.reshape(param, (n, n))  # Reshape to the original image size

    else:
        # Size of the image. In case it's not square
        param = np.reshape(param, (Nx, Ny))  # Reshape to the original image size

    param = param.T  # Transpose


    # Maximum and Minimum values for the representation depending on the normalization:
    if norm is True:
        vmin = 0
        vmax = 1
    else:
        vmin = np.min(param)
        vmax = np.max(param)

    plt.figure(figsize=(3, 3), num=title, clear=True, layout='compressed')
    plt.imshow(param, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.colorbar(cmap='RdBu_r')

    if title is not None:  # Title of the image
        plt.axis('off')
        plt.title(title, fontsize=7, style="italic")

    if save is True:
        plt.savefig(f"{title}.png")  # Save the plot

    plt.show()
    plt.close('all')



def plot_Mueller(M, Nx=None, Ny=None, title='Mueller matrix', save=False, norm=False):
    """Plots the whole Mueller matrix as: M00 M01 M02 M03
                                          M10 M11 M12 M13
                                          M20 M21 M22 M23
                                          M30 M31 M32 M33
    Parameters:
        M (4x4xN array): Mueller matrix to plot.
        Nx, Ny (int): Size of the image
        title (str): Set plot title. Default: 'Mueller matrix'
        save (bool): If True downloads the plot. Default False
        norm (bool): If True vmin = -1 and vmax = 1. Default False

    Returns:
        Plot of the Mueller matrix
    """

    if Nx is None and Ny is None:
        n = int(np.sqrt(np.shape(M)[2]))  # Size of the image. Only works for square images


        M00 = M[0, 0, :].reshape((n, n)).T
        M01 = M[0, 1, :].reshape((n, n)).T
        M02 = M[0, 2, :].reshape((n, n)).T
        M03 = M[0, 3, :].reshape((n, n)).T
        M10 = M[1, 0, :].reshape((n, n)).T
        M11 = M[1, 1, :].reshape((n, n)).T
        M12 = M[1, 2, :].reshape((n, n)).T
        M13 = M[1, 3, :].reshape((n, n)).T
        M20 = M[2, 0, :].reshape((n, n)).T
        M21 = M[2, 1, :].reshape((n, n)).T
        M22 = M[2, 2, :].reshape((n, n)).T
        M23 = M[2, 3, :].reshape((n, n)).T
        M30 = M[3, 0, :].reshape((n, n)).T
        M31 = M[3, 1, :].reshape((n, n)).T
        M32 = M[3, 2, :].reshape((n, n)).T
        M33 = M[3, 3, :].reshape((n, n)).T

    else:
        # Nx and Ny are the size of the image. In case it's not square
        M00 = M[0, 0, :].reshape((Nx, Ny)).T
        M01 = M[0, 1, :].reshape((Nx, Ny)).T
        M02 = M[0, 2, :].reshape((Nx, Ny)).T
        M03 = M[0, 3, :].reshape((Nx, Ny)).T
        M10 = M[1, 0, :].reshape((Nx, Ny)).T
        M11 = M[1, 1, :].reshape((Nx, Ny)).T
        M12 = M[1, 2, :].reshape((Nx, Ny)).T
        M13 = M[1, 3, :].reshape((Nx, Ny)).T
        M20 = M[2, 0, :].reshape((Nx, Ny)).T
        M21 = M[2, 1, :].reshape((Nx, Ny)).T
        M22 = M[2, 2, :].reshape((Nx, Ny)).T
        M23 = M[2, 3, :].reshape((Nx, Ny)).T
        M30 = M[3, 0, :].reshape((Nx, Ny)).T
        M31 = M[3, 1, :].reshape((Nx, Ny)).T
        M32 = M[3, 2, :].reshape((Nx, Ny)).T
        M33 = M[3, 3, :].reshape((Nx, Ny)).T

    plt.rcParams['figure.dpi'] = 300  # Change to get a good quality plot. Otherwise is very blurry
    plt.rcParams['savefig.dpi'] = 300

    # Maximum and Minimum values for the representation depending on the normalization:
    if norm is True:
        vmin = -1
        vmax = 1
    else:
        vmin = -np.max(M00)
        vmax = np.max(M00)

    # Create Multiple images for the MM image:
    fig = plt.figure(figsize=(4.5, 3.5), num=title, clear=True, layout='compressed')  # layout='constrained')#

    # Adds subplots # cmap='seismic' or 'RdBu_r'
    fig.add_subplot(4, 4, 1)
    plt.imshow(M00, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M00", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 2)
    plt.imshow(M01, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M01", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 3)
    plt.imshow(M02, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M02", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 4)
    plt.imshow(M03, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M03", fontsize=3, style="italic")

    fig.add_subplot(4, 4, 5)
    plt.imshow(M10, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M10", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 6)
    plt.imshow(M11, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M11", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 7)
    plt.imshow(M12, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M12", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 8)
    plt.imshow(M13, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M13", fontsize=3, style="italic")

    fig.add_subplot(4, 4, 9)
    plt.imshow(M20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M20", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 10)
    plt.imshow(M21, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M21", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 11)
    plt.imshow(M22, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M22", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 12)
    plt.imshow(M23, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M23", fontsize=3, style="italic")

    fig.add_subplot(4, 4, 13)
    plt.imshow(M30, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M30", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 14)
    plt.imshow(M31, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M31", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 15)
    plt.imshow(M32, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M32", fontsize=3, style="italic")
    fig.add_subplot(4, 4, 16)
    plt.imshow(M33, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title("M33", fontsize=3, style="italic")

    cax = plt.axes((0.85, 0.1, 0.015, 0.8))  # Colorbar position
    plt.colorbar(cax=cax, cmap='RdBu_r')

    if save is True:
        plt.savefig(f"{title}.png")  # Save the plot

    plt.show()
    plt.close('all')


def plot_IPPtetrahedron(P1, P2, P3):
    """Plots the IPPs distribution in the tetrahedral physically feasible region.

    Parameters:
        P1, P2, P3 (1D array) each: IPPs

    Returns:
        Plot of the IPPs distribution. and the feasible region
    """

    ### Customization ###
    points_color = 'winter'  # Color of the IPP points
    tetraedron_color = '#800000'
    line_color = 'orange'  # Color of the edges of the tetraedron
    ####################

    # Create 3D figure
    plt.figure('SPLTV', figsize=(10, 5))
    ax = plt.axes(projection='3d')

    ax.axes.set_xlim3d(left=0, right=1)  # Axis limits
    ax.axes.set_ylim3d(bottom=0, top=1)
    ax.axes.set_zlim3d(bottom=0, top=1)

    # Vertices of the tetraedron
    p0 = np.array([0, 0, 0])  # Coordinates
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 1, 1])
    p3 = np.array([1, 1, 1])

    # Edges of the tetraedron
    x1, y1, z1 = [p0[0], p2[0]], [p0[1], p2[1]], [p0[2], p2[2]]  # Line from p0 to p2
    x2, y2, z2 = [p0[0], p3[0]], [p0[1], p3[1]], [p0[2], p3[2]]  # p0-p3
    x3, y3, z3 = [p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]]  # p2-p3
    x4, y4, z4 = [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]  # p0-p1
    x5, y5, z5 = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]  # p1-p2
    x6, y6, z6 = [p1[0], p3[0]], [p1[1], p3[1]], [p1[2], p3[2]]  # p1-p2

    # Plot lines created above
    ax.plot(x1, y1, z1, color=line_color)
    ax.plot(x2, y2, z2, color=line_color)
    ax.plot(x3, y3, z3, color=line_color)
    ax.plot(x4, y4, z4, color=line_color)
    ax.plot(x5, y5, z5, color=line_color)
    ax.plot(x6, y6, z6, color=line_color)

    # Planes of the tetraedron
    verts1 = [((p0), (p2), (p3))]  # Vertices of the planes
    verts2 = [((p0), (p1), (p2))]
    verts3 = [((p0), (p1), (p3))]
    verts4 = [((p1), (p2), (p3))]

    srf1 = Poly3DCollection(verts1, alpha=.25, facecolor=tetraedron_color)  # Create the surfaces
    srf2 = Poly3DCollection(verts2, alpha=.25, facecolor=tetraedron_color)
    srf3 = Poly3DCollection(verts3, alpha=.25, facecolor=tetraedron_color)
    srf4 = Poly3DCollection(verts4, alpha=.25, facecolor=tetraedron_color)

    plt.gca().add_collection3d(srf1)  # Add surfaces to the figure
    plt.gca().add_collection3d(srf2)
    plt.gca().add_collection3d(srf3)
    plt.gca().add_collection3d(srf4)

    # Labels
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')

    # IPPs data
    ax.scatter3D(P1, P2, P3, c=P3, cmap=points_color, s=1, alpha=0.5)

    # Show plot
    # ax.view_init(30, 0)  # Change view uncommenting and changing angles
    plt.show()

    # For a 3d interactive plot:
    # Before running the function plot_IPPtetrahedron(), run in the CONSOLE the
    # following command: %matplotlib qt


def select_pixels_image (Image, Nx = None, Ny = None, title = None, save = False):
    """Select a number of pixels of a given polarimetric observable for analyzing a particular ROI (region of interest).

    Parameters:
        Image (1D array): Polarimetric observable to plot
        Nx, Ny (int): original size of the image
        title (str): Title of the plot. Default no title
        save (bool): True if want to save the plot. Default False

        Returns:
        punts (list of int)
        size_image (ndarray: {2,}): size of the image
        Plot of the polarimetric observable and selected pixels
    """

    if Nx and Ny is None:
        n = int(np.sqrt(len(Image)))  # Size of the image. Only works for square images
        Image = np.reshape(Image, (n, n))  # Reshape to the original image size
        Nx = n
        Ny = n

    else:
        # Size of the image. In case it's not square
        Image = np.reshape(Image, (Nx, Ny))  # Reshape to the original image size

    # Show the matrix as a plot
    fig, ax = plt.subplots()
    cax = ax.matshow(Image, cmap='viridis')

    punts = []

    def on_click(event):
        # Obtain the click coordinates
        x, y = int(event.xdata), int(event.ydata)
        print(f"Selected: ({x}, {y})")

        # Save the coordinates
        punts.append((x, y))

        # Show the selected pixel on the Image
        ax.plot(x, y, 'ro')  # red and circular
        fig.canvas.draw()

    # Connect the click event to the function
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    if title is not None:  # Title of the image
        plt.axis('off')
        plt.title(title, fontsize=7, style="italic")

    if save is True:
        plt.savefig(f"{title}.png")  # Save the plot

    plt.show()

    # Print the selected pixels
    print("Selected Pixels:", punts)

    size_image = np.array(punts[1]) - np.array(punts[0])

    return punts, size_image

# FUNCION PARA RETARDO

def calculate_retardance(MM_retarder):
    """
    Calcula la Retardancia Total (R) de una matriz de Mueller de retardo puro.
    
    Parámetros:
        MM_retarder (4x4xN array): Componente de retardo (MR) o matriz de Mueller.

    Devuelve:
        R (N array): Retardancia total en grados.
    """
    
    # Extraer la submatriz 3x3 de retardo (m_R). Shape es (3, 3, N)
    m_R = MM_retarder[1:4, 1:4, :]
    
    # Calcular la traza de m_R
    Tr_mR = np.trace(m_R, axis1=0, axis2=1).real
    
    # Aplicar la fórmula R = arccos((Tr(m_R) - 1) / 2)
    arg_arccos = (Tr_mR - 1) / 2
    
    # Limitar el argumento a [-1, 1] para evitar errores de np.arccos
    arg_arccos = np.clip(arg_arccos, -1.0, 1.0)
    
    # Retardancia en grados
    R_deg = np.arccos(arg_arccos) * 180 / np.pi
    
    return R_deg
