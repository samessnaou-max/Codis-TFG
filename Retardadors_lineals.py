# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 19:01:21 2025

@author: Usuario
"""


import numpy as np
import PolarimetricFunctions as pf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import glob
import os

#import pymueller
#from pymueller.decomposition import lu_chipman

import Lu_Chip as lc

import matplotlib.animation as animation
from matplotlib.widgets import Button


import warnings

# Suppress warnings for invalid values in sqrt
warnings.filterwarnings("ignore", category=RuntimeWarning)


import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


plt.rcParams.update({
    "font.family": "serif", 
    "mathtext.fontset": "cm", 
    "mathtext.rm": "serif",
    "axes.labelsize": 15,
    "font.size": 11,
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10,
	"figure.titlesize": 11
})

 



#%% Functions

list_parameters = ['ret', 'diat', 'pol', 'pdelta', 'ps', 'coeff', 'ipps']
subtitles = {'Retardance': r'$\Delta$', 'Diattenuation': r'$D$', 'Polarizance': r'$\mathcal{P}$', 'Pdelta': r'$P_\Delta$', 'Ps': r'$P_S$', 'Coefficients': r'$c$'}

#%%% Plots

def plot_IPPS(ipps, title='',  x=None, y=None, xtitle=None, ytitle=None, name=None, lims=[0,1], num=3, origin='lower',rot=False, color='viridis'):
    """
    Plots a heatmap of the IPPS.

    Parameters:
        ipps (3xNxN array): Values of the ipps P1, P2, P3
        title (string): Title of the plot
        x (N,) array: X axis values
        y (N,) array: Y axis value

    Returns:
        Shows a plot
    """
    
    if x is None:
        extent = None
        axis = 'off'
        heatmap_width = 3.3
        heatmap_height = 3.1
    else:
        extent = [x.min(), x.max(), y.min(), y.max()]
        axis='on'
        heatmap_width = 3.5
        heatmap_height = 3.1
    
    

    if rot==True:
        ipps=ipps.transpose(0,2,1)


    if num == 3:
        fig, axes = plt.subplots(1, 3, figsize=(heatmap_width * 3, heatmap_height))
        
    elif num == 2:
        fig, axes = plt.subplots(1, 2, figsize=(heatmap_width * 2, heatmap_height))
 
        

    # Plot each colormap
    for i, ax in enumerate(axes.flat):  
        vmin=lims[0] if lims is not None else None
        vmax=lims[1] if lims is not None else 1 if np.max(ipps[i])>1 else None
        
    
        
        im = ax.imshow(
            ipps[i], aspect="auto", extent=extent,
            origin=origin, cmap=color, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"$P_{i+1}$")  # Set title for each subplot
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.axis(axis)
        fig.colorbar(im, ax=ax)  # Add colorbar for each subplot

    fig.suptitle(title)
    
    #Adjust layout
    if title!='':
        plt.tight_layout(rect=[0, 0, 1, 1.08])
    else:
        plt.tight_layout(rect=[0, 0, 1, 1])
        

        
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    
    plt.show()



def plot_parameter(parameter, title='', x=None, y=None, xtitle=None, ytitle=None, subtitle='M',lims=[0,1], name=None, num=3, rot=False, origin='lower', color='viridis'):
    """
    Plots the specified parameter for M0, M1, M2, M3

    Parameters:
        parameter (mxNxN): Parameter to plot
        
        title (string) : Title of the plot
        
        x (N,) array: x axis values
        
        y (N,) array: y axis values
        
        xtitle (string): x axis title
        
        ytitle (string): y axis title
        
        
        
    Returns:
        Shows a plot

    """

    if x is None:
        extent = None
        axis = 'off'
        heatmap_width = 3.3
        heatmap_height = 3.1
    else:
        extent = [x.min(), x.max(), y.min(), y.max()]
        axis='on'
        heatmap_width = 3.5
        heatmap_height = 3.1
        
         
    
    if num == 3:
        fig, axes = plt.subplots(1, 3, figsize=(heatmap_width * 3, heatmap_height))
        
    elif num == 4:
        fig, axes = plt.subplots(1, 4, figsize=(heatmap_width * 4, heatmap_height))
        
    elif num == 2:
        fig, axes = plt.subplots(1, 2, figsize=(heatmap_width * 2, heatmap_height))
        
    elif num == 1:
        fig, ax = plt.subplots(1, 1, figsize=(heatmap_width, heatmap_height))
        axes=np.array([[ax]])
        parameter = parameter[np.newaxis, :, :] 
    else: 
        print("Number not valid. It must be 1-4")
        return
    
    if rot==True:
        parameter=parameter.transpose(0,2,1)

    


    vmin=lims[0] if lims is not None else None
    vmax=lims[1] if lims is not None else None  
        
    for i, ax in enumerate(axes.flat):  # Iterate over the subplots
        
        im = ax.imshow(
            parameter[i], aspect="auto", extent=extent, 
            origin=origin, cmap=color, vmin=vmin, vmax=vmax
        )
        
        ax.set_title(subtitle+f"$_{i}$" if num!=1 else '')  # Set title for each subplot
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.axis(axis)
        fig.colorbar(im, ax=ax)


    if num==1:
         ax.set_title(title, loc='left')
    else:
         fig.suptitle(title, ha='center')

    #Adjust layout
    if num==3:
        plt.tight_layout(rect=[0, 0, 1, 1] if title=='' else [0, 0, 1, 1.08])
        
    if num==2:
        plt.tight_layout(rect=[0, 0, 1, 1])
    
    if num==4:
        plt.tight_layout(rect=[0, 0, 1, 1.02])
    if num==1:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
   
    
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_dif_IPPS(ipps, title='',  x=None, y=None, xtitle=None, ytitle=None, name=None, lims=[0,1], origin='lower',rot=False, color='viridis'):
    """
    Plots a heatmap of the difference between IPPS.

    Parameters:
        ipps (3xNxN array): Values of the ipps P1, P2, P3
        title (string): Title of the plot
        x (N,) array: X axis values
        y (N,) array: Y axis value

    Returns:
        Shows a plot
    """
    
    heatmap_width = 3.3
    heatmap_height = 2.7
    

    if rot==True:
        ipps=ipps.transpose(0,2,1)
    
    if x is None:
        extent = None
        axis = 'off'
    else:
        extent = [x.min(), x.max(), y.min(), y.max()]
        axis='on'

    fig, axes = plt.subplots(1, 2, figsize=(heatmap_width * 2, heatmap_height))
   
        

    # Plot each colormap
    for i, ax in enumerate(axes.flat):  
        vmin=lims[0] if lims is not None else None
        vmax=lims[1] if lims is not None else 1 if np.max(ipps[i])>1 else None
        
    
        im = ax.imshow(
            ipps[i+1]-ipps[i], aspect="auto", extent=extent,
            origin=origin, cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax.set_title(f"P{i+2}-P{i+1}")  # Set title for each subplot
        ax.set_xlabel(xtitle)
        ax.axis(axis)
        ax.set_ylabel(ytitle)
        fig.colorbar(im, ax=ax)  # Add colorbar for each subplot

    fig.suptitle(title)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1.06])
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
#%%% Computation of parameters   
def parameters_ChDecomp(data, height, width, parameters):
    """
    Computes (if selected) the IPP values of the introduced matrices, as well 
    as the coefficients from its characteristic decomposition, and the 
    parameters of the matrices obtained from said decomposition of the 
    introduced Mueller matrices.

    Parameters
    ----------
    data (4x4xN array): Collection of Mueller matrices
    
    height (int) : Height of the image
    
    width (int) : Width of the image
    
    parameters (array): List of parameters we want to compute, to select between: 
        ret, diat, pol, pdelta, ps, coeff, ipps

    Returns
    -------
    Dictionary of parameters, each entry is a (3 x height x width)

    """
    
    ret=1 if 'ret' in parameters else None
    diat=1 if 'diat' in parameters else None
    polar=1 if 'pol' in parameters else None
    pd=1 if 'pdelta' in parameters else None
    coeffs=1 if 'coeff' in parameters else None
    ipp=1 if 'ipps' in parameters else None
    ps=1 if 'ps' in parameters else None
    
    
    dim =np.shape(data)[2]
    if height*width!=  dim:
        print("Height and width do not correspond with the dimension of the data")
        return None

    retardance = np.zeros((3, height, width))
    polarizance = np.zeros((3, height, width))
    diattenuation = np.zeros((3, height, width))
    pdelta = np.zeros((3, height, width))
    sphericalpty = np.zeros((3, height, width))

    coefficients = np.zeros((4, height, width))
    IPPS = np.zeros((3, height, width))

    for j in range(dim):
        M_total = data[:,:, j]
        M_total = M_total.reshape(4,4,1)
        
        y = j % width
        x = j // width
        
        if ipp==1:
            ipps = pf.IPPs(M_total)
            IPPS[0, y, x] = ipps[0][0]
            IPPS[1, y, x] = ipps[1][0]
            IPPS[2, y, x] = ipps[2][0]
        
        if 1 in [ret, diat, polar, pd, coeffs, ps]:
            coefs, matrius = pf.Characteristic_decomposition(M_total)
            
            coefficients[3, y, x] = coefs[3,0] if coeffs==1 else None
            
            for i in range(3):
                matriu = matrius[i]
                
                coefficients[i, y, x] = coefs[i,0] if coeffs==1 else None
                #retards[i, nt, npy] = pf.Retardance(matriu)[0]
                
                if ret==1:
                    try:
                        r=lc.Lu_Chip(matriu)[2][0,0]
                    except:
                        r=float('NaN')
                    
                    retardance[i, y, x] = r*180/np.pi
                
                if polar==1:
                    polarizance[i, y, x] = pf.Polarizance(matriu/matriu[0,0,0])[1][0]
                if diat==1:
                    diattenuation[i, y, x] = pf.Diattenuation(matriu/matriu[0,0,0])[1][0] 
                if pd==1:
                    pdelta[i, y, x] = pf.Pdelta(matriu)[0] 
                if ps==1:
                    sphericalpty[i,y,x] = pf.Ps(matriu/matriu[0,0,0])[0] 
                
            
    result = {}

    if ret == 1:
        result["Retardance"] = retardance
    if polar == 1:
        result["Polarizance"] = polarizance
    if diat == 1:
        result["Diattenuation"] = diattenuation
    if pd == 1:
        result["Pdelta"] = pdelta
    if coeffs == 1:
        result["Coefficients"] = coefficients
    if ipp == 1:
        result["IPPS"] = IPPS
    if ps == 1:
        result['Ps'] = sphericalpty
        
    return result     
    

def parameters_MM (data, height, width, parameters):
    
    dim =np.shape(data)[2]
    if height*width!=  dim:
        print("Height and width do not correspond with the dimension of the data")
        return None
    
    ret=1 if 'ret' in parameters else None
    diat=1 if 'diat' in parameters else None
    polar=1 if 'pol' in parameters else None
    pd=1 if 'pdelta' in parameters else None
    ps=1 if 'ps' in parameters else None
    
    retardance = np.zeros((height, width))
    polarizance = np.zeros((height, width))
    diattenuation = np.zeros((height, width))
    pdelta = np.zeros((height, width))
    sphericalpty = np.zeros((height, width))
    
    for j in range(dim):
        M_total = data[:,:, j]
        matriu = M_total.reshape(4,4,1)
        
        y = j % width
        x = j // width
    
        if ret==1:
            try:
                r=lc.Lu_Chip(matriu)[2][0,0]
            except:
                r=float('NaN')
            
            retardance[y, x] = r*180/np.pi
        
        if polar==1:
            polarizance[y, x] = pf.Polarizance(matriu/matriu[0,0,0])[1][0]
        if diat==1:
            diattenuation[y, x] = pf.Diattenuation(matriu/matriu[0,0,0])[1][0] 
        if pd==1:
            pdelta[y, x] = pf.Pdelta(matriu)[0] 
        if ps==1:
            sphericalpty[y,x] = pf.Ps(matriu/matriu[0,0,0])[0] 

    
    result ={}
    
    if ret == 1:
        result["Retardance"] = retardance
    if polar == 1:
        result["Polarizance"] = polarizance
    if diat == 1:
        result["Diattenuation"] = diattenuation
    if pd == 1:
        result["Pdelta"] = pdelta

    if ps == 1:
        result['Ps'] = sphericalpty
    
    return result
    

#%%% TETRAEDRE

def plot_IPPtetrahedron(P1, P2, P3, color_data, color_label=r'$\mathbf{p}_y$'):
    """
    Plots the IPPs distribution in the tetrahedral physically feasible region.

    Parameters:
        P1, P2, P3 (1D array): IPPs values.
        color_data (1D array): Data used to color the points (e.g., py).
        color_label (string): Label for the color bar.

    Returns:
        Plot of the IPPs distribution and the feasible region.
    """
    
    ### Customization ###
    points_color = 'viridis'  # Colormap
    tetraedron_color = '#B0B0B0'
    line_color = 'black'
    ####################

    fig = plt.figure('SPLTV', figsize=(8, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')


    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1.05)

    try:
        ax.set_box_aspect((1, 1, 1.05))
    except:
        pass

    # Vertices del tetraedro
    p0 = np.array([0, 0, 0])
    p1 = np.array([0, 0, 1])
    p2 = np.array([0, 1, 1])
    p3 = np.array([1, 1, 1])

    # Aristas del tetraedro
    edges = [
        (p0, p2), (p0, p3), (p2, p3),
        (p0, p1), (p1, p2), (p1, p3)
    ]
    for a, b in edges:
        xs, ys, zs = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]
        ax.plot(xs, ys, zs, color=line_color, linewidth=1.2)

    # Planos del tetraedro
    verts = [
        (p0, p2, p3),
        (p0, p1, p2),
        (p0, p1, p3),
        (p1, p2, p3),
    ]
    for v in verts:
        srf = Poly3DCollection([v], alpha=.18, facecolor=tetraedron_color, edgecolor=line_color)
        ax.add_collection3d(srf)

    # Etiquetas de ejes 3D
    ax.set_xlabel('$P_1$', labelpad=10)
    ax.set_ylabel('$P_2$', labelpad=10)
    ax.set_zlabel('$P_3$', labelpad=12)

    sc = ax.scatter(
        P1, P2, P3, 
        c=color_data, 
        cmap=points_color, 
        s=20, 
        alpha=0.95
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)
    
    cbar.set_label(color_label, rotation=0, y=-0.05, labelpad=-10, ha='center') 

    #plt.tight_layout()
    plt.show()


#%%% Simulations

def simulation_retarders(p,  N=300, N_Thetas=200, N_Phis=200, Maxsigma_theta=np.pi):
    """
    Parameters
    ----------
    p : float between 0 and 1
        Percentage of photons interacting by isotropic scattering
    px : float between 0 and 1
        Px fixed value for each diattenuator.
    N : int, optional 
        Number of interactions. The default is 300.
    N_Thetas : int, optional
        Number of variations for sigma Theta. The default is 100.
    N_Py : int, optional
        Number of variations for Py. The default is 100.
    Maxsigma_theta: float, optional
        Maximum deviation (in radians)

    Returns
    -------
    MMD: (N_Thetas, N_Py, 4, 4) array
        Matrices resulting from the simulation

    """    
    
        
    # Definition of parameters
    N_B = int(p * N)  # Number of interactions via isotropic scattering (interaction B)
    N_A = N - N_B     # Number of interactions via other means (interaction A)
    
    sigmaTheta = np.linspace(0, Maxsigma_theta, N_Thetas)  # Deviation values
    ThetaM = (np.pi / 180) * 20  # Mean value of theta (set to 60 degrees)

    
    # Parameters of the retarder (phi)
    Maxsigma_phi = 2*np.pi
    sigmaPhi = 0*np.linspace(0, Maxsigma_phi, N_Phis)
    PhiM = 1*np.linspace(0, np.pi*2, N_Phis) 
    
    # Calculation
    
    # Initialize matrices
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
    A = np.zeros((4, 4, N_A)) 
     
    MMR = np.zeros((N_Thetas, N_Phis, 4, 4))  
    
    k = 0
    for nt in range(N_Thetas):  
        
        theta = ThetaM + sigmaTheta[nt] * np.random.randn(N_A)  
    
        RN = pf.rotator_MM(-theta)  
        RP = pf.rotator_MM(theta)  
    
        for nph in range(N_Phis):  
            k += 1  
    
            phi = PhiM[nph] + sigmaPhi[nph] * np.random.randn(N_A)
            
            M1 = pf.retarder_MM(phi)  
    
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)  # The matrix multiplication assumes the matrices are in the 2nd and 3rd dimension
            # A has shape (N_A, 4, 4)
            
            # --------------------------- TOTAL MATRIX ---------------------------
            A_T = np.sum(A, axis=0)  # Incoherent sum of all interaction matrices A
            B_T = N_B * M_dis  # Matrices corresponding to interaction B
            M_total = A_T + B_T  
    
            MMR[nt, nph, :, :] = M_total
        
    return MMR, PhiM*180/np.pi, sigmaTheta*180/np.pi


def simulation_retarders_sigma_phi_vs_theta(p, N=3000, N_Thetas=200, N_Phis=200, Maxsigma_theta=np.pi):
    """
    Simula matrius de Mueller variant la dispersió de la retardància (sigma_phi)
    contra l'angle mitjà de l'eix ràpid (ThetaM).
    
    Parameters
    ----------
    p : float between 0 and 1
        Percentage of photons interacting by isotropic scattering
    ... (altres paràmetres) ...

    Returns
    -------
    MMD: (N_Thetas, N_Phis, 4, 4) array
        Matrices resulting from the simulation
    
    sigmaPhi: (N_Phis,) array
        Valors de sigma_phi utilitzats (eix X)
    
    ThetaM_array: (N_Thetas,) array
        Valors de ThetaM utilitzats (eix Y)
    """    
        
    # Definition of parameters
    N_B = int(p * N)
    N_A = N - N_B
    
    MaxThetaM = np.pi 
    ThetaM_array = np.linspace(0, MaxThetaM, N_Thetas) 
    sigmaTheta_const = np.pi / 4  
    
    Maxsigma_phi = 2*np.pi
    sigmaPhi = 0*np.linspace(0, Maxsigma_phi, N_Phis) 
    PhiM_const = np.pi 
    
    # Calculation
    
    # Initialize matrices
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
    A = np.zeros((4, 4, N_A)) 
     
    MMR = np.zeros((N_Thetas, N_Phis, 4, 4))  
    
    k = 0
    for nt in range(N_Thetas):
        
        ThetaM = ThetaM_array[nt]
        
        theta = ThetaM + sigmaTheta_const * np.random.randn(N_A)
        
        RN = pf.rotator_MM(-theta)
        RP = pf.rotator_MM(theta)
        
        for nph in range(N_Phis):
            k += 1
            
            phi = PhiM_const + sigmaPhi[nph] * np.random.randn(N_A)
            
            M1 = pf.retarder_MM(phi)
            
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)
            
            A_T = np.sum(A, axis=0)
            B_T = N_B * M_dis
            M_total = A_T + B_T
            
            MMR[nt, nph, :, :] = M_total
            
    return MMR, ThetaM_array*180/np.pi, sigmaPhi*180/np.pi


def simulation_retarders3(p,  N=300, N_Thetas=200, N_Phis=200, Maxsigma_theta=np.pi):
    """
    Parameters
    ----------
    p : float between 0 and 1
        Percentage of photons interacting by isotropic scattering
    px : float between 0 and 1
        Px fixed value for each diattenuator.
    N : int, optional 
        Number of interactions. The default is 300.
    N_Thetas : int, optional
        Number of variations for sigma Theta (eix Y). The default is 100.
    N_Phis : int, optional
        Number of variations for sigma Phi (eix X). The default is 100.
    Maxsigma_theta: float, optional
        Maximum deviation (in radians)

    Returns
    -------
    MMD: (N_Thetas, N_Phis, 4, 4) array
        Matrices resulting from the simulation
    
    sigmaPhi: (N_Phis,) array
        Valors de sigma_phi utilitzats (eix X)
    
    sigmaTheta: (N_Thetas,) array
        Valors de sigma_theta utilitzats (eix Y)
    """    
    
        
    # Definition of parameters
    N_B = int(p * N)  # Number of interactions via isotropic scattering (interaction B)
    N_A = N - N_B     # Number of interactions via other means (interaction A)
    
    sigmaTheta = np.linspace(0, Maxsigma_theta, N_Thetas)  # Deviation values (0 a 180 graus)
    ThetaM = (np.pi / 180) * 20  # Mean value of theta (constant a 20 graus)

    Maxsigma_phi = 2*np.pi
    sigmaPhi = 0*np.linspace(0, Maxsigma_phi, N_Phis)
    
    PhiM_const = np.pi 
    
    # Calculation
    
    # Initialize matrices
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
    A = np.zeros((4, 4, N_A)) 
     
    MMR = np.zeros((N_Thetas, N_Phis, 4, 4))  
    
    k = 0
    for nt in range(N_Thetas):  
        theta = ThetaM + sigmaTheta[nt] * np.random.randn(N_A)  
    
        RN = pf.rotator_MM(-theta)  
        RP = pf.rotator_MM(theta)  
    
        for nph in range(N_Phis):  
            k += 1  
    
            phi = PhiM_const + sigmaPhi[nph] * np.random.randn(N_A)
            
            M1 = pf.retarder_MM(phi)  
    
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)
            
            # --------------------------- TOTAL MATRIX ---------------------------
            A_T = np.sum(A, axis=0)  # Incoherent sum of all interaction matrices A
            B_T = N_B * M_dis  # Matrices corresponding to interaction B
            M_total = A_T + B_T  
    
            MMR[nt, nph, :, :] = M_total
        
    return MMR, sigmaPhi*180/np.pi, sigmaTheta*180/np.pi

#%%% Processing
def process_data(MM, params_list, height, width, x=None, y=None, xtitle=None, ytitle=None, general=False, chdecomp=True, save=False, filename=None, plotsdir=''):
    """
    Processes the given MMs and returns the plots of the specified parameters

    Parameters
    ----------
    MM : (n, m, 4, 4) 
        DESCRIPTION.
    params_list : array
        List of parameters we want to calculate to choose between:
            'diat','ret', 'pol', 'pdelta', 'ps', 'coeff', 'ipps'
    filename: string, optional
        If we want to save the plot, name by which we want it to be saved
    save: bool, optional
        Whether we want to save txt files with the data obtained or not
        The default is False.
    plotsdir : string, optional
        If we want to specify a route to where the files should be saved. The default is ''.

    Returns
    -------
    None.

    """
    
    if MM.shape[-2:] == (4, 4):
        if len(MM.shape) == 4:
            MM = MM.reshape(height*width, 4, 4).transpose(1, 2, 0)
        elif len(MM.shape) != 3:
            print("Unexpected shape")
    else:
        print("Not a 4x4 matrix in the last dimensions")
            
    
    if chdecomp==True:
        params_chd = parameters_ChDecomp(MM, height, width, params_list)     
    
        if filename==None:
            for param in params_chd.keys():
                if param=='IPPS':
                    plot_IPPS(params_chd[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle)
                elif param=='Retardance':
                    plot_parameter(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180])
                else:
                    plot_parameter(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle)
    
        else:
            for param in params_chd.keys():
                if param=='IPPS':
                    plot_IPPS(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, name=plotsdir+filename+'_'+param)
                elif param=='Retardance':
                    plot_parameter(params_chd[param],subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], name=plotsdir+filename+'_'+param)
                else:
                    plot_parameter(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, name=plotsdir+filename+'_'+param)

        
        #If specified, save the IPP data    
        if save==True:
            np.save("Data/"+filename, params_chd['IPPS'])
            np.save("Data/Inc_PhiM", y)
            np.save("Data/Inc_PyM", x)
            
    if general==True:
        params_gen = parameters_MM(MM, width, height, params_list)
        
        #Plot of the parameters of the original MM
        for key in params_gen.keys():
            if filename==None:
                plot_parameter(params_gen[param], param, x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1)
            
            else:
                plot_parameter(params_gen[param], param, x=x, y=y, xtitle=xtitle, ytitle=ytitle, name=plotsdir+filename+'_'+param, num=1)
                
    return None

#%% Retarders sigma_theta vs phi --> simulation_retarders

filename="R_p0"
xtitle=r'$\Delta$'
ytitle = r'$\sigma_\theta$'

MMR,x,y=simulation_retarders(0.2)

height=len(y)
width=len(x)
MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
data_chd_re = parameters_ChDecomp(MMR, len(y), len(x), ['diat', 'ret', 'pol', 'coeff', 'ps', 'ipps'])
data_gen_re = parameters_MM(MMR, len(y), len(x), ['diat', 'ret', 'pdelta', 'ps','pol'])

#%%%% Plots Ch Decomp sigma_theta vs phi --> simulation_retarders

save = False
    
plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=2, xtitle=xtitle, ytitle=ytitle, rot=True, name = filename +'_IPPS2' if save is True else None)
plot_parameter(data_chd_re['Retardance'],  x=x, y=y,  xtitle=xtitle, ytitle=ytitle, subtitle=r'$\Delta$', lims=[0,180], rot=True, name= filename +'_DRet' if save is True else None)
# plot_parameter(data_chd_re['Diattenuation'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  rot=True, name= filename +'_DDiat' if save is True else None)
# plot_parameter(data_chd_re['Polarizance'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$\mathcal{P}$',  rot=True, name= filename +'_DPol' if save is True else None)
#plot_parameter(data_chd_re['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', rot=True, name= filename +'_DPdelta' if save is True else None)
#plot_parameter(data_chd_re['Ps'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_S$',  rot=True, name= filename +'_DPs' if save is True else None)
#plot_parameter(data_chd_re['Coefficients'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=True, name= filename +'_Coefs' if save is True else None)
plot_parameter(data_gen_re['Retardance'], title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, rot=True, name= filename +'_Ret' if save is True else None)

#%%%% Plots General sigma_theta vs phi --> simulation_retarders

save = False
    
plot_parameter(data_gen_re['Retardance'], title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, rot=True, name= filename +'_Ret' if save is True else None)
# plot_parameter(data_gen_re['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Diat' if save is True else None)
# plot_parameter(data_gen_re['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Pol' if save is True else None)
plot_parameter(data_gen_re['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
plot_parameter(data_gen_re['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Ps' if save is True else None)


#%%%% Save data

# np.save("Data/"+filename, data_chd_re['IPPS'])
# np.save("Data/Re_Phi", x)
# np.save("Data/Re_sigmaTheta", y)

#%% Sigma_phi vs theta --> simulation_retarders_sigma_phi_vs_theta

save = False

xtitle2=r'$\theta$'
ytitle2 = r'$\sigma_\phi$'

MMR,x_thet,y_sigPhi=simulation_retarders_sigma_phi_vs_theta(0)

height=len(y_sigPhi)
width=len(x_thet)
MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
data_chd_re = parameters_ChDecomp(MMR, len(y_sigPhi), len(x_thet), ['diat', 'ret', 'pol', 'coeff', 'ps', 'ipps'])
data_gen_re = parameters_MM(MMR, len(y_sigPhi), len(x_thet), ['diat', 'ret', 'pdelta', 'ps','pol'])


plot_IPPS(data_chd_re['IPPS'],  x=x_thet, y=y_sigPhi, num=3, xtitle=xtitle2, ytitle=ytitle2, rot=True, name = filename +'_IPPS2' if save is True else None)
plot_parameter(data_chd_re['Retardance'],  x=x_thet, y=y_sigPhi,  xtitle=xtitle2, ytitle=ytitle2, subtitle=r'$\Delta$', lims=[0,180], rot=True, name= filename +'_DRet' if save is True else None)
plot_parameter(data_gen_re['Retardance'], title=r'Retardance ($\Delta$)', x=x_thet, y=y_sigPhi, xtitle=xtitle2, ytitle=ytitle2, lims=[0,180], num=1, rot=True, name= filename +'_Ret' if save is True else None)
plot_parameter(data_gen_re['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x_thet, y=y_sigPhi, xtitle=xtitle2, ytitle=ytitle2, num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
plot_parameter(data_gen_re['Ps'], title=r'Spherical Purity ($P_S$)',  x=x_thet, y=y_sigPhi, xtitle=xtitle2, ytitle=ytitle2, num=1, rot=True, name= filename +'_Ps' if save is True else None)
plot_parameter(data_chd_re['Coefficients'], x=x_thet, y=y_sigPhi, xtitle=xtitle2, ytitle=ytitle2, subtitle=r'$c$', num=4, rot=True, name= filename +'_Coefs' if save is True else None)

#%% Sigma_theta vs sigma_phi --> simulation_retarders3

save = False

filename="R_sigma_p0"
xtitle=r'$\sigma_\phi$'
ytitle = r'$\sigma_\theta$'

MMR, x_sigma, y_sigma = simulation_retarders3(0) 

height=len(y_sigma)
width=len(x_sigma)
MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
data_chd_re = parameters_ChDecomp(MMR, len(y_sigma), len(x_sigma), ['diat', 'ret', 'pol', 'coeff', 'ps', 'ipps', 'pdelta'])
data_gen_re = parameters_MM(MMR, len(y_sigma), len(x_sigma), ['diat', 'ret', 'pdelta', 'ps','pol'])

plot_parameter(data_chd_re['Retardance'],  
               x=x_sigma, # Utilitza sigmaPhi
               y=y_sigma, # Utilitza sigmaTheta
               xtitle=xtitle, 
               ytitle=ytitle, 
               subtitle=r'$\Delta$', 
               lims=[0,180],
               num=3,       
               rot=True, 
               name= filename +'_DRet' if save is True else None)
#Coeficients de la Descomposició Característica ($c_0, c_1, c_2, c_3$)
plot_parameter(data_chd_re['Coefficients'],  
               x=x_sigma, # Utilitza sigmaPhi
               y=y_sigma, 
               xtitle=xtitle, ytitle=ytitle, 
               subtitle=subtitles['Coefficients'], 
               num=4, 
               rot=True, 
               name= filename +'_Coefs' if save is True else None)

# Retardància Total $R$ (General de la Matriu de Mueller)
plot_parameter(data_gen_re['Retardance'], 
               title=r'Retardance ($\Delta$)', 
               x=x_sigma, # Utilitza sigmaPhi
               y=y_sigma, 
               xtitle=xtitle, ytitle=ytitle, 
               lims=[0,180], 
               num=1, 
               rot=True, 
               name= filename +'_Ret_Total' if save is True else None)

plot_IPPS(data_chd_re['IPPS'],  
          x=x_sigma, # Utilitza sigmaPhi
          y=y_sigma,
          num=3, 
          xtitle=xtitle, ytitle=ytitle, 
          rot=True, 
          name = filename +'_IPPS' if save is True else None)

plot_parameter(data_gen_re['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x_sigma, y=y_sigma, xtitle=xtitle, ytitle=ytitle, num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
plot_parameter(data_gen_re['Ps'], title=r'Spherical Purity ($P_S$)', x=x_sigma, y=y_sigma, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Ps' if save is True else None)


# %% TETRAEDRE
N_Thetas_3D = 20
N_Py_3D = 20

MMD_3D, x_3D, y_3D = simulation_retarders(0.2, N=300, N_Thetas=20, N_Phis=20)

MMD_3D_reshaped = MMD_3D.reshape(N_Thetas_3D * N_Py_3D, 4, 4).transpose(1, 2, 0)

data_chd_di_3D = parameters_ChDecomp(MMD_3D_reshaped, N_Thetas_3D, N_Py_3D, ['ipps'])
ipps_data_3D = data_chd_di_3D['IPPS']

P1_1D = ipps_data_3D[0, :, :].flatten()
P2_1D = ipps_data_3D[1, :, :].flatten()
P3_1D = ipps_data_3D[2, :, :].flatten()


py_2D = np.tile(x_3D, (N_Thetas_3D, 1))
py_1D_color = py_2D.flatten()

plot_IPPtetrahedron(
    P1_1D, P2_1D, P3_1D, 
    py_1D_color, color_label=r'$\mathbf{\Delta}$'
)

