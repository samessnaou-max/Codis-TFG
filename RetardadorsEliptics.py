# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 14:24:05 2025

@author: Usuario
"""
# -*- coding: utf-8 -*-

import numpy as np
import PolarimetricFunctionsElipticas as pf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
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
    

#%% TETRAEDRE

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

    # Ajustamos el tamaño de la figura para acomodar el gráfico
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
        s=40, 
        alpha=0.95
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.1)

    cbar.set_label(color_label, rotation=0, y=-0.05, labelpad=-10, ha='center') 

    plt.show()

#%%% Simulations

def simulation_retarders(chi_value, N=2000, N_points=20):
    """
    Simulación basada en los parámetros: Delta vs sigma_phi (sigma_theta).
    
    Parámetros de la imagen:
    γ = 0; φ = 0°; I = 100; N_0 = 200; 
    Δ ∈ [0, 2π]; σ_φ ∈ [0, π/2];
    
    Parámetros
    ----------
    N : int, Número de interacciones totales. El valor por defecto es 200.
    N_points : int, Número de puntos en cada eje de variación (I). El valor por defecto es 100.

    Retorna
    -------
    MMR: (N_Deltas, N_Thetas, 4, 4) array
        Matrices de Mueller resultantes de la simulación.
    Deltas: (N_points,) array
        Valores de Retardancia (Δ).
    sigmaTheta: (N_points,) array
        Valores de Desviación Angular (σ_φ).
    """    
    
    
    N_B = 0
    N_A = N - N_B 
    
    Maxsigma_theta = np.pi/2 # Rango [0, π/2]
    sigmaTheta = np.linspace(0, Maxsigma_theta, N_points)
    ThetaM = 0 # Valor medio de theta (orientación), φ = 0°
    
    MaxDelta = 2*np.pi # Rango [0, 2π]
    Deltas = np.linspace(0, MaxDelta, N_points)
    sigmaDelta = 0 
    
    ChiM = chi_value 
    sigmaChi = 0 
    
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
    MMR = np.zeros((N_points, N_points, 4, 4))  
    
    for nph in range(N_points):  
        delta_m = Deltas[nph]
        
        delta = delta_m + sigmaDelta * np.random.randn(N_A) 
        
        chi = ChiM + sigmaChi * np.random.randn(N_A)
        
        M1 = pf.retarder_MM(delta, chi)  
    
        for nt in range(N_points):  
            
            sigma_t = sigmaTheta[nt]
            
            theta = ThetaM + sigma_t * np.random.randn(N_A)  
        
            RN = pf.rotator_MM(-theta)  
            RP = pf.rotator_MM(theta)  
            
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0) 
            
            # --------------------------- MATRIZ TOTAL ---------------------------
            A_T = np.sum(A, axis=0)  
            B_T = N_B * M_dis  
            M_total = A_T + B_T  
    
            MMR[nph, nt, :, :] = M_total
        
    return MMR, Deltas*180/np.pi, sigmaTheta*180/np.pi


def simulation_retarders2(Max_Theta_Avg, N=2000, N_points=200):
    """
    Simulación: Retardancia (Δ, Eje X) vs Elipticidad Media (χM, Eje Y).
    
    Induce despolarización mediante el Promedio Orientacional (suma incoherente sobre θ ∈ [0, π]).

    Parámetros
    ----------
    N : int, Número de interacciones totales (N_A).
    N_points : int, Número de puntos en cada eje de variación.
    Max_Theta_Avg: float, Límite superior del promedio angular (pi por defecto).

    Retorna
    -------
    MMR: (N_Deltas, N_Chis, 4, 4) array
    Deltas: (N_points,) array
    ChiM: (N_points,) array
    """

    N_B = 0
    N_A = N - N_B 

    MaxDelta = 2 * np.pi
    Deltas = np.linspace(0, MaxDelta, N_points)
    sigmaDelta = 0

    Max_Chi = np.pi / 4
    ChiM = np.linspace(-Max_Chi, Max_Chi, N_points)
    sigmaChi = 0

    theta = np.linspace(0, Max_Theta_Avg, N_A, endpoint=False)
    

    RN = pf.rotator_MM(-theta)
    RP = pf.rotator_MM(theta)

    M_dis = np.zeros((4, 4))
    M_dis[0, 0] = 1
    MMR = np.zeros((N_points, N_points, 4, 4))

    for nph in range(N_points):  
        delta_m = Deltas[nph]
        delta = delta_m + sigmaDelta * np.random.randn(N_A)

        for nch in range(N_points):  
            
            chi_m = ChiM[nch]
            chi = chi_m + sigmaChi * np.random.randn(N_A) 

            M1 = pf.retarder_MM(delta, chi)

            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)

            M_total = np.sum(A, axis=0) + N_B * M_dis

            MMR[nph, nch, :, :] = M_total
        
    return MMR, Deltas * 180 / np.pi, ChiM * 180 / np.pi



def simulation_retarders_Delta_vs_SigmaChi_FixedSigmaTheta(N=2000, N_points=200, Max_SigmaChi=np.pi/8, Fixed_SigmaTheta=np.pi/4):
    """
    Simulación: Retardancia (Δ, Eje X) vs Desviación Estándar de la Elipticidad (σ_chi, Eje Y).

    La despolarización es inducida por σ_chi variable Y una dispersión angular FIJA (σ_theta = Fixed_SigmaTheta).

    Parámetros
    ----------
    N : int, Número de interacciones totales (N_A).
    N_points : int, Número de puntos en cada eje de variación.
    Max_SigmaChi: float, Máxima desviación de la elipticidad (π/4 por defecto).
    Fixed_SigmaTheta: float, Valor constante para la dispersión angular (π/4 por defecto).

    Retorna
    -------
    MMR: (N_Deltas, N_SigmaChi, 4, 4) array
    Deltas: (N_points,) array
    SigmaChi: (N_points,) array
    """

    # --------------------------- Configuración de Ejes ---------------------------
    N_B = 0
    N_A = N - N_B  

    # Eje X: Retardancia (Δ)
    MaxDelta = 2 * np.pi
    Deltas = np.linspace(0, MaxDelta, N_points)
    sigmaDelta = 0

    # Eje Y: Desviación Estándar de la Elipticidad (σ_chi)
    SigmaChi = np.linspace(0, Max_SigmaChi, N_points)
    
    ChiM = 0 
    
    ThetaM = 0
    sigmaTheta_fixed = Fixed_SigmaTheta 
    
    M_dis = np.zeros((4, 4))
    M_dis[0, 0] = 1
    MMR = np.zeros((N_points, N_points, 4, 4))

    
    theta = ThetaM + sigmaTheta_fixed * np.random.randn(N_A) 
    RN = pf.rotator_MM(-theta)
    RP = pf.rotator_MM(theta)

    for nph in range(N_points):  
        delta_m = Deltas[nph]
        
        delta = delta_m + sigmaDelta * np.random.randn(N_A)

        for nsc in range(N_points):  
            sigma_chi_current = SigmaChi[nsc] 
            
            chi = ChiM + sigma_chi_current * np.random.randn(N_A) 

            M1 = pf.retarder_MM(delta, chi)

            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)
            
            M_total = np.sum(A, axis=0) + N_B * M_dis

            MMR[nph, nsc, :, :] = M_total
        
    return MMR, Deltas * 180 / np.pi, SigmaChi * 180 / np.pi



def simulation_retarders_Delta_vs_SigmaTheta_FixedSigmaChi(N=2000, N_points=200, Max_SigmaTheta=np.pi/4, Fixed_SigmaChi=np.pi/8):
    """
    Simulación: Retardancia (Δ, Eje X) vs Desviación Estándar de la Orientación (σ_theta, Eje Y).

    La despolarización es inducida por σ_theta variable Y una dispersión de elipticidad FIJA (σ_chi = Fixed_SigmaChi).

    Parámetros
    ----------
    N : int, Número de interacciones totales (N_A).
    N_points : int, Número de puntos en cada eje de variación.
    Max_SigmaTheta: float, Máxima desviación angular (π/2 por defecto).
    Fixed_SigmaChi: float, Valor constante para la dispersión de la elipticidad (π/4 por defecto).

    Retorna
    -------
    MMR: (N_Deltas, N_SigmaTheta, 4, 4) array
    Deltas: (N_points,) array
    SigmaTheta: (N_points,) array
    """

    N_B = 0
    N_A = N - N_B  

    MaxDelta = 2 * np.pi
    Deltas = np.linspace(0, MaxDelta, N_points)
    sigmaDelta = 0

    SigmaTheta = np.linspace(0, Max_SigmaTheta, N_points)
    
    ChiM = 0 
    
    sigmaChi_fixed = Fixed_SigmaChi 
    
    ThetaM = 0
    
    M_dis = np.zeros((4, 4))
    M_dis[0, 0] = 1
    MMR = np.zeros((N_points, N_points, 4, 4))

    
    chi = ChiM + sigmaChi_fixed * np.random.randn(N_A) 

    for nph in range(N_points):  # Bucle 1: Retardancia (Eje X)
        delta_m = Deltas[nph]
        
        delta = delta_m + sigmaDelta * np.random.randn(N_A)

        M1 = pf.retarder_MM(delta, chi)

        for nst in range(N_points):  
            sigma_theta_current = SigmaTheta[nst] 
            
            theta = ThetaM + sigma_theta_current * np.random.randn(N_A) 

            RN = pf.rotator_MM(-theta)
            RP = pf.rotator_MM(theta)
            
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)
            
            # --------------------------- MATRIZ TOTAL ---------------------------
            M_total = np.sum(A, axis=0) + N_B * M_dis

            MMR[nph, nst, :, :] = M_total
        
    return MMR, Deltas * 180 / np.pi, SigmaTheta * 180 / np.pi

def simulation_sigmas_vs_sigmas(Fixed_Delta, N=2000, N_points=200, Max_Sigma_chi=np.pi/8, Max_Sigma_theta=np.pi/4):
    """
    Simulación: Desviación de la Elipticidad (σ_chi, Eje X) vs 
                Desviación de la Orientación (σ_theta, Eje Y).
    
    La Retardancia (Δ) se mantiene fija.

    Parámetros
    ----------
    Fixed_Delta: float, Valor constante de la Retardancia (Δ) para todo el mapa (ej: π/2 o π).
    N : int, Número de interacciones totales (N_A).
    N_points : int, Número de puntos en cada eje de variación.
    Max_Sigma: float, Máximo valor para ambas dispersiones (σ_chi_max y σ_theta_max).

    Retorna
    -------
    MMR: (N_SigmaChi, N_SigmaTheta, 4, 4) array
    SigmaChi: (N_points,) array
    SigmaTheta: (N_points,) array
    """

    N_B = 0
    N_A = N - N_B  

    SigmaChi = np.linspace(0, Max_Sigma_chi, N_points)
    
    SigmaTheta = np.linspace(0, Max_Sigma_theta, N_points)
    
    # Parámetros Fijos
    DeltaM = Fixed_Delta 
    ChiM = 0              
    ThetaM = 0            
    sigmaDelta = 0

    M_dis = np.zeros((4, 4))
    M_dis[0, 0] = 1
    MMR = np.zeros((N_points, N_points, 4, 4))

    for nph in range(N_points):  
        sigma_chi_current = SigmaChi[nph] 
        
        chi = ChiM + sigma_chi_current * np.random.randn(N_A) 

        delta = DeltaM + sigmaDelta * np.random.randn(N_A)

        M1 = pf.retarder_MM(delta, chi)


        for nst in range(N_points):  
            sigma_theta_current = SigmaTheta[nst] 
            
            theta = ThetaM + sigma_theta_current * np.random.randn(N_A) 

            RN = pf.rotator_MM(-theta)
            RP = pf.rotator_MM(theta)
            
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)
            
            M_total = np.sum(A, axis=0) + N_B * M_dis

            MMR[nph, nst, :, :] = M_total 
        
    return MMR, SigmaChi * 180 / np.pi, SigmaTheta * 180 / np.pi


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
            

# %% Retarders: Simulación Delta vs sigma_phi

filename="R_Delta_vs_sigmaPhi"
xtitle=r'$\Delta$' # Eje x: Retardancia (0° a 360°)
ytitle = r'$\sigma_\theta$' # Eje y: Desviación Angular (0° a 90°)

# Ejecutar la simulación.
chi_nv = [0, np.pi/16, np.pi/8, np.pi/6, np.pi/4]

for chi_v in chi_nv:

    MMR, x, y = simulation_retarders(chi_value=chi_v, N=2000, N_points=200) 
    
    height=len(y) # N_points
    width=len(x)  # N_points
    

    
    MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
    

    data_chd_re = parameters_ChDecomp(MMR, height, width, ['diat', 'ret', 'pol', 'coeff','ipps'])
    data_gen_re = parameters_MM(MMR, height, width, ['ret', 'pdelta'])
    

    save = False
    
    # %%%% Plots Ch Decomp (Ejes: X=Delta, Y=sigma_phi)
    
    # IPPS (P1 y P2 vs P3)
    plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=3, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)
    plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=2, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)    
    # Retardancia Despolarizada (Debe ser cero o cercana, ya que γ=0 y no hay dispersión en Delta)
    #plot_parameter(data_chd_re['Retardance'], title='Retardancia Descompuesta', x=x, y=y,  xtitle=xtitle, ytitle=ytitle, subtitle=r'$\Delta$', lims=[0,180], rot=False, color='jet', origin='lower', name= filename +'_DRet' if save is True else None)
    # Purity Spherical (Ps)
    #plot_parameter(data_chd_re['Ps'], title='Ps Descompuesta', x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_S$',  rot=True, origin='upper', name= filename +'_DPs' if save is True else None)
    
    plot_parameter(data_chd_re['Coefficients'], x=x, y=y,  xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=False, color='jet', origin='lower', name= filename +'_Coefs' if save is True else None)
    
    # %%% Plots General (Ejes: X=Delta, Y=sigma_phi)
    
    # Retardancia Total (Debe ser igual a Delta)
    #plot_parameter(data_gen_re['Retardance'], title=r'Retardancia ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,360], num=1, rot=False, color='jet', origin='lower', name= filename +'_Ret' if save is True else None)
    # Polarimetric Purity (Pdelta)
    #plot_parameter(data_gen_re['Pdelta'], title=r'Pureza Polarimétrica ($P_\Delta$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=False, color='jet', origin='lower', name= filename +'_Pdelta' if save is True else None)
    # Spherical Purity (Ps)
    #plot_parameter(data_gen_re['Ps'], title=r'Pureza Esférica ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=False, color='jet', origin='lower', name= filename +'_Ps' if save is True else None)
    
    #plot_parameter(data_chd_re['Coefficients'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=subtitles['Coefficients'], num=4, rot=False, color='jet', origin='lower', name= filename +'_Coefs' if save is True else None)
    
    
    # %% TETRAEDRE
    # Redefinimos los parámetros de la simulación para generar solo 20 puntos (4x5)
    N_Thetas_3D = 20
    N_Py_3D = 20
    
    MMD_3D, x_3D, y_3D = simulation_retarders(chi_value=chi_v, N=20, N_points=20)
    
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


# %% Retarders: Simulación Delta vs Chi

filename="R_Delta_vs_sigmaPhi"
xtitle=r'$\Delta$' 
ytitle = r'$\chi$' 

theta_max_n = [np.pi, np.pi/1.5, np.pi/2, np.pi/3, np.pi/4]

for theta_max in theta_max_n:

    MMR, x, y = simulation_retarders2(Max_Theta_Avg=theta_max, N=3000, N_points=200) 
    
    height=len(y) # N_points
    width=len(x)  # N_points
    
    
    MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
    
    data_chd_re = parameters_ChDecomp(MMR, height, width, [ 'ret', 'coeff', 'ps', 'ipps'])
    data_gen_re = parameters_MM(MMR, height, width, ['ret', 'pdelta'])
    

    save = False
    
    # %%%% Plots Ch Decomp (Ejes: X=Delta, Y=sigma_phi)
    
    # IPPS (P1 y P2 vs P3)
    plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=3, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)
    plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=2, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)
    # Retardancia Despolarizada (Debe ser cero o cercana, ya que γ=0 y no hay dispersión en Delta)
    #plot_parameter(data_chd_re['Retardance'], title='Retardancia Descompuesta', x=x, y=y,  xtitle=xtitle, ytitle=ytitle, subtitle=r'$\Delta$', lims=[0,180], rot=False, color='jet', origin='lower', name= filename +'_DRet' if save is True else None)
    # Purity Spherical (Ps)
    #plot_parameter(data_chd_re['Ps'], title='Ps Descompuesta', x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_S$',  rot=True, origin='upper', name= filename +'_DPs' if save is True else None)
    
    plot_parameter(data_chd_re['Coefficients'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=False, color='jet', origin='lower', name= filename +'_Coefs' if save is True else None)
    
    # %%% Plots General (Ejes: X=Delta, Y=sigma_phi)
    
    #plot_parameter(data_gen_re['Retardance'], title=r'Retardancia ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,360], num=1, rot=False, color='jet', origin='lower', name= filename +'_Ret' if save is True else None)
    #plot_parameter(data_gen_re['Pdelta'], title=r'Pureza Polarimétrica ($P_\Delta$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=False, color='jet', origin='lower', name= filename +'_Pdelta' if save is True else None)
    #plot_parameter(data_gen_re['Ps'], title=r'Pureza Esférica ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=False, color='jet', origin='lower', name= filename +'_Ps' if save is True else None)
    #plot_parameter(data_chd_re['Coefficients'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=subtitles['Coefficients'], num=4, rot=False, color='jet', origin='lower', name= filename +'_Coefs' if save is True else None)
    
    
    # %% TETRAEDRE
    
    N_Thetas_3D = 20
    N_Py_3D = 20
    
    MMD_3D, x_3D, y_3D = simulation_retarders2(Max_Theta_Avg=theta_max, N=200, N_points=20)
    
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


#%% DELTA VS SIGMA_CHI PARA SIGMA_THETA Y THETA FIXA

filename="R_Delta_vs_SigmaChi"
xtitle=r'$\Delta$'
ytitle = r'$\sigma_\chi$'
save = False

MMR, x, y = simulation_retarders_Delta_vs_SigmaChi_FixedSigmaTheta(N=3000, N_points=200, Fixed_SigmaTheta=np.pi/4) 

height=len(y) # y = SigmaChi
width=len(x)  # x = Delta

MMR = MMR.reshape(width*height, 4, 4).transpose(1, 2, 0) 
data_chd_re = parameters_ChDecomp(MMR, height, width, ['ipps', 'pdelta', 'coeff'])
data_gen_re = parameters_MM(MMR, height, width, ['pdelta']) # Solo necesitamos Pdelta de la general


print("Generando plots de IPPS (P1, P2, P3)...")
plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=3, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)
plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=2, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)


plot_parameter(data_chd_re['Coefficients'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=False, color='jet', origin='lower', name= filename +'_Coefs' if save is True else None)

#plot_parameter(data_gen_re['Pdelta'], title=r'Pureza Polarimétrica ($P_\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=False, color='jet', origin='lower', name= filename +'_Pdelta' if save is True else None)


# %% TETRAEDRE

N_Thetas_3D = 20
N_Py_3D = 20

MMD_3D, x_3D, y_3D = simulation_retarders_Delta_vs_SigmaChi_FixedSigmaTheta(N=20, N_points=20, Fixed_SigmaTheta=np.pi/4) 

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


#%% DELTA VS SIGMA_THETA PARA SIGMA_CHI Y CHI FIXA

filename="R_Delta_vs_SigmaTheta"
xtitle=r'$\Delta$'
ytitle = r'$\sigma_\theta$'
save = False


MMR, x, y = simulation_retarders_Delta_vs_SigmaTheta_FixedSigmaChi(N=3000, N_points=200, Fixed_SigmaChi=np.pi/8) 

height=len(y) # y = SigmaTheta
width=len(x)  # x = Delta


MMR = MMR.reshape(width*height, 4, 4).transpose(1, 2, 0) 
data_chd_re = parameters_ChDecomp(MMR, height, width, ['ipps', 'pdelta', 'coeff'])
data_gen_re = parameters_MM(MMR, height, width, ['pdelta']) 



print("Generando plots de IPPS (P1, P2, P3)...")
# Se espera que los valores de Pureza sean bajos en general (debido a Sigma_Chi fijo alto)
plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=3, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)
plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=2, xtitle=xtitle, ytitle=ytitle, rot=False, color='jet', origin='lower', name = filename +'_IPPS' if save is True else None)
plot_parameter(data_chd_re['Coefficients'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=False, color='jet', origin='lower', name= filename +'_Coefs' if save is True else None)


# %% TETRAEDRE

N_Thetas_3D = 20
N_Py_3D = 20

MMD_3D, x_3D, y_3D = simulation_retarders_Delta_vs_SigmaTheta_FixedSigmaChi(N=20, N_points=20, Fixed_SigmaChi=np.pi/8) 

# 1. Preparar los datos aplanados de los IPPs
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


#%% SIGMA_THETA vs SIGMA_CHI para diferentes valores de DELTA

deltas = [np.pi, np.pi/2, np.pi/4, 3*np.pi/2, 2*np.pi]

#filename=f"R_SigmaVSigma_Delt"
xtitle=r'$\sigma_\chi$'  # Eje X: Sigma_Chi
ytitle = r'$\sigma_\theta$ ' # Eje Y: Sigma_Theta
save = False

for FIXED_DELTA_RAD in deltas:
    MMR, x, y = simulation_sigmas_vs_sigmas(Fixed_Delta=FIXED_DELTA_RAD, N=3000, N_points=200, Max_Sigma_chi=np.pi/8, Max_Sigma_theta=np.pi/4) 
    
    height=len(y) # y = SigmaTheta
    width=len(x)  # x = SigmaChi
    
    MMR = MMR.reshape(width*height, 4, 4).transpose(1, 2, 0) 
    data_chd_re = parameters_ChDecomp(MMR, height, width, ['ipps', 'pdelta', 'coeff'])
    data_gen_re = parameters_MM(MMR, height, width, ['pdelta']) 
    
    
    plot_IPPS(data_chd_re['IPPS'],  
              x=x, y=y, 
              num=3, 
              #title=f"IPPs ($\Delta = {int(FIXED_DELTA_RAD*180/np.pi)}^\circ$)",
              xtitle=xtitle, ytitle=ytitle, 
              rot=True, color='jet',
              origin='lower')
    
    plot_parameter(data_chd_re['Coefficients'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=False, color='jet', origin='lower')
    
    # Pureza Polarimétrica (Pdelta)
    #plot_parameter(data_gen_re['Pdelta'], title=r'Pureza Polarimétrica ($P_\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=False, color='jet', origin='lower', name= filename +'_Pdelta' if save is True else None)
    
    # %% TETRAEDRE
    
    N_Thetas_3D = 20
    N_Py_3D = 20
    
    MMD_3D, x_3D, y_3D = simulation_sigmas_vs_sigmas(Fixed_Delta=FIXED_DELTA_RAD, N=200, N_points=20, Max_Sigma_chi=np.pi/8, Max_Sigma_theta=np.pi/4)
    
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
        py_1D_color, color_label=r'$\mathbf{\sigma_\chi}$'
    )

