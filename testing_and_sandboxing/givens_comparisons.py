import sys
from sympy import *
from sympy.physics.quantum import TensorProduct
import numpy as np
from scipy.linalg import cossin
from functools import reduce
from scipy.optimize import fsolve
from scipy import optimize
import math

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

U_O = np.array([[ 0.62923587, -0.27810894,  0.06225542,  0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181,  0.06330487, -0.25546329,  0.1811286 , -0.15270451,  0.58015675, -0.03866454],
 [ 0.14146621,  0.33503048,  0.63506461,  0.14609694, -0.17964295,  0.2131513 ,  0.56863418,  0.20503807],
 [-0.16206701,  0.04719766,  0.46284018, -0.12025008,  0.7869089 ,  0.20527736, -0.27689368, -0.06921729],
 [ 0.25624259,  0.38011453, -0.34262361, -0.67547194,  0.23405039,  0.00414033,  0.21956521,  0.33644285],
 [ 0.23268996, -0.28038369, -0.23106108,  0.05510674, -0.02434206,  0.89985678, -0.00166854,  0.02183679],
 [ 0.0135536 ,  0.25475662, -0.4235493 ,  0.54214077,  0.46218213, -0.0468658 ,  0.43173278, -0.24372693],
 [-0.64909721, -0.01092904, -0.15625139,  0.19494701,  0.02944304,  0.09593373,  0.0091941 ,  0.71132259]])

V_t = [[-0.54409413, -0.10171701, -0.25889078,  0.79157488],
 [-0.39583923,  0.59434125, -0.58362529, -0.38658932],
 [ 0.7315188,   0.35371355, -0.41023319,  0.41409624],
 [-0.11024121,  0.71505164,  0.65120336,  0.22908997]]

def decompose_unitary(U):
    n = len(U)
    givens_array = []
    givens_angles = []
    for i in range(n):
        for j in range(i+1, n):
            G = np.eye(n)
            G[i][i] = np.conjugate(U[i][i] / (np.sqrt(np.square(np.abs(U[i][i])) + np.square(np.abs(U[j][i])))))
            G[i][j] = np.conjugate(U[j][i] / (np.sqrt(np.square(np.abs(U[i][i])) + np.square(np.abs(U[j][i])))))
            G[j][i] = -U[j][i] / (np.sqrt(np.square(np.abs(U[i][i])) + np.square(np.abs(U[j][i]))))
            G[j][j] = U[i][i] / (np.sqrt(np.square(np.abs(U[i][i])) + np.square(np.abs(U[j][i]))))
            U = G @ U
            givens_array.append(G)

            u_1 = G[j][j]
            u_2 = -G[j][i]

            theta = np.arctan2(np.abs(u_2), np.abs(u_1))
            psi = np.angle(u_2) - np.angle(u_1)
            psi = (psi + np.pi) % (2*np.pi) - np.pi

            givens_angles.append(theta)
            givens_angles.append(psi)

    R_T = U
    print(R_T)
    # for G in reversed(givens_array):
    #     R_T = np.conjugate(G.T) @ R_T

    diag = [R_T[x][x] if np.abs(R_T[x][x]) <= 1 else np.sign(R_T[x][x]) for x in range(n)]
    diag = np.round(diag, 6)
    diag_angles = [np.acos(x) for x in diag]


    diag_angles.extend(givens_angles)
    return diag_angles

def general_U4(x):
    diag = np.diag([np.exp(x[0] * 1j), np.exp(x[1] * 1j), np.exp(x[2] * 1j), np.exp(x[3] * 1j)])
    G_1_2 = np.array([[np.cos(x[4]), np.exp(-x[5] * 1j) * np.sin(x[4]), 0, 0],
                      [-np.exp(x[5] * 1j) * np.sin(x[4]), np.cos(x[4]), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    
    G_1_3 = np.array([[np.cos(x[6]), 0, np.exp(-x[7] * 1j) * np.sin(x[6]), 0],
                      [0, 1, 0, 0],
                      [-np.exp(x[7] * 1j) * np.sin(x[6]), 0, np.cos(x[6]), 0],
                      [0, 0, 0, 1]])
    
    G_1_4 = np.array([[np.cos(x[8]), 0, 0, np.exp(-x[9] * 1j) * np.sin(x[8])],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [-np.exp(x[9] * 1j) * np.sin(x[8]), 0, 0, np.cos(x[8])]])
    
    G_2_3 = np.array([[1, 0, 0, 0],
                      [0, np.cos(x[10]), np.exp(-x[11] * 1j) * np.sin(x[10]), 0],
                      [0, -np.exp(x[11] * 1j) * np.sin(x[10]), np.cos(x[10]), 0],
                      [0, 0, 0, 1]])
    
    G_2_4 = np.array([[1, 0, 0, 0],
                      [0, np.cos(x[12]), 0, np.exp(-x[13] * 1j) * np.sin(x[12])],
                      [0, 0, 1, 0],
                      [0, -np.exp(x[13] * 1j) * np.sin(x[12]), 0, np.cos(x[12])]])
    
    G_3_4 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, np.cos(x[14]), np.exp(-x[15] * 1j) * np.sin(x[14])],
                      [0, 0, -np.exp(x[15] * 1j) * np.sin(x[14]), np.cos(x[14])]])
    return np.conjugate(G_3_4 @ G_2_4 @ G_2_3 @ G_1_4 @ G_1_3 @ G_1_2).T @ diag

print(general_U4(decompose_unitary(V_t)))

# print(general_U4([0, 0, 0, 0, 1.079482, 0, ]))