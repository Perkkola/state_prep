import sys
from sympy import *
from sympy.physics.quantum import TensorProduct
import numpy as np
from scipy.linalg import cossin
from functools import reduce
from scipy.optimize import fsolve
from scipy import optimize
import math

I = np.eye(2)
I_4 = np.eye(4)
I_8 = np.eye(8)


def general_U4(x):
    diag = np.diag([np.exp(x[0] * 1j), np.exp(x[1] * 1j), np.exp(x[2] * 1j), np.exp(x[3] * 1j)])
    G_1_2 = np.array([[np.cos(x[4]), -np.exp(x[5] * 1j) * np.sin(x[4]), 0, 0],
                      [np.exp(-x[5] * 1j) * np.sin(x[4]), np.cos(x[4]), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    
    G_1_3 = np.array([[np.cos(x[6]), 0, -np.exp(x[7] * 1j) * np.sin(x[6]), 0],
                      [0, 1, 0, 0],
                      [np.exp(-x[7] * 1j) * np.sin(x[6]), 0, np.cos(x[6]), 0],
                      [0, 0, 0, 1]])
    
    G_1_4 = np.array([[np.cos(x[8]), 0, 0, -np.exp(x[9] * 1j) * np.sin(x[8])],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [np.exp(-x[9] * 1j) * np.sin(x[8]), 0, 0, np.cos(x[8])]])
    
    G_2_3 = np.array([[1, 0, 0, 0],
                      [0, np.cos(x[10]), -np.exp(x[11] * 1j) * np.sin(x[10]), 0],
                      [0, np.exp(-x[11] * 1j) * np.sin(x[10]), np.cos(x[10]), 0],
                      [0, 0, 0, 1]])
    
    G_2_4 = np.array([[1, 0, 0, 0],
                      [0, np.cos(x[12]), 0, -np.exp(x[13] * 1j) * np.sin(x[12])],
                      [0, 0, 1, 0],
                      [0, np.exp(-x[13] * 1j) * np.sin(x[12]), 0, np.cos(x[12])]])
    
    G_3_4 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, np.cos(x[14]), -np.exp(x[15] * 1j) * np.sin(x[14])],
                      [0, 0, np.exp(-x[15] * 1j) * np.sin(x[14]), np.cos(x[14])]])
    return diag @ G_3_4 @ G_2_4 @ G_2_3 @ G_1_4 @ G_1_3 @ G_1_2



U_O = np.array([[ 0.62923587, -0.27810894,  0.06225542,  0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181,  0.06330487, -0.25546329,  0.1811286 , -0.15270451,  0.58015675, -0.03866454],
 [ 0.14146621,  0.33503048,  0.63506461,  0.14609694, -0.17964295,  0.2131513 ,  0.56863418,  0.20503807],
 [-0.16206701,  0.04719766,  0.46284018, -0.12025008,  0.7869089 ,  0.20527736, -0.27689368, -0.06921729],
 [ 0.25624259,  0.38011453, -0.34262361, -0.67547194,  0.23405039,  0.00414033,  0.21956521,  0.33644285],
 [ 0.23268996, -0.28038369, -0.23106108,  0.05510674, -0.02434206,  0.89985678, -0.00166854,  0.02183679],
 [ 0.0135536 ,  0.25475662, -0.4235493 ,  0.54214077,  0.46218213, -0.0468658 ,  0.43173278, -0.24372693],
 [-0.64909721, -0.01092904, -0.15625139,  0.19494701,  0.02944304,  0.09593373,  0.0091941 ,  0.71132259]])

def func(x):
    A = np.array(general_U4(x[:16])).reshape(4, 4)
    B = np.array(general_U4(x[16:32])).reshape(4, 4)
    C = np.array(general_U4(x[32:48])).reshape(4, 4)
    D = np.array(general_U4(x[48:64])).reshape(4, 4)
    E = np.array(general_U4(x[64:80])).reshape(4, 4)
    F = np.array(general_U4(x[80:96])).reshape(4, 4)

    U_A = np.kron(I, A)
    U_B = np.kron(B, I)
    U_C = np.kron(I, C)
    U_D = np.kron(D, I)
    U_E = np.kron(I, E)
    U_F = np.kron(F, I)

    eqs = U_C @ U_B @ U_A @ U_O - (U_D.conjugate().T) @ (U_E.conjugate().T) @ (U_F.conjugate().T)
    eqs = np.absolute(eqs.reshape(64,))
    return np.append(eqs, np.zeros(32))

def func_ABCD(x):
    global E_angles, F_angles
    A = np.array(general_U4(x[:16])).reshape(4, 4)
    B = np.array(general_U4(x[16:32])).reshape(4, 4)
    C = np.array(general_U4(x[32:48])).reshape(4, 4)
    D = np.array(general_U4(x[48:64])).reshape(4, 4)

    E = np.array(general_U4(E_angles)).reshape(4, 4)
    F = np.array(general_U4(F_angles)).reshape(4, 4)

    U_A = np.kron(I, A)
    U_B = np.kron(B, I)
    U_C = np.kron(I, C)
    U_D = np.kron(D, I)
    U_E = np.kron(I, E)
    U_F = np.kron(F, I)

    eqs = U_C @ U_B @ U_A @ U_O - (U_D.conjugate().T) @ (U_E.conjugate().T) @ (U_F.conjugate().T)
    eqs = np.absolute(eqs.reshape(64,))

    return eqs

def func_CDEF(x):
    global A_angles, B_angles
    A = np.array(general_U4(A_angles)).reshape(4, 4)
    B = np.array(general_U4(B_angles)).reshape(4, 4)
    C = np.array(general_U4(x[:16])).reshape(4, 4)
    D = np.array(general_U4(x[16:32])).reshape(4, 4)

    E = np.array(general_U4(x[32:48])).reshape(4, 4)
    F = np.array(general_U4(x[48:64])).reshape(4, 4)

    U_A = np.kron(I, A)
    U_B = np.kron(B, I)
    U_C = np.kron(I, C)
    U_D = np.kron(D, I)
    U_E = np.kron(I, E)
    U_F = np.kron(F, I)

    eqs = U_C @ U_B @ U_A @ U_O - (U_D.conjugate().T) @ (U_E.conjugate().T) @ (U_F.conjugate().T)
    eqs = np.absolute(eqs.reshape(64,))

    return eqs

def init():
    init_angles = [0.1 for x in np.random.random_sample(16)]
    A_angles = init_angles.copy()
    B_angles = init_angles.copy()
    C_angles = init_angles.copy()
    D_angles = init_angles.copy()
    E_angles = init_angles.copy()
    F_angles = init_angles.copy()

    return (A_angles, B_angles, C_angles, D_angles, E_angles, F_angles)

A_angles, B_angles, C_angles, D_angles, E_angles, F_angles = init()

# for _ in range(2):
#     ABCD = optimize.broyden1(func_ABCD, np.array([A_angles, B_angles, C_angles, D_angles]).flatten())
#     A_angles = ABCD[:16]
#     B_angles = ABCD[16:32]
#     C_angles = ABCD[32:48]
#     D_angles = ABCD[48:64]

#     CDEF = optimize.broyden1(func_CDEF, np.array([C_angles, D_angles, E_angles, F_angles]).flatten())

#     C_angles = CDEF[:16]
#     D_angles = CDEF[16:32]
#     E_angles = CDEF[32:48]
#     F_angles = CDEF[48:64]

# roots = fsolve(func, np.array([A_angles, B_angles, C_angles, D_angles, E_angles, F_angles]).flatten())
roots = optimize.broyden1(func, np.array([A_angles, B_angles, C_angles, D_angles, E_angles, F_angles]).flatten())

# print(f"ABCD: {ABCD}\n")
# print(f"CDEF: {CDEF}\n")
print(f"Roots: {roots}\n")
print(f"Func values: {func(np.array([A_angles, B_angles, C_angles, D_angles, E_angles, F_angles]).flatten())}")
# print(f"U_O reconstructed:")
exit()