import numpy as np
import sys
from decimal import *
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def clean_matrix(M):
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-10: M[i][j] = 0
            else: M[i][j] = float('{0:.6f}'.format(M[i][j]))
    return M

def complex_givens_for(a, b, swap=False):
    if np.abs(b) == 0:
        print("here")
        return np.eye(2, dtype=float)
    r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)

    u1 = a / r
    u2 = b / r
    if swap:
        G2 = np.array([[u2,         -u1],
                       [np.conj(u1), np.conj(u2)]], dtype=float)
    else:
        G2 = np.array([[np.conj(u1), np.conj(u2)],
                    [u2,         -u1]], dtype=float)
    return G2

def two_level_decomposition_left(U):
    n = U.shape[0]
    A = U.copy().astype(float)
    G_list = []
    for j in range(n):
        for i in range(j + 1, n):
            a = A[j, j]
            b = A[i, j]
            G2 = complex_givens_for(a, b)
            G = np.eye(n, dtype=float)
            G[np.ix_([j, i], [j, i])] = G2
            
            A = G @ A
            G_list.append(G)
    return G_list, A

def create_C(U):
    n = U.shape[0]
    A = U.copy().astype(float)
    
    G_list = []
    for j in range(1):
        for i in range(j + 1, 2):
            a = A[j, j]
            b = A[i, j]
            G2 = complex_givens_for(a, b, True)
            G = np.eye(n, dtype=float)
            print(G2)
            print("\n")
            G[np.ix_([j, i], [j, i])] = G2
            
            A = G @ A
            # print(A)
            # print("\n")
            G_list.append(G)
    return G_list, A

def create_A(U):
    n = U.shape[0]
    A = U.copy().astype(float)
    G_list = []
    for j in range(n):
        for i in range(j + 1, n):
            a = A[j, j]
            b = A[i, j]
            if j == 1 and i==2:
                G2 = complex_givens_for(a, b, True)
            else:
                G2 = complex_givens_for(a, b, False)
            G = np.eye(n, dtype=float)
            G[np.ix_([j, i], [j, i])] = G2
            

            A = G @ A
            print(clean_matrix(A))
            print("\n")
            G_list.append(G)
                
    return G_list, A

U= np.array([[ 0.62923587, -0.27810894,  0.06225542 , 0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
             [-0.12832786, -0.72115181 , 0.06330487 ,-0.25546329  ,0.1811286,  -0.15270451, 0.58015675, -0.03866454],
             [ 0.14146621, 0.33503048 , 0.63506461,  0.14609694, -0.17964295 , 0.2131513, 0.56863418,  0.20503807],
             [-0.16206701 , 0.04719766 , 0.46284018 ,-0.12025008  ,0.7869089,   0.20527736, -0.27689368 ,-0.06921729],
             [ 0.25624259 , 0.38011453 ,-0.34262361, -0.67547194,  0.23405039 , 0.00414033, 0.21956521 , 0.33644285],
             [ 0.23268996 ,-0.28038369, -0.23106108  ,0.05510674 ,-0.02434206 , 0.89985678, -0.00166854 , 0.02183679],
             [ 0.0135536  , 0.25475662 ,-0.4235493,   0.54214077 , 0.46218213, -0.0468658, 0.43173278, -0.24372693],
             [-0.64909721 ,-0.01092904, -0.15625139 , 0.19494701 , 0.02944304 , 0.09593373, 0.0091941  , 0.71132259]])

I = np.eye(2)
SWAP = np.array([[1, 0, 0 ,0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

SWAP_I = np.kron(SWAP, I)
I_SWAP = np.kron(I, SWAP)

U_1 = U[:4,:4]

# print(U_1)
G_list, A= two_level_decomposition_left(U_1)
# G_list, A= create_A(U_1)
G_0 = np.eye(len(G_list[0]))

for G in G_list:
    G_0 = G @ G_0


A_U = np.kron(I, G_0)

U = clean_matrix(A_U @ U)
# print(U)
# print("\n")
# exit()

U = SWAP_I @ I_SWAP @ U

# print(U)
# print("\n")

# exit()
U_2 = U[4:, :4]
# print(U_2)
G_list, A= two_level_decomposition_left(U_2)

G_0 = np.eye(len(G_list[0]))

for G in G_list:
    G_0 = G @ G_0

B_U =  np.kron(I, G_0)
# B_U = I_SWAP @ SWAP_I @ np.kron(I, G_0)

U = clean_matrix(B_U @ U)



print(U)
print("\n")
# exit()

# I_SWAP @ SWAP_I @ BB @ SWAP_I @ I_SWAP

###################################

U_2 = U[4:,:4]

# print(U_1)
G_list, C= create_C(U_2)
G_0 = np.eye(len(G_list[0]))
print(G_0)
print("\n")
for G in G_list:
    G_0 = G @ G_0


C_U = np.kron(I, G_0)

U = clean_matrix(C_U @ U)
print(U)