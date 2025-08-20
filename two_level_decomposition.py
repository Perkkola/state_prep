import numpy as np
import sys
# from decimal import *
from scipy.linalg import cossin

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def clean_matrix(M):
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-10: M[i][j] = float(0.0)
            M[i][j] = float('{0:.8f}'.format(M[i][j]))
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

def single_qubit_givens(U, j, i):
    A = U.copy().astype(float)
    a = A[j, j]
    b = A[i, j]
    G2 = complex_givens_for(a, b)

    return G2

def two_level_decomposition_left(U):
    n = U.shape[0]
    A = U.copy().astype(float)
    G_list = []
    for j in range(n-1):
        for i in range(j + 1, n):
            a = A[j, j]
            b = A[i, j]
            G2 = complex_givens_for(a, b)
            G = np.eye(n, dtype=float)
            G[np.ix_([j, i], [j, i])] = G2
            A = G @ A
            G_list.append(G)

    G_0 = np.eye(n)

    for G in G_list:
        G_0 = G @ G_0

    return G_0


def two_level_decomposition_left_upper(U):
    n = U.shape[0]
    A = U.copy().astype(float)
    G_list = []
    for j in range(n-1, 0, -1):
        for i in range(j - 1, -1, -1):
            G = create_single_G(A, j, i)
            A = G @ A
            G_list.append(G)

    G_0 = np.eye(n)

    for G in G_list:
        G_0 = G @ G_0

    return G_0

def two_level_decomposition_left_inverted(U):
    n = U.shape[0]
    A = U.copy().astype(float)
    X = np.array([[0, 1],
                  [1, 0]])
    G_list = []
    for j in range(n-1, 0, -1):
        for i in range(j-1, -1, -1):
            a = A[n-j-1, n-i-2]
            b = A[n-i-1, n-i-2]
            G2 = complex_givens_for(a, b, True)
            G = np.kron(X, X) @ np.eye(n, dtype=float)
            G[np.ix_([i, j], [n-i-1, n-j-1])] = G2
            print(A)
            print("\n")
            print(a)
            print(b)
            print("\n")
            print(G)
            print("\n")
            A = G @ A
            G_list.append(G)

    G_0 = np.eye(n)

    for G in G_list:
        G_0 = G @ G_0

    return G_0

def create_single_G(U, j, i, swap = False):
    n = U.shape[0]
    A = U.copy().astype(float)
    a = A[j, j]
    b = A[i, j]
    G2 = complex_givens_for(a, b, swap)
    G = np.eye(n, dtype=float)
    G[np.ix_([j, i], [j, i])] = G2

    return G

def create_single_G_inverted(U, j, i):
    n = U.shape[0]
    A = U.copy().astype(float)
    a = A[j, j]
    b = A[i, j]
    G2 = complex_givens_for(a, b)
    G = np.eye(n, dtype=float)
    G[np.ix_([j, i], [j, i])] = G2

    return G

    


U= np.array([[ 0.62923587, -0.27810894,  0.06225542 , 0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
             [-0.12832786, -0.72115181 , 0.06330487 ,-0.25546329  ,0.1811286,  -0.15270451, 0.58015675, -0.03866454],
             [ 0.14146621, 0.33503048 , 0.63506461,  0.14609694, -0.17964295 , 0.2131513, 0.56863418,  0.20503807],
             [-0.16206701 , 0.04719766 , 0.46284018 ,-0.12025008  ,0.7869089,   0.20527736, -0.27689368 ,-0.06921729],
             [ 0.25624259 , 0.38011453 ,-0.34262361, -0.67547194,  0.23405039 , 0.00414033, 0.21956521 , 0.33644285],
             [ 0.23268996 ,-0.28038369, -0.23106108  ,0.05510674 ,-0.02434206 , 0.89985678, -0.00166854 , 0.02183679],
             [ 0.0135536  , 0.25475662 ,-0.4235493,   0.54214077 , 0.46218213, -0.0468658, 0.43173278, -0.24372693],
             [-0.64909721 ,-0.01092904, -0.15625139 , 0.19494701 , 0.02944304 , 0.09593373, 0.0091941  , 0.71132259]])





I = np.eye(2)
X = np.array([[0, 1],
            [1, 0]])

X_I_I = np.kron(X, np.kron(I, I))
I_X_I = np.kron(I, np.kron(X, I))
I_I_X = np.kron(I, np.kron(I, X))
X_X_I = np.kron(X, np.kron(X, I))
I_X_X = np.kron(I, np.kron(X, X))
X_I_X = np.kron(X, np.kron(I, X))
X_X_X = np.kron(X, np.kron(X, X))

SWAP = np.array([[1, 0, 0 ,0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

SWAP_I = np.kron(SWAP, I)
I_SWAP = np.kron(I, SWAP)

A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])


A_X = np.kron(A.copy(), X)
A_U = np.kron(I, A.copy())





u, cs, vdh = cossin(U, p=4, q=4)

print(u)
print("\n")


# u_1 = u[:4, :4]
# A_U = np.kron(I, u_1.T)
# u = clean_matrix(A_U @ u)



# u = X_X_I @ u
# u = SWAP_I @ I_SWAP @ u
# u_1 = u[4:, 4:]
# G = create_single_G(u_1, 0, 1)
# B_U =  I_SWAP @ SWAP_I @  np.kron(I, G)
# u = clean_matrix(B_U @ u)
# u = X_X_I @ u
# print(clean_matrix(u))
# print("\n")

# u = X_X_I @ u
# u = SWAP_I @ I_SWAP @ u
# print(clean_matrix(u))
# print("\n")
# u_1 = u[:4, 4:]
# print(u_1)
# exit()
# G = create_single_G(u_1, 2, 3)
# B_U =  I_SWAP @ SWAP_I @  np.kron(I, G)
# u = clean_matrix(B_U @ u)
# u = I_I_X @ u
# print(clean_matrix(u))
# print("\n")

# exit()






u_1 = u[4:, 4:]
G = create_single_G(u_1, 0, 1)
A_U = np.kron(I, G)
u = clean_matrix(A_U @ u)

u_1 = u[4:, 4:]
G = create_single_G(u_1, 0, 2)
A_U = np.kron(I, G)
u = clean_matrix(A_U @ u)

u_1 = u[4:, 4:]
G = create_single_G(u_1, 0, 3)
A_U = np.kron(I, G)
u = clean_matrix(A_U @ u)

u_1 = u[:4, :4]
G = create_single_G(u_1, 1, 2)
A_U = np.kron(I, G)
u = clean_matrix(A_U @ u)

u_1 = u[:4, :4]
G = create_single_G(u_1, 1, 3)
A_U = np.kron(I, G)
u = clean_matrix(A_U @ u)

u_1 = u[:4, :4]
G = create_single_G(u_1, 2, 3)
A_U = np.kron(I, G)
u = clean_matrix(A_U @ u)


u = SWAP_I @ I_SWAP @ u
u_1 = u[4:, 4:]
print(clean_matrix(u))
print("\n")
exit()
G = create_single_G(u_1, 2, 3)
B_U =  I_SWAP @ SWAP_I @  np.kron(I, G)
u = clean_matrix(B_U @ u)


u = I_I_X @ u

u = X_I_I @ u
print(clean_matrix(u))
print("\n")
exit()

# print(I_SWAP @ SWAP_I @ A_U @ SWAP_I @ I_SWAP @ (np.kron(I, np.kron(I, X))))
# exit()
# A_U = np.kron(I, A.copy())
# B_U = np.kron(A.copy(), I)
# print(A_X)





U_2 = U.copy()
G_2_list = []

############## AI
U_2_2 = U_2[:4, :4]
G_2 = two_level_decomposition_left(U_2_2)
A_U = np.kron(I, G_2)
U_2 = clean_matrix(A_U @ U_2)

print(U_2)
print("\n")
# exit()

# ################# BI
# U_2 = SWAP_I @ I_SWAP @ U_2
# U_2_2 = U_2[:4, :4]
# G_2 = create_single_G(U_2_2, 0, 1)
# B_U = I_SWAP @ SWAP_I @ np.kron(I, G_2)
# U_2 = clean_matrix(B_U @ U_2)

# U_2 = SWAP_I @ I_SWAP @ U_2
# U_2_2 = U_2[:4, :4]
# G_2 = create_single_G(U_2_2, 0, 2)
# B_U = I_SWAP @ SWAP_I @ np.kron(I, G_2)
# U_2 = clean_matrix(B_U @ U_2)

# U_2 = SWAP_I @ I_SWAP @ U_2
# U_2_2 = U_2[:4, :4]
# G_2 = create_single_G(U_2_2, 0, 3)
# B_U = I_SWAP @ SWAP_I @ np.kron(I, G_2)
# U_2 = clean_matrix(B_U @ U_2)
# print(U_2)
# print("\n")

# ######################### AI




# U_2_2 = U_2[:4, :4]
# G_2 = create_single_G(U_2_2, 0, 1)
# A_U = np.kron(I, G_2)
# U_2 = clean_matrix(A_U @ U_2)








G_list = []

G = create_single_G(U, 0, 1)
U = G @ U
G_list.append(G)
G = create_single_G(U, 0, 2)
U = G @ U
G_list.append(G)
G = create_single_G(U, 0, 3)
U = G @ U
G_list.append(G)
G = create_single_G(U, 1, 2)
U = G @ U
G_list.append(G)
G = create_single_G(U, 1, 3)
U = G @ U
G_list.append(G)
G = create_single_G(U, 2, 3)
U = G @ U
G_list.append(G)


G = create_single_G(U, 0, 4)
U = G @ U
G_list.append(G)
G = create_single_G(U, 0, 5)
U = G @ U
G_list.append(G)
G = create_single_G(U, 0, 6)
U = G @ U
G_list.append(G)
G = create_single_G(U, 0, 7)
U = G @ U
G_list.append(G)







G = create_single_G(U, 1, 4)
U = G @ U
G_list.append(G)
G = create_single_G(U, 1, 5)
U = G @ U
G_list.append(G)
G = create_single_G(U, 1, 6)
U = G @ U
G_list.append(G)
G = create_single_G(U, 1, 7)
U = G @ U
G_list.append(G)
G = create_single_G(U, 2, 4)
U = G @ U
G_list.append(G)
G = create_single_G(U, 2, 5)
U = G @ U
G_list.append(G)
G = create_single_G(U, 2, 6)
U = G @ U
G_list.append(G)
G = create_single_G(U, 2, 7)
U = G @ U
G_list.append(G)
G = create_single_G(U, 3, 4)
U = G @ U
G_list.append(G)
G = create_single_G(U, 3, 5)
U = G @ U
G_list.append(G)
G = create_single_G(U, 3, 6)
U = G @ U
G_list.append(G)
G = create_single_G(U, 3, 7)
U = G @ U
G_list.append(G)
G = create_single_G(U, 4, 5)
U = G @ U
G_list.append(G)
G = create_single_G(U, 4, 6)
U = G @ U
G_list.append(G)
G = create_single_G(U, 4, 7)
U = G @ U
G_list.append(G)
G = create_single_G(U, 5, 6)
U = G @ U
G_list.append(G)
G = create_single_G(U, 5, 7)
U = G @ U
G_list.append(G)
G = create_single_G(U, 6, 7)
U = G @ U
G_list.append(G)


print(clean_matrix(U))
print("\n")
exit()
print("U ///////////////////////")
two_level_decomposition_left(U)
# print("A_U ///////////////////////")
# two_level_decomposition_left(A_U)
# print("B_U ///////////////////////")
# two_level_decomposition_left(B_U)

exit()



U_1 = U[:4,:4]
G = two_level_decomposition_left(U_1)
A_U = np.kron(I, G)
U = clean_matrix(A_U @ U)

print(U)
print("\n")
# exit()


U = SWAP_I @ I_SWAP @ U
print(U)
print("\n")
U_2 = U[4:, :4]
G = two_level_decomposition_left(U_2)
# G = create_single_G(U_2, 0, 2)
B_U = I_SWAP @ SWAP_I @ np.kron(I, G)
U = clean_matrix(B_U @ U)

print(U)
print("\n")
# exit()

# # I_SWAP @ SWAP_I @ BB @ SWAP_I @ I_SWAP

# U_1 = U[:4,:4]
# G = two_level_decomposition_left(U_1)
# A_U = np.kron(I, G)
# U = clean_matrix(A_U @ U)

# print(U)
# print("\n")
