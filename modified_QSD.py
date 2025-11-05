import numpy as np
import sys
from scipy.linalg import cossin, qr
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def clean_matrix(M):
    M = M.copy()
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-10: M[i][j] = float(0.0)
            if np.abs(np.imag(M[i][j])) < 1e-10: M[i][j] = np.real(M[i][j])
            M[i][j] = '{0:.8}'.format(M[i][j])
    return M

I = np.eye(2)
Z_4 = np.zeros((4, 4))

U = np.array([[ 0.62923587, -0.27810894,  0.06225542 , 0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
             [-0.12832786, -0.72115181 , 0.06330487 ,-0.25546329  ,0.1811286,  -0.15270451, 0.58015675, -0.03866454],
             [ 0.14146621, 0.33503048 , 0.63506461,  0.14609694, -0.17964295 , 0.2131513, 0.56863418,  0.20503807],
             [-0.16206701 , 0.04719766 , 0.46284018 ,-0.12025008  ,0.7869089,   0.20527736, -0.27689368 ,-0.06921729],
             [ 0.25624259 , 0.38011453 ,-0.34262361, -0.67547194,  0.23405039 , 0.00414033, 0.21956521 , 0.33644285],
             [ 0.23268996 ,-0.28038369, -0.23106108  ,0.05510674 ,-0.02434206 , 0.89985678, -0.00166854 , 0.02183679],
             [ 0.0135536  , 0.25475662 ,-0.4235493,   0.54214077 , 0.46218213, -0.0468658, 0.43173278, -0.24372693],
             [-0.64909721 ,-0.01092904, -0.15625139 , 0.19494701 , 0.02944304 , 0.09593373, 0.0091941  , 0.71132259]])

u, cs, vdh = cossin(U, p=4, q=4)

u_1 = u[:4, :4]
u_2 = u[4:, 4:]

u_1_u_2_dgr = u_1 @ np.conj(u_2.T)

eigval, eigvec = np.linalg.eig(u_1_u_2_dgr)

diag = np.diag([np.sqrt(x) for x in eigval])
V = eigvec
W = diag @ np.conj(V.T) @ u_2

block_diag = np.block([[diag, Z_4],
                        [Z_4, np.conj(diag.T)]])

I_V = np.kron(I, V)
I_W = np.kron(I, W)

# print(block_diag)

def extract_single_qubit_unitaries(mat):
    for i in range(0, len(mat), 2):
        yield mat[i:i+2, i:i+2]


it = list(extract_single_qubit_unitaries(block_diag))

print(block_diag)
# print(clean_matrix(I_V @ block_diag @ I_W))
# print(u_1)
# print(u_2)

