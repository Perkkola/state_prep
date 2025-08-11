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

U4 = general_U4([x for x in np.random.random_sample(16)])
print(U4)
print(U4 @ (U4.conjugate().T))
# print(np.exp(np.pi * 1j))
exit()

U_O = np.array([[ 0.62923587, -0.27810894,  0.06225542,  0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181,  0.06330487, -0.25546329,  0.1811286 , -0.15270451,  0.58015675, -0.03866454],
 [ 0.14146621,  0.33503048,  0.63506461,  0.14609694, -0.17964295,  0.2131513 ,  0.56863418,  0.20503807],
 [-0.16206701,  0.04719766,  0.46284018, -0.12025008,  0.7869089 ,  0.20527736, -0.27689368, -0.06921729],
 [ 0.25624259,  0.38011453, -0.34262361, -0.67547194,  0.23405039,  0.00414033,  0.21956521,  0.33644285],
 [ 0.23268996, -0.28038369, -0.23106108,  0.05510674, -0.02434206,  0.89985678, -0.00166854,  0.02183679],
 [ 0.0135536 ,  0.25475662, -0.4235493 ,  0.54214077,  0.46218213, -0.0468658 ,  0.43173278, -0.24372693],
 [-0.64909721, -0.01092904, -0.15625139,  0.19494701,  0.02944304,  0.09593373,  0.0091941 ,  0.71132259]])


u_o, cs_o, vdh_o = cossin(U_O, p=4, q=4)

def func(x):
    # A = np.array(x[:16]).reshape(4, 4)
    # B = np.array(x[16:32]).reshape(4, 4)
    # C = np.array(x[32:48]).reshape(4, 4)
    # D = np.array(x[48:64]).reshape(4, 4)
    # E = np.array(x[64:80]).reshape(4, 4)
    # F = np.array(x[80:96]).reshape(4, 4)

    # U_A = np.kron(I, A)
    # U_B = np.kron(B, I)
    # U_C = np.kron(I, C)
    # U_D = np.kron(D, I)
    # U_E = np.kron(I, E)
    # U_F = np.kron(F, I)

    # u_dc, cs_dc, vdh_dc = cossin(U_D @ U_C, p=4, q=4)

    # eqs = (np.array(vdh_o) - np.conjugate((U_C @ U_B @ U_A ).T)).flatten()
    # eqs = np.append(eqs, (np.conjugate((C @ B @ A).T) @ C @ B @ A - I_4).flatten())
    # eqs = np.append(eqs, (np.conjugate((F @ E @ D).T) @ F @ E @ D - I_4).flatten())

    return [np.exp(1j * x[0]) + 1]


# print(len(func([0.1 for x in np.random.random_sample(96)])))
# exit()
root = optimize.broyden2(func, [0.1])
# root = fsolve(func, [0.1])
# root = fsolve(func, [0.1 for x in np.random.random_sample(64)])
print(root)
print('Func values')
print(func(root))
exit()

U_O = Matrix([[ 0.62923587, -0.27810894,  0.06225542,  0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181,  0.06330487, -0.25546329,  0.1811286 , -0.15270451,  0.58015675, -0.03866454],
 [ 0.14146621,  0.33503048,  0.63506461,  0.14609694, -0.17964295,  0.2131513 ,  0.56863418,  0.20503807],
 [-0.16206701,  0.04719766,  0.46284018, -0.12025008,  0.7869089 ,  0.20527736, -0.27689368, -0.06921729],
 [ 0.25624259,  0.38011453, -0.34262361, -0.67547194,  0.23405039,  0.00414033,  0.21956521,  0.33644285],
 [ 0.23268996, -0.28038369, -0.23106108,  0.05510674, -0.02434206,  0.89985678, -0.00166854,  0.02183679],
 [ 0.0135536 ,  0.25475662, -0.4235493 ,  0.54214077,  0.46218213, -0.0468658 ,  0.43173278, -0.24372693],
 [-0.64909721, -0.01092904, -0.15625139,  0.19494701,  0.02944304,  0.09593373,  0.0091941 ,  0.71132259]])



I = np.eye(2)

SWAP = Matrix([[1, 0, 0 ,0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

# SWAP_I = Matrix(TensorProduct(SWAP, I))
# I_SWAP = Matrix(TensorProduct(I, SWAP))

B = np.array([[-0.317886,    0.77150728 ,-0.34916656 , 0.42638917],
 [-0.92321982, -0.17538743,  0.03804274 ,-0.33978988],
 [ 0.05827144, -0.35261667, -0.93047666 ,-0.08049298],
 [-0.20788961 ,-0.49967715 , 0.10415689,  0.83441956]])

A = np.array([[ 5.74415501e-01 , 1.29190548e-02 ,-6.87365420e-01,  4.44306999e-01],
 [-3.31787022e-01 ,-5.92553316e-01 ,-5.79567048e-01, -4.50444199e-01],
 [ 5.88479490e-01 , 2.53728912e-01,  3.34068965e-04, -7.67667518e-01],
 [-4.62229483e-01,  7.64418279e-01, -4.37756444e-01 ,-1.01871958e-01]])

U_B = np.kron(B, I)
U_A = np.kron(I, A)

u, cs, vdh = cossin(U_B, p=4, q=4)

# print(U_B)
# print(U_A)

print(u)
print(cs)
print(vdh)
exit()


# A = MatrixSymbol('A', 4, 4).as_explicit()
# B = MatrixSymbol('B', 4, 4).as_explicit()
# U_A = Matrix(TensorProduct(I, A))
# U_B = Matrix(TensorProduct(B, I))



I = Matrix(np.eye(2))

variables_indices = [x for x in range(1, 16)]

variables_string_A = reduce(lambda x, y: f"{x} " + f"a_{y}", variables_indices, f"a_0")
variables_symbols_A = symbols(variables_string_A)
variables_symbols_matrix_A = np.array(list(variables_symbols_A)).reshape((4, 4))

A = Matrix(variables_symbols_matrix_A)
U_A = Matrix(TensorProduct(I, A))

variables_string_B = reduce(lambda x, y: f"{x} " + f"b_{y}", variables_indices, f"b_0")
variables_symbols_B = symbols(variables_string_B)
variables_symbols_matrix_B = np.array(list(variables_symbols_B)).reshape((4, 4))

B = Matrix(variables_symbols_matrix_B)
U_B = Matrix(TensorProduct(B, I))
U_B_T = Matrix(TensorProduct(B, I)).transpose()

# pprint(U_B)
symbols = list(variables_symbols_A)
symbols.extend(list(variables_symbols_B))

# print(variables_symbols_B)



# matrix_symbols = A.free_symbols.copy()
# matrix_symbols.update(B.free_symbols)
# matrix_symbols = tuple(list(matrix_symbols))


U_O = Matrix([[ 0.62923587, -0.27810894,  0.06225542,  0.32819821,  0.21411186, -0.26067223, -0.16945149 , 0.52213037],
 [-0.12832786, -0.72115181,  0.06330487, -0.25546329,  0.1811286 , -0.15270451,  0.58015675, -0.03866454],
 [ 0.14146621,  0.33503048,  0.63506461,  0.14609694, -0.17964295,  0.2131513 ,  0.56863418,  0.20503807],
 [-0.16206701,  0.04719766,  0.46284018, -0.12025008,  0.7869089 ,  0.20527736, -0.27689368, -0.06921729],
 [ 0.25624259,  0.38011453, -0.34262361, -0.67547194,  0.23405039,  0.00414033,  0.21956521,  0.33644285],
 [ 0.23268996, -0.28038369, -0.23106108,  0.05510674, -0.02434206,  0.89985678, -0.00166854,  0.02183679],
 [ 0.0135536 ,  0.25475662, -0.4235493 ,  0.54214077,  0.46218213, -0.0468658 ,  0.43173278, -0.24372693],
 [-0.64909721, -0.01092904, -0.15625139,  0.19494701,  0.02944304,  0.09593373,  0.0091941 ,  0.71132259]])


v = Matrix([1, 0, 0, 0, 0, 0, 0, 0])
v_2 = Matrix([0, 1, 0, 0, 0, 0, 0, 0])
v_5 = Matrix([0, 0, 0, 0, 1, 0, 0, 0])
v_8 = Matrix([0, 0, 0, 0, 0, 0, 0, 1])


I_8 = Matrix(np.eye(8))

equations_matrix = U_B * U_A * U_O

eqs_1 = list(U_A * U_O.col(0) - U_B_T * v)
eqs_2 = list(U_A * U_O.col(7) - U_B_T * v_8)
# eqs_2 = list((U_B_T * U_O).col(0) - U_A * v)
# eqs_3 = list(U_A * (U_O.col(1)) - U_B_T * v_2)
# eqs_2 = list((U_B_T * U_O).col(1) - U_A * v_2)
# eqs_1 = list(equations_matrix.col(0) - v)
# eqs_2 = list(equations_matrix.row(0) - v.transpose())
# eqs_3 = list(equations_matrix.col(1) - v_2)
# eqs_4 = list(equations_matrix.row(1) - v_2.transpose())


# eqs_2 = list((equations_matrix.col(1) - v_2)[1:])
# eqs_3 = list(equations_matrix.col(2)[3:6])
# eqs_2 = [A.row(x).norm() ** 2 - 1 for x in range(4)]
# eqs_3 = [A.col(x).norm() ** 2 - 1 for x in range(4)]
eqs_1.extend(eqs_2)
# eqs_1.extend(eqs_3)
# eqs_1.extend(eqs_4)

# pprint(eqs_3)
# exit()


solutions = linsolve(eqs_1, symbols)

print(solutions)
exit()
solutions_list = list(list(solutions)[0])


first_solutions_matrix_A = Matrix(np.array(solutions_list[:16]).reshape(4, 4))
first_solutions_matrix_B = Matrix(np.array(solutions_list[16:]).reshape(4, 4))

pprint(first_solutions_matrix_A)
pprint(first_solutions_matrix_B)


U_A = Matrix(TensorProduct(I, first_solutions_matrix_A))
U_B = Matrix(TensorProduct(first_solutions_matrix_B, I))

pprint(simplify(U_B * U_A * U_O).row(0))

exit()

# I_4 = Matrix(np.eye(4))

first_solutions_matrix = Matrix(np.array(list(solutions)).reshape(4, 4).T)
first_solutions_transpose = Matrix(np.array(list(solutions)).reshape(4, 4))
U_2 = TensorProduct(I, first_solutions_matrix)

solution_matrix = U_2 * U_O

print(np.array(first_solutions_matrix * first_solutions_transpose))
# print(np.array(solution_matrix))

# new_eqs_1 = [first_solutions_matrix.row(x).norm() ** 2 - 1 for x in range(4)]
# new_eqs_2 = [first_solutions_matrix.col(x).norm() ** 2 - 1 for x in range(4)]

# new_eqs_1.extend(new_eqs_2)


# second_solutions = nonlinsolve(new_eqs_1, tuple(solutions.free_symbols))

# print(second_solutions)
# print("Col norm")
# print(np.array(simplify(solution_matrix.col(0).norm() ** 2)))
# print("Row norm")
# print(np.array(simplify(solution_matrix.row(0).norm() ** 2)))


# Solve for unitarity ##################################

# first_solutions_matrix_transpose = Matrix(np.array(list(solutions)).reshape(4, 4))
# solutions_matrix_equations = first_solutions_matrix * first_solutions_matrix_transpose - I_4
# solutions_matrix_equations_system = list(solutions_matrix_equations)

# solutions_2 = nonlinsolve(solutions_matrix_equations_system, tuple(solutions.free_symbols))
