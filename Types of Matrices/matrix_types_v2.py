# This version of matrix_types uses allclose() function instead of array_equal() function
# for catching all the edge cases

import numpy as np

def check_matrix_type(A):
    d = A.ndim
    if is_real(A):
        print('Real Matrix')
    if is_complex(A):
        print('Complex Matrix')

    if is_null(A):
        print('Zero or Null Matrix')
    if is_row(A):
        print('Row Matrix')

    if d == 0:
        print('You have entered a scalar value (a zero-dimensional matrix).')
    elif is_singleton(A):
            print('Singleton Matrix')
    elif d == 2 and np.shape(A)[0] != 1: 
        if is_column(A):
            print('Column Matrix') 
            # Note: Column matrix is considered 2-dimensional
        else:
            if is_rect(A):
                print('Rectangular Matrix')
            # Many types require the matrix to be a Square Matrix
            if is_square(A) and np.any(A):
                print('Square Matrix')
                if is_diag(A):
                    print('Diagonal Matrix')
                    if is_scalar(A):
                        print('Scalar Matrix')
                        if is_identity(A):
                            m = np.shape(A)[0]
                            print(f'Identity Matrix of order {m}')
                else:
                    if is_upper_tri(A):
                        print('Upper Triangular Matrix')
                    if is_lower_tri(A):
                        print('Lower Triangular Matrix')
                    if is_strict_tri(A):
                        print('Strictly Triangular Matrix')
                
                if is_symmetric(A):
                    print('Symmetric Matrix')
                elif is_skew_symc(A):
                    print('Skew-symmetric Matrix')

                if is_involutory(A):
                    print('Involutory Matrix')
                if is_idempotent(A):
                    print('Idempotent Matrix')
                if is_nilpotent(A):
                    k = nilpotence_index(A)
                    print(f'Nilpotent Matrix with index {k}')
                if is_orthogonal(A):
                    print('Orthogonal Matrix')
                if is_complex(A):
                    if is_unitary(A):
                        print('Unitary Matrix')
# Functions for individual types of matrices
def is_singleton(A):
    return np.size(A)==1

def is_row(A):
    return (np.size(np.shape(A))==1 or np.shape(A)[0]==1) and np.size(A)>1

def is_column(A):
    return np.shape(A)[0]>1 and np.shape(A)[1]==1

def is_rect(A):
    return not np.shape(A)[0]==np.shape(A)[1]

def is_square(A):
    return np.shape(A)[0]==np.shape(A)[1]

def is_null(A):
    return not np.any(A)

def is_real(A):
    return np.isrealobj(A)

def is_complex(A):
    return np.iscomplexobj(A)

def is_diag(A):
    return np.allclose(A,np.diag(np.diagonal(A)))

def is_scalar(A):
    return np.allclose(A,np.diag(np.diagonal(A))) and \
       len(np.unique(np.diagonal(A)))==1

def is_identity(A):
    return np.allclose(A,np.diag(np.diagonal(A))) and \
       len(np.unique(np.diagonal(A)))==1 and \
        A[0,0]==1

def is_upper_tri(A):
    return np.allclose(A,np.triu(A))

def is_lower_tri(A):
    return np.allclose(A,np.tril(A))

def is_strict_tri(A):
    B = np.copy(A)
    np.fill_diagonal(B,0)
    return np.allclose(A,B)

def is_symmetric(A):
    return np.allclose(A,A.T)

def is_skew_symc(A):
    return np.allclose(-A,A.T)

def is_involutory(A):
    m = np.shape(A)[0]
    A_sq = np.matmul(A,A)
    return np.allclose(A_sq,np.eye(m))

def is_idempotent(A):
    A_sq = np.matmul(A,A)
    return np.allclose(A,A_sq)

def is_nilpotent(A):
    m = np.shape(A)[0]
    for power in range(2,11):
        A_pow = np.linalg.matrix_power(A,power)
        if np.allclose(np.zeros([m,m]),A_pow):
            return True
        elif power==10:
            return False
        else:
            continue

def nilpotence_index(A):
    if is_nilpotent(A):
        m = np.shape(A)[0]
        power = 2
        while not np.allclose(np.zeros([m,m]),np.linalg.matrix_power(A,power)):
            power += 1
        return power
    else:
        print('Input is not a Nilpotent Matrix')

def is_orthogonal(A):
    m = np.shape(A)[0]
    return np.allclose(np.matmul(A,A.T),np.eye(m))

def is_unitary(A):
    m = np.shape(A)[0]
    return np.allclose(np.matmul(A,np.conj(A.T)),np.eye(m))
