import numpy as np
from numpy.linalg import matrix_power

# Main function checking type of a matrix
def check_matrix_type(A):
    d = A.ndim
    # If it is 1-dimensional matrix,
    # check for Singleton or Row matrix
    if d == 1:
        check_singleton(A)
        if A.size > 1:
            check_row(A)
    # If it is 2-dimensional matrix
    elif d > 1: 
        if A.shape[1] == 1:
            check_column(A) 
            # Note: Column matrix is considered 2-dimensional
        else:
            check_null(A)
            # Check for Real or Complex
            check_real(A)
            if check_complex(A):
                check_unitary(A)
            
            check_rect(A)
            # Many types require the matrix to be a Square Matrix
            if check_square(A):
                if check_diag(A):
                    check_identity(A)
                    check_scalar(A)
                elif not check_null(A):
                    check_strict_tri(A)
                else:
                    check_tri(A)
                # Check for Symmetric or Skew-symmetric
                check_symmetric(A)
                check_skew_symc(A)
                # Check for special types
                check_involutary(A)
                check_idempotent(A)
                check_nilpotent(A)
                check_orthogonal(A)

# Functions for individual types of matrices
def check_singleton(A):
    # Checks for Singleton Matrix
    s = np.size(A)
    if (s==1):
        print('Singleton Matrix')
        return True
    else:
        return False

def check_row(A):
    # Checks for Row Matrix
    s = np.size(A)
    p = np.size(np.shape(A))
    if (p==1) and (s>1):
        print('Row Matrix')
        return True
    else:
        return False

def check_column(A):
    # Checks for Column Matrix
    (m,n) = np.shape(A)
    if (m>1) and (n==1):
        print('Column Matrix')
        return True
    else:
        return False

def check_rect(A):
    # Checks for Rectangular Matrix
    (m,n) = np.shape(A)
    if not(m==n):
        print('Rectangular Matrix')
        return True
    else:
        return False

def check_square(A):
    # Checks for Square Matrix
    (m,n) = np.shape(A)
    if (m==n):
        print('Square Matrix')
        return True
    else:
        return False

def check_null(A):
    # Checks for Null or Zero Matrix
    if not np.any(A):
        print('Null Matrix')
        return True
    else:
        return False

def check_real(A):
    # Checks for Real Matrix
    if np.isrealobj(A):
        print('Real Matrix')
        return True
    else:
        return False

def check_complex(A):
    # Checks for Complex Matrix
    if np.iscomplexobj(A):
        print('Complex Matrix')
        return True
    else:
        return False

def check_diag(A):
    # Checks for Diagonal Matrix
    if np.array_equal(A,np.diag(np.diagonal(A))):
        print('Diagonal Matrix')
        return True
    else:
        return False

def check_scalar(A):
    # Checks for Scalar Matrix
    if np.array_equal(A,np.diag(np.diagonal(A))) and \
       (len(np.unique(np.diagonal(A)))==1):
        print('Scalar Matrix')
        return True
    else:
        return False

def check_identity(A):
    # Checks for Identity Matrix
    m = np.shape(A)[0]
    if np.array_equal(A,np.diag(np.diagonal(A))) and \
       (len(np.unique(np.diagonal(A)))==1) and \
        A[0,0]==1:
        print(f'Identity Matrix of order {m}')
        return True
    else:
        return False

def check_tri(A):
    # Checks for Triangular Matrix
    if np.array_equal(A,np.triu(A)):
        print('Upper Triangular Matrix')
    elif np.array_equal(A,np.tril(A)):
        print('Lower Triangular Matrix')
        return True
    else:
        return False

def check_strict_tri(A):
    # Checks for Strict Triangular Matrix
    B = np.copy(A)
    np.fill_diagonal(B,0)
    if np.array_equal(A,B):
        print('Strictly Triangular Matrix')
        return True
    else:
        return False

def check_symmetric(A):
    # Checks for Symmetric Matrix
    if np.array_equal(A,A.T):
        print('Symmetric Matrix')
        return True
    else:
        return False

def check_skew_symc(A):
    # Checks for Skew-symmetric Matrix
    if np.array_equal(-A,A.T):
        print('Skew-symmetric Matrix')
        return True
    else:
        return False

def check_involutary(A):
    # Checks for Involutary Matrix
    m = np.shape(A)[0]
    A_sq = np.matmul(A,A)
    if np.array_equal(A_sq,np.eye(m)):
        print('Involutary Matrix')
        return True
    else:
        return False

def check_idempotent(A):
    # Checks for Idempotent Matrix
    A_sq = np.matmul(A,A)
    if np.array_equal(A,A_sq):
        print('Idempotent Matrix')
        return True
    else:
        return False

def check_nilpotent(A):
    # Checks for Nilpotent Matrix
    m = np.shape(A)[0]
    for power in range(2,11):
        A_pow = matrix_power(A,power)
        if np.array_equal(np.zeros([m,m]),A_pow):
            print(f'Nilpotent Matrix with index {power}')
            return True    
        elif power==10:
            return False
        else:
            continue

def check_orthogonal(A):
    # Checks for Orthogonal Matrix
    m = np.shape(A)[0]
    if np.array_equal(np.matmul(A,A.T),np.eye(m)):
        print('Orthogonal Matrix')
        return True
    else:
        return False

def check_unitary(A):
    # Checks for Unitary Matrix
    m = np.shape(A)[0]
    if np.array_equal(np.matmul(A,np.conj(A.T)),np.eye(m)):
        print('Unitary Matrix')
        return True
    else:
        return False