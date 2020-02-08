import numpy as np
import numpy.random as npr
import numpy.linalg as la
import sys
sys.path.insert(0,'../utility')
from matrixmath import specrad


def gen_ex3_ABC():
    A = np.array([[0.7, 0.2, 0],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.4, 0.3]])
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.2, 0.6]])
    C = np.array([[1.0, 0.3],
                  [0.4, 1.0],
                  [0.6, 0.4]])
    return A, B, C

def gen_rand_ABC(rho=None, seed=1):
    npr.seed(seed)
    if rho is None:
        rho = 0.9
    n = 5
    m = 5
    p = 3
    A = npr.randn(n, n).round(1)
    A = A*(rho/specrad(A))
    B = npr.rand(n, m).round(1)
    C = npr.rand(n, p).round(1)
    return A, B, C


def gen_rand_problem_data(rho=None, seed=1):
    npr.seed(seed)

    A, B, C = gen_rand_ABC(rho, seed)
    # A, B, C = gen_ex3_ABC()

    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]

    q = np.copy(n)
    r = np.copy(m)
    s = np.copy(p)

    Ai = npr.randn(q, n, n)
    Bj = npr.randn(r, n, m)
    Ck = npr.randn(s, n, p)
    Ai = Ai / la.norm(Ai, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    Bj = Bj / la.norm(Bj, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    Ck = Ck / la.norm(Ck, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    varAi = 0.1*np.ones(q)
    varBj = 0.1*np.ones(r)
    varCk = 0.1*np.ones(s)

    Q = np.eye(n)
    R = np.eye(m)
    S = 6*np.eye(p)

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    problem_data_values = [A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S]
    problem_data = dict(zip(problem_data_keys, problem_data_values))
    return problem_data