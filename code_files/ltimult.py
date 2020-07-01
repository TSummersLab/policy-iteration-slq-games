import numpy as np
import numpy.linalg as la

import warnings
from warnings import warn

from matrixmath import kron, vec, is_pos_def, specrad


def cost_operator_P(problem_data, K, L):
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    q = Ai.shape[0]
    r = Bj.shape[0]
    s = Ck.shape[0]

    AKL = A + np.dot(B, K) + np.dot(C, L)
    Aunc_P = np.sum([varAi[i]*kron(Ai[i].T) for i in range(q)], axis=0)
    BKunc_P = np.sum([varBj[j]*kron(np.dot(K.T, Bj[j].T)) for j in range(r)], axis=0)
    CLunc_P = np.sum([varCk[k]*kron(np.dot(L.T, Ck[k].T)) for k in range(s)], axis=0)
    return kron(AKL.T) + Aunc_P + BKunc_P + CLunc_P


def cost_operator_S(problem_data, K, L):
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    q = Ai.shape[0]
    r = Bj.shape[0]
    s = Ck.shape[0]

    AKL = A + np.dot(B, K) + np.dot(C, L)
    Aunc_S = np.sum([varAi[i]*kron(Ai[i]) for i in range(q)], axis=0)
    BKunc_S = np.sum([varBj[j]*kron(np.dot(Bj[j], K)) for j in range(r)], axis=0)
    CLunc_S = np.sum([varCk[k]*kron(np.dot(Ck[k], L)) for k in range(s)], axis=0)
    return kron(AKL) + Aunc_S + BKunc_S + CLunc_S


def mean_square_stable(problem_data, K, L):
    return specrad(cost_operator_P(problem_data, K, L)) < 1


def gdlyap(problem_data, K, L, show_warn=False, check_pd=False):
    """
    Solve a discrete-time generalized Lyapunov equation
    for stochastic linear systems with multiplicative noise.
    """

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n = A.shape[1]

    stable = True
    # Compute matrix and vector for the linear equation solver
    Alin_P = np.eye(n*n)-cost_operator_P(problem_data, K, L)
    blin_P = vec(Q) + np.dot(kron(K.T), vec(R)) - np.dot(kron(L.T), vec(S))

    # Solve linear equations
    xlin_P = la.solve(Alin_P, blin_P)

    # Reshape
    P = np.reshape(xlin_P, [n, n])

    if check_pd:
        stable = is_pos_def(P)

    if not stable:
        P = None
        if show_warn:
            warnings.simplefilter('always', UserWarning)
            warn('System is possibly not mean-square stable')
    return P
