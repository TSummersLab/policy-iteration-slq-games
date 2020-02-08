import numpy as np
import numpy.linalg as la
import numpy.random as npr
import scipy.linalg as sla
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../utility')
from matrixmath import mdot, specrad, solveb, dlyap, dare_gain, is_pos_def, vec, sympart, kron, mdot

import warnings
from warnings import warn


def gdlyap(problem_data, K, L, matrixtype='P', algo='linsolve', show_warn=False, check_pd=False, P00=None, S00=None):
    """
    Solve a discrete-time generalized Lyapunov equation
    for stochastic linear systems with multiplicative noise.
    """

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]
    q = Ai.shape[0]
    r = Bj.shape[0]
    s = Ck.shape[0]

    AKL = A + np.dot(B, K) + np.dot(C, L)

    stable = True
    if algo=='linsolve':
        if matrixtype=='P':
            # Intermediate terms
            Aunc_P = np.sum([varAi[i]*kron(Ai[i].T) for i in range(q)], axis=0)
            BKunc_P = np.sum([varBj[j]*kron(np.dot(K.T, Bj[j].T)) for j in range(r)], axis=0)
            CLunc_P = np.sum([varCk[k]*kron(np.dot(L.T, Ck[k].T)) for k in range(s)], axis=0)

            # Compute matrix and vector for the linear equation solver
            Alin_P = np.eye(n*n) - (kron(AKL.T) + Aunc_P + BKunc_P + CLunc_P)
            blin_P = vec(Q) + np.dot(kron(K.T), vec(R)) - np.dot(kron(L.T), vec(S))
            # Solve linear equations
            xlin_P = la.solve(Alin_P, blin_P)
            # Reshape
            P = np.reshape(xlin_P, [n, n])
            if check_pd:
                stable = is_pos_def(P)
        # elif matrixtype=='S':
        #     # Intermediate terms
        #     Aunc_S = np.zeros([n*n,n*n])
        #     for i in range(q):
        #         Aunc_S = Aunc_S + a[i]*kron(Ai[i])
        #     BKunc_S = np.zeros([n*n,n*n])
        #     for j in range(r):
        #         BKunc_S = BKunc_S + b[j]*kron(np.dot(Bj[j],K))
        #     # Compute matrix and vector for the linear equation solver
        #     Alin_S = np.eye(n*n) - kron(AK) - Aunc_S - BKunc_S
        #     blin_S = vec(S0)
        #     # Solve linear equations
        #     xlin_S = la.solve(Alin_S, blin_S)
        #     # Reshape
        #     S = np.reshape(xlin_S, [n, n])
        #     if check_pd:
        #         stable = is_pos_def(S)
        # elif matrixtype=='PS':
        #     P = gdlyap(problem_data, K, L, matrixtype='P', algo='linsolve')
        #     S = gdlyap(problem_data, K, L, matrixtype='S', algo='linsolve')
#     elif algo=='iterative':
#         # Implicit iterative solution to generalized discrete Lyapunov equation
#         # Inspired by https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7553367
#         # In turn inspired by https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122100X0020X/1-s2.0-089812219500119J/main.pdf?x-amz-security-token=AgoJb3JpZ2luX2VjECgaCXVzLWVhc3QtMSJIMEYCIQD#2F00Re8b3wnBnFpZQrjkOeXrNI4bYZ1J6#2F9BcJptZYAAIhAOQjTsZX573uFFEr7QveHx4NaZYWxlZfRN6hr5h1GJWWKuMDCOD#2F#2F#2F#2F#2F#2F#2F#2F#2F#2FwEQAhoMMDU5MDAzNTQ2ODY1IgxqkGe6i8wGmEj6YAwqtwNDKbotYDExP2D6PO8MrlIKYmHCtJhTu1CXLv0N5NKsYT90H2rJTNU0MvqsUsnXtbn6C9t9ed31XTf#2BHc7KrGmpOils7zgrjV1QG4LP0Fu2OcT4#2F#2FOGLWNvVjWY9gOLEHSeG5LhvBbxJiZVrI#2Bm1QAIVz5dxH5DVB27A2e9OmRrswrpPWuxQV#2BUvLkz2dVM4qSkvaDA#2F3KEJk9s0XE74mjO4ZHX7d9Q2aYwxsvFbII6Hms#2FZmB6125tBTwzd0K5xDit5kaoiYadOetp3M#2FvCdaiO0QeQwkV4#2FUaprOIIQGwJaMJuMNe7xInQxF#2B#2FmER81JhWEpBHBmz#2F5p0d2tU7F2oTDc2OR#2BV5dTKab47zgUw648fDT7ays0TQzqTMGnGcX9wIQpxSCam2E8Bhg6tsEs0#2FudddgnsiId368q70xai6ucMfabMSCqnv7O0OZqPVwY5b7qk4mxKIehpIzV6rrtXSAGrH95WGlgGz#2Fhmg9Qq6AUtb8NSqyYw0uZ00E#2FPZmNTnI3nwxjOA5qhyEbw3uXogRwYrv0dLkd50s7oO3mlYFeJDBurhx11t9p94dFqQq7sDY70m#2F4xMNCcmuUFOrMBY1JZuqtQ7QFBVbgzV#2B4xSHV6#2FyD#2F4ezttczZY3eSASJpdC4rjYHXcliiE7KOBHivchFZMIYeF3J4Nvn6UykX5sNfRANC2BDPrgoCQUp95IE5kgYGB8iEISlp40ahVXK62GhEASJxMjJTI9cJ2M#2Ff#2BJkwmqAGjTsBwjxkgiLlHc63rBAEJ2e7xoTwDDql3FSSYcvKzwioLfet#2FvXWvjPzz44tB3#2BTvYamM0uq47XPlUFcTrw#3D&AWSAccessKeyId=ASIAQ3PHCVTYWXNG3EKG&Expires=1554423148&Signature=Ysi80usGGEjPCvw#2BENTSD90NgVs#3D&hash=e5cf30dad62b0b57d7b7f5ba524cccacdbb36d2f747746e7fbebb7717b415820&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=089812219500119J&tid=spdf-a9dae0e9-65fd-4f31-bf3f-e0952eb4176c&sid=5c8c88eb95ed9742632ae57532a4a6e1c6b1gxrqa&type=client
#         # Faster for large systems i.e. >50 states
#         # Options
#         max_iters = 1000
#         epsilon_P = 1e-5
#         epsilon_S = 1e-5
#         # Initialize
#         if matrixtype=='P' or matrixtype=='PS':
#             if P00 is None:
#                 P = np.zeros([n,n])
#             else:
#                 P = P00
#         if matrixtype=='S' or matrixtype=='PS':
#             if S00 is None:
#                 S = np.zeros([n,n])
#             else:
#                 S = S00
#         iterc = 0
#         converged = False
#         stop = False
#         while not stop:
#             if matrixtype=='P' or matrixtype=='PS':
#                 P_prev = P
#                 APAunc = np.zeros([n,n])
#                 for i in range(q):
#                     APAunc += a[i]*mdot(Ai[i].T,P,Ai[i])
#                 BPBunc = np.zeros([n,n])
#                 for j in range(r):
#                     BPBunc += b[j]*mdot(K.T,Bj[j].T,P,Bj[j],K)
#                 AAP = AK.T
#                 QQP = sympart(Q + mdot(K.T,R,K) + APAunc + BPBunc)
#                 P = dlyap(AAP,QQP)
#                 if np.any(np.isnan(P)) or not is_pos_def(P):
#                     stable = False
#                 converged_P = la.norm(P-P_prev,2)/la.norm(P,2) < epsilon_P
#             if matrixtype=='S' or matrixtype=='PS':
#                 S_prev = S
#                 ASAunc = np.zeros([n,n])
#                 for i in range(q):
#                     ASAunc += a[i]*mdot(Ai[i],S,Ai[i].T)
#                 BSBunc = np.zeros([n,n])
#                 for j in range(r):
#                     BSBunc = b[j]*mdot(Bj[j],K,S,K.T,Bj[j].T)
#                 AAS = AK
#                 QQS = sympart(S0 + ASAunc + BSBunc)
#                 S = dlyap(AAS,QQS)
#                 if np.any(np.isnan(S)) or not is_pos_def(S):
#                     stable = False
#                 converged_S = la.norm(S-S_prev,2)/la.norm(S,2) < epsilon_S
#             # Check for stopping condition
#             if matrixtype=='P':
#                 converged = converged_P
#             elif matrixtype=='S':
#                 converged = converged_S
#             elif matrixtype=='PS':
#                 converged = converged_P and converged_S
#             if iterc >= max_iters:
#                 stable = False
#             else:
#                 iterc += 1
#             stop = converged or not stable
# #        print('\ndlyap iters = %s' % str(iterc))
#
#     elif algo=='finite_horizon':
#         P = np.copy(Q)
#         Pt = np.copy(Q)
#         S = np.copy(Q)
#         St = np.copy(Q)
#         converged = False
#         stop = False
#         while not stop:
#             if matrixtype=='P' or matrixtype=='PS':
#                 APAunc = np.zeros([n,n])
#                 for i in range(q):
#                     APAunc += a[i]*mdot(Ai[i].T,Pt,Ai[i])
#                 BPBunc = np.zeros([n,n])
#                 for j in range(r):
#                     BPBunc += b[j]*mdot(K.T,Bj[j].T,Pt,Bj[j],K)
#                 Pt = mdot(AK.T,Pt,AK)+APAunc+BPBunc
#                 P += Pt
#                 converged_P = np.abs(Pt).sum() < 1e-15
#                 stable = np.abs(P).sum() < 1e10
#             if matrixtype=='S' or matrixtype=='PS':
#                 ASAunc = np.zeros([n,n])
#                 for i in range(q):
#                     ASAunc += a[i]*mdot(Ai[i],St,Ai[i].T)
#                 BSBunc = np.zeros([n,n])
#                 for j in range(r):
#                     BSBunc = b[j]*mdot(Bj[j],K,St,K.T,Bj[j].T)
#                 St = mdot(AK,Pt,AK.T)+ASAunc+BSBunc
#                 S += St
#                 converged_S = np.abs(St).sum() < 1e-15
#                 stable = np.abs(S).sum() < 1e10
#             if matrixtype=='P':
#                 converged = converged_P
#             elif matrixtype=='S':
#                 converged = converged_S
#             elif matrixtype=='PS':
#                 converged = converged_P and converged_S
#             stop = converged or not stable
    if not stable:
        P = None
        S = None
        if show_warn:
            warnings.simplefilter('always', UserWarning)
            warn('System is possibly not mean-square stable')
    if matrixtype=='P':
        return P
    elif matrixtype=='S':
        return S
    elif matrixtype=='PS':
        return P, S


def qfun(problem_data, P, outputs_needed=None):
    if outputs_needed is None:
        outputs_needed = 'xuv_matrix'
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    if outputs_needed == 'xuv_list' or outputs_needed == 'xuv_matrix':
        APAi = np.sum([varAi[i]*mdot(Ai[i].T, P, Ai[i]) for i in range(q)], axis=0)
    BPBj = np.sum([varBj[j]*mdot(Bj[j].T, P, Bj[j]) for j in range(r)], axis=0)
    CPCk = np.sum([varCk[k]*mdot(Ck[k].T, P, Ck[k]) for k in range(s)], axis=0)

    if outputs_needed == 'uv_list' or outputs_needed == 'xuv_list':
        Qux = mdot(B.T, P, A)
        Qvx = mdot(C.T, P, A)
        Quu = R + mdot(B.T, P, B) + BPBj
        Quv = mdot(B.T, P, C)
        Qvu = Quv.T
        Qvv = -S + mdot(C.T, P, C) + CPCk
        if outputs_needed == 'uv_list':
            outputs = Qux, Qvx, Quu, Quv, Qvu, Qvv
        elif outputs_needed == 'xuv_list':
            Qxx = Q + mdot(A.T, P, A) + APAi
            outputs = Qxx, Qux, Qvx, Quu, Quv, Qvu, Qvv
    elif outputs_needed == 'xuv_matrix':
        ABC = np.hstack([A, B, C])
        X = sla.block_diag(Q, R, -S)
        Y = mdot(ABC.T, P, ABC)
        Z = sla.block_diag(APAi, BPBj, CPCk)
        outputs = X + Y + Z
    return outputs


def policy_iteration(problem_data, K0=None, L0=None, num_iterations=100):
    # Policy iteration
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]

    if K0 is None:
        K0 = np.zeros([m, n])
    if L0 is None:
        L0 = np.zeros([p, n])
    K = np.copy(K0)
    L = np.copy(L0)

    # Check initial policies are stabilizing
    if specrad(A+B.dot(K0)+C.dot(L0)) > 1:
        raise Exception("Initial policies are not stabilizing!")

    P_history = np.zeros([num_iterations, n, n])
    K_history = np.zeros([num_iterations, m, n])
    L_history = np.zeros([num_iterations, p, n])
    c_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        # Policy evaluation
        P = gdlyap(problem_data, K, L)

        # Record history
        P_history[i] = P
        K_history[i] = K
        L_history[i] = L
        c_history[i] = np.trace(P)

        # Policy improvement
        Qux, Qvx, Quu, Quv, Qvu, Qvv = qfun(problem_data, P, outputs_needed='uv_list')
        QuvQvvinv = solveb(Quv, Qvv)
        QvuQuuinv = solveb(Qvu, Quu)
        K = -la.solve(Quu-QuvQvvinv.dot(Qvu), Qux-QuvQvvinv.dot(Qvx))
        L = -la.solve(Qvv-QvuQuuinv.dot(Quv), Qvx-QvuQuuinv.dot(Qux))

    return P, K, L, P_history, K_history, L_history, c_history


def value_iteration(problem_data, P0=None, num_iterations=100):
    # Value iteration
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]

    if P0 is None:
        P0 = np.eye(n)
    P = np.copy(P0)

    P_history = np.zeros([num_iterations, n, n])
    c_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        # Record history
        P_history[i] = P
        c_history[i] = np.trace(P)

        # Value improvement
        Qxx, Qux, Qvx, Quu, Quv, Qvu, Qvv = qfun(problem_data, P, outputs_needed='xuv_list')
        QxuQxv = np.vstack([Qux, Qvx])
        QuuQuvQvuQvv = np.block([[Quu, Quv], [Qvu, Qvv]])
        P = Qxx - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv))

    # Policy synthesis
    QuvQvvinv = solveb(Quv, Qvv)
    QvuQuuinv = solveb(Qvu, Quu)
    K = -la.solve(Quu-QuvQvvinv.dot(Qvu), Qux-QuvQvvinv.dot(Qvx))
    L = -la.solve(Qvv-QvuQuuinv.dot(Quv), Qvx-QvuQuuinv.dot(Qux))

    return P, K, L, P_history, c_history


def verify_gare(problem_data, P, algo_str=None):
    # Verify that the GARE is solved by the solution P
    if algo_str is None:
        algo_str = ''

    Qxx, Qux, Qvx, Quu, Quv, Qvu, Qvv = qfun(problem_data, P, outputs_needed='xuv_list')
    QxuQxv = np.vstack([Qux, Qvx])
    QuuQuvQvuQvv = np.block([[Quu, Quv], [Qvu, Qvv]])

    print(algo_str)
    print('-'*len(algo_str))
    print('Left-hand side of the GARE')
    print(P)
    print('')
    print('Right-hand side of the GARE')
    print(Qxx - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv)))
    print('\n')
    return


if __name__ == "__main__":
    from time import time
    from data_io import save_problem_data, load_problem_data
    from problem_data_gen import gen_rand_problem_data

    # Problem data
    problem_data_id = 1581199445 # 5-state random system
    # problem_data_id = 1581200195 # 3-state example system
    problem_data = load_problem_data(problem_data_id)

    # problem_data = gen_rand_problem_data(rho=0.9, seed=1)
    # problem_data_id = int(time())
    # save_problem_data(problem_data_id, problem_data)

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]
    q = Ai.shape[0]
    r = Bj.shape[0]
    s = Ck.shape[0]

    # Initial gains
    initial_gain_method = 'zero'
    # initial_gain_method = 'dare'
    if initial_gain_method == 'zero':
        K0 = np.zeros([m, n])
        L0 = np.zeros([p, n])
    elif initial_gain_method == 'dare':
        Pare, Kare = dare_gain(A, B, Q, R)
        K0 = Kare
        L0 = np.zeros([p, n])

    AKL0 = A + B.dot(K0) + C.dot(L0)
    QKL0 = Q + mdot(K0.T, R, K0) - mdot(L0.T, S, L0)
    P0 = gdlyap(problem_data, K0, L0)

    # Settings
    num_iterations = 20
    t_history = np.arange(num_iterations)+1

    # Policy iteration
    P_pi, K_pi, L_pi, P_history_pi, K_history_pi, L_history_pi, c_history_pi = policy_iteration(problem_data, K0, L0, num_iterations)
    verify_gare(problem_data, P_pi, algo_str='Policy iteration')

    # Value iteration
    # Start value iteration at the same initial P as from policy iteration
    P_vi, K_vi, L_vi, P_history_vi, c_history_vi = value_iteration(problem_data, P0, num_iterations)
    verify_gare(problem_data, P_vi, algo_str='Value iteration')

    # Plotting
    plt.close('all')

    # Cost-to-go matrix
    fig, ax = plt.subplots(ncols=2)
    plt.suptitle('Value matrix (P)')
    ax[0].imshow(P_pi)
    ax[1].imshow(P_vi)
    # ax[0].set_title('ARE')
    ax[0].set_title('Policy iteration')
    ax[1].set_title('Value iteration')
    # Specific stuff for Ben's monitor
    fig.canvas.manager.window.setGeometry(3600, 600, 800, 600)
    fig.canvas.manager.window.showMaximized()
    plt.show()

    # Cost over iterations
    fig, ax = plt.subplots()
    # ax.plot(np.ones(num_iterations)*np.trace(Pare), 'r--')
    ax.plot(t_history, c_history_pi)
    ax.plot(t_history, c_history_vi)
    # plt.legend(['ARE', 'Policy iteration', 'Value iteration'])
    plt.legend(['Policy iteration', 'Value iteration'])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    if num_iterations <= 20:
        plt.xticks(np.arange(num_iterations)+1)

    # Specific stuff for Ben's monitor
    fig.canvas.manager.window.setGeometry(3600, 600, 800, 600)
    fig.canvas.manager.window.showMaximized()
    plt.show()