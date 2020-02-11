import numpy as np
import numpy.linalg as la

import warnings
from warnings import warn

import sys
sys.path.insert(0,'../utility')
from matrixmath import kron, vec, is_pos_def


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