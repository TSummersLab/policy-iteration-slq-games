import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../utility')
from matrixmath import mdot, specrad, solveb, dlyap, dare_gain


def policy_iteration(A, B, C, Q, R, S, K0=None, L0=None, num_iterations=100):
    # Policy iteration
    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]

    if K0 is None:
        K0 = np.zeros([m, n])
    if L0 is None:
        L0 = np.zeros([p, n])
    K = np.copy(K0)
    L = np.copy(K0)

    # Check initial policies are stabilizing
    if specrad(A+B.dot(K0)+C.dot(L0)) > 1:
        raise Exception("Initial policies are not stabilizing!")

    P_history = np.zeros([num_iterations, n, n])
    K_history = np.zeros([num_iterations, m, n])
    L_history = np.zeros([num_iterations, p, n])
    c_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        # Policy evaluation
        AKL = A + B.dot(K) + C.dot(L)
        QKL = Q + mdot(K.T, R, K) - mdot(L.T, S, L)
        P = dlyap(AKL.T, QKL)

        # Policy improvement

        # Evaluate Q-function
        Qux = mdot(B.T, P, A)
        Qvx = mdot(C.T, P, A)
        Quu = R + mdot(B.T, P, B)
        Quv = mdot(B.T, P, C)
        Qvu = Quv.T
        Qvv = -S + mdot(C.T, P, C)

        QuuQvvinv = solveb(Quu, Qvv)
        QvuQuuinv = solveb(Qvu, Quu)

        K = -la.solve(Quu-QuuQvvinv.dot(Qvu), Qux-QuuQvvinv.dot(Qvx))
        L = -la.solve(Qvv-QvuQuuinv.dot(Quv), Qvx-QvuQuuinv.dot(Qux))

        # Record history
        P_history[i] = P
        K_history[i] = K
        L_history[i] = L
        c_history[i] = np.trace(P)

    return P, K, L, P_history, K_history, L_history, c_history


A = np.array([[0.7, 0.2, 0],
              [0.3, 0.5, 0.2],
              [0.2, 0.4, 0.3]])

B = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [0.2, 0.6]])

C = np.array([[1.0, 0.3],
              [0.4, 1.0],
              [0.6, 0.4]])

n = A.shape[1]
m = B.shape[1]
p = C.shape[1]

Q = np.eye(n)
R = np.eye(m)
S = 1000*np.eye(p)

K0 = None
L0 = None

num_iterations = 20

P, K, L, P_history, K_history, L_history, c_history = policy_iteration(A, B, C, Q, R, S, K0, L0, num_iterations=num_iterations)

# Verify that the GARE is solved by the solution
Qux = mdot(B.T, P, A)
Qvx = mdot(C.T, P, A)
Quu = R + mdot(B.T, P, B)
Quv = mdot(B.T, P, C)
Qvu = Quv.T
Qvv = -S + mdot(C.T, P, C)

QxuQxv = np.vstack([Qux, Qvx])
QuuQuvQvuQvv = np.block([[Quu, Quv], [Qvu, Qvv]])

print('Right-hand side of the GARE')
print(P)
print('')
print('Left-hand side of the GARE')
print(Q + mdot(A.T, P, A) - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv)))
print('')

# Plotting
plt.plot(c_history)