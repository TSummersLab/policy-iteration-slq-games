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
        AKL = A + B.dot(K) + C.dot(L)
        QKL = Q + mdot(K.T, R, K) - mdot(L.T, S, L)
        P = dlyap(AKL.T, QKL)

        # Record history
        P_history[i] = P
        K_history[i] = K
        L_history[i] = L
        c_history[i] = np.trace(P)

        # Policy improvement
        # Evaluate Q-function
        Qux = mdot(B.T, P, A)
        Qvx = mdot(C.T, P, A)
        Quu = R + mdot(B.T, P, B)
        Quv = mdot(B.T, P, C)
        Qvu = Quv.T
        Qvv = -S + mdot(C.T, P, C)

        QuvQvvinv = solveb(Quv, Qvv)
        QvuQuuinv = solveb(Qvu, Quu)

        K = -la.solve(Quu-QuvQvvinv.dot(Qvu), Qux-QuvQvvinv.dot(Qvx))
        L = -la.solve(Qvv-QvuQuuinv.dot(Quv), Qvx-QvuQuuinv.dot(Qux))

    return P, K, L, P_history, K_history, L_history, c_history


def value_iteration(A, B, C, Q, R, S, P0=None, num_iterations=100):
    # Value iteration
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
        Qux = mdot(B.T, P, A)
        Qvx = mdot(C.T, P, A)
        Quu = R + mdot(B.T, P, B)
        Quv = mdot(B.T, P, C)
        Qvu = Quv.T
        Qvv = -S + mdot(C.T, P, C)
        QxuQxv = np.vstack([Qux, Qvx])
        QuuQuvQvuQvv = np.block([[Quu, Quv], [Qvu, Qvv]])
        P = Q + mdot(A.T, P, A) - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv))

    # Policy synthesis
    QuvQvvinv = solveb(Quv, Qvv)
    QvuQuuinv = solveb(Qvu, Quu)
    K = -la.solve(Quu-QuvQvvinv.dot(Qvu), Qux-QuvQvvinv.dot(Qvx))
    L = -la.solve(Qvv-QvuQuuinv.dot(Quv), Qvx-QvuQuuinv.dot(Qux))
    return P, K, L, P_history, c_history


def verify_gare(A, B, C, Q, R, S, P, algo_str=None):
    # Verify that the GARE is solved by the solution P

    if algo_str is None:
        algo_str = ''

    Qux = mdot(B.T, P, A)
    Qvx = mdot(C.T, P, A)
    Quu = R + mdot(B.T, P, B)
    Quv = mdot(B.T, P, C)
    Qvu = Quv.T
    Qvv = -S + mdot(C.T, P, C)

    QxuQxv = np.vstack([Qux, Qvx])
    QuuQuvQvuQvv = np.block([[Quu, Quv], [Qvu, Qvv]])

    print(algo_str)
    print('-'*len(algo_str))
    print('Left-hand side of the GARE')
    print(P)
    print('')
    print('Right-hand side of the GARE')
    print(Q + mdot(A.T, P, A) - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv)))
    print('\n')
    return


if __name__ == "__main__":
    # Problem data
    A = np.array([[0.7, 0.2, 0],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.4, 0.3]])
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.2, 0.6]])
    C = np.array([[1.0, 0.3],
                  [0.4, 1.0],
                  [0.6, 0.4]])

    # npr.seed(6)
    # A = npr.randn(5, 5).round(1)
    # A = A*(0.9/specrad(A))
    # B = npr.rand(5, 3).round(1)
    # C = npr.rand(5, 2).round(1)

    n = A.shape[1]
    m = B.shape[1]
    p = C.shape[1]

    Q = np.eye(n)
    R = np.eye(m)
    S = 3*np.eye(p)

    # Initial gains
    K0 = np.zeros([m, n])
    L0 = np.zeros([p, n])

    Pare, Kare = dare_gain(A, B, Q, R)
    K0 = Kare

    AKL0 = A + B.dot(K0) + C.dot(L0)
    QKL0 = Q + mdot(K0.T, R, K0) - mdot(L0.T, S, L0)
    P0 = dlyap(AKL0.T, QKL0)

    # Settings
    num_iterations = 20
    t_history = np.arange(num_iterations)+1

    # Policy iteration
    P_pi, K_pi, L_pi, P_history_pi, K_history_pi, L_history_pi, c_history_pi = policy_iteration(A, B, C, Q, R, S, K0, L0, num_iterations)
    verify_gare(A, B, C, Q, R, S, P_pi, algo_str='Policy iteration')

    # Value iteration
    # Start value iteration at the same initial P as from policy iteration
    P_vi, K_vi, L_vi, P_history_vi, c_history_vi = value_iteration(A, B, C, Q, R, S, P0, num_iterations)
    verify_gare(A, B, C, Q, R, S, P_vi, algo_str='Value iteration')

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