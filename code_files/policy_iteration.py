import numpy as np
import numpy.linalg as la
import numpy.random as npr

from ltimult import gdlyap
from matrixmath import mdot, specrad, solveb, dare_gain, is_pos_def, svec2, smat2, kron


def groupdot(A, x):
    """
    Perform dot product over groups of matrices,
    suitable for performing many LTI state transitions in a vectorized fashion
    """
    return np.einsum('...ik,...k', A, x)


def groupquadform(A, x):
    """
    Perform quadratic form over many vectors stacked in matrix x with respect to cost matrix A
    Equivalent to np.array([mdot(x[i].T, A, x[i]) for i in range(x.shape[0])])
    """
    return np.sum(np.dot(x, A)*x, axis=1)


def sample_ABCrand(problem_data):
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk = [problem_data[key] for key in problem_data_keys]
    q, r, s = [M.shape[0] for M in [Ai, Bj, Ck]]
    Arand = A + np.sum([npr.randn()*(varAi[i]**0.5)*Ai[i] for i in range(q)], axis=0)
    Brand = B + np.sum([npr.randn()*(varBj[j]**0.5)*Bj[j] for j in range(r)], axis=0)
    Crand = C + np.sum([npr.randn()*(varCk[k]**0.5)*Ck[k] for k in range(s)], axis=0)
    return Arand, Brand, Crand


def sample_ABCrand_multi(problem_data, nt, nr):
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk = [problem_data[key] for key in problem_data_keys]
    q, r, s = [M.shape[0] for M in [Ai, Bj, Ck]]

    # Sample scalar random variables
    a = npr.randn(nt, nr, q)*(varAi**0.5)
    b = npr.randn(nt, nr, r)*(varBj**0.5)
    c = npr.randn(nt, nr, s)*(varCk**0.5)

    # Generate Arand etc. such that Arand[i, j] == A + np.sum([a[i, j][k]*Ai[k] for k in range(q)], axis=0)
    Arand_all = A + np.moveaxis(np.einsum('ijk,k...', a, Ai), [-2, -1], [0, 1])
    Brand_all = B + np.moveaxis(np.einsum('ijk,k...', b, Bj), [-2, -1], [0, 1])
    Crand_all = C + np.moveaxis(np.einsum('ijk,k...', c, Ck), [-2, -1], [0, 1])

    return Arand_all, Brand_all, Crand_all


def rollout(problem_data, K, L, sim_options):
    """Simulate closed-loop state response"""
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n, m, p = [M.shape[1] for M in [A, B, C]]

    sim_options_keys = ['xstd', 'ustd', 'vstd', 'wstd', 'nt', 'nr', 'group_option']
    xstd, ustd, vstd, wstd, nt, nr, group_option = [sim_options[key] for key in sim_options_keys]

    qfun_estimator = sim_options['qfun_estimator']

    if qfun_estimator == 'direct':
        # First step
        # Sample initial states, defender control inputs, and attacker control inputs
        x0, u0, v0 = [npr.randn(nr, dim)*std for dim, std in zip([n, m, p], [xstd, ustd, vstd])]

        Qval = np.zeros(nr)

        if group_option == 'single':
            # Iterate over rollouts
            for k in range(nr):
                x0_k, u0_k, v0_k = [var[k] for var in [x0, u0, v0]]

                # Initialize
                x = np.copy(x0_k)

                # Iterate over timesteps
                for i in range(nt):
                    # Compute controls
                    if i == 0:
                        u, v = np.copy(u0_k), np.copy(v0_k)
                    else:
                        u, v = np.dot(K, x), np.dot(L, x)

                    # Accumulate cost
                    Qval[k] += mdot(x.T, Q, x) + mdot(u.T, R, u) - mdot(v.T, S, v)

                    # Randomly sample state transition matrices using multiplicative noise
                    Arand, Brand, Crand = sample_ABCrand(problem_data)

                    # Additive noise
                    w = npr.randn(n)*wstd

                    # Transition the state using multiplicative and additive noise
                    x = np.dot(Arand, x) + np.dot(Brand, u) + np.dot(Crand, v) + w

        elif group_option == 'group':
            # Randomly sample state transition matrices using multiplicative noise
            Arand_all, Brand_all, Crand_all = sample_ABCrand_multi(problem_data, nt, nr)

            # Randomly sample additive noise
            w_all = npr.randn(nt, nr, n)*wstd

            # Initialize
            x = np.copy(x0)

            # Iterate over timesteps
            for i in range(nt):
                # Compute controls
                if i == 0:
                    u = np.copy(u0)
                    v = np.copy(v0)
                else:
                    u = groupdot(K, x)
                    v = groupdot(L, x)

                # Accumulate cost
                Qval += groupquadform(Q, x) + groupquadform(R, u) - groupquadform(S, v)

                # Look up stochastic dynamics and additive noise
                Arand, Brand, Crand = Arand_all[i], Brand_all[i], Crand_all[i]
                w = w_all[i]

                # Transition the state using multiplicative and additive noise
                x = groupdot(Arand, x) + groupdot(Brand, u) + groupdot(Crand, v) + w

        return x0, u0, v0, Qval

    elif qfun_estimator == 'lsadp' or qfun_estimator == 'lstdq':
        # Sample initial states, defender control inputs, and attacker control inputs
        x0 = xstd*npr.randn(nr, n)
        u_explore_hist = ustd*npr.randn(nr, nt, m)
        v_explore_hist = vstd*npr.randn(nr, nt, p)

        x_hist = np.zeros([nr, nt, n])
        u_hist = np.zeros([nr, nt, m])
        v_hist = np.zeros([nr, nt, p])
        c_hist = np.zeros([nr, nt])

        if group_option == 'single':
            # Iterate over rollouts
            for k in range(nr):
                # Initialize
                x = np.copy(x0[k])

                # Iterate over timesteps
                for i in range(nt):
                    # Compute controls
                    u = np.dot(K, x) + u_explore_hist[k, i]
                    v = np.dot(L, x) + v_explore_hist[k, i]

                    # Compute cost
                    c = mdot(x.T, Q, x) + mdot(u.T, R, u) - mdot(v.T, S, v)

                    # Record history
                    x_hist[k, i] = x
                    u_hist[k, i] = u
                    v_hist[k, i] = v
                    c_hist[k, i] = c

                    # Randomly sample state transition matrices using multiplicative noise
                    Arand, Brand, Crand = sample_ABCrand(problem_data)

                    # Additive noise
                    w = npr.randn(n)*wstd

                    # Transition the state using multiplicative and additive noise
                    x = np.dot(Arand, x) + np.dot(Brand, u) + np.dot(Crand, v) + w

        elif group_option == 'group':
            # Randomly sample state transition matrices using multiplicative noise
            Arand_all, Brand_all, Crand_all = sample_ABCrand_multi(problem_data, nt, nr)

            # Randomly sample additive noise
            w_all = npr.randn(nt, nr, n)*wstd

            # Initialize
            x = np.copy(x0)

            # Iterate over timesteps
            for i in range(nt):
                # Compute controls
                u = groupdot(K, x) + u_explore_hist[:, i]
                v = groupdot(L, x) + v_explore_hist[:, i]

                # Compute cost
                c = groupquadform(Q, x) + groupquadform(R, u) - groupquadform(S, v)

                # Record history
                x_hist[:, i] = x
                u_hist[:, i] = u
                v_hist[:, i] = v
                c_hist[:, i] = c

                # Look up stochastic dynamics and additive noise
                Arand, Brand, Crand = Arand_all[i], Brand_all[i], Crand_all[i]
                w = w_all[i]

                # Transition the state using multiplicative and additive noise
                x = groupdot(Arand, x) + groupdot(Brand, u) + groupdot(Crand, v) + w

        return x_hist, u_hist, v_hist, c_hist


def qfun(problem_data, problem_data_known=None, P=None, K=None, L=None, sim_options=None, output_format=None):
    """Compute or estimate Q-function matrix"""
    if problem_data_known is None:
        problem_data_known = True
    if output_format is None:
        output_format = 'list'
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n, m, p = [M.shape[1] for M in [A, B, C]]
    q, r, s = [M.shape[0] for M in [Ai, Bj, Ck]]

    if P is None:
        P = gdlyap(problem_data, K, L)

    if problem_data_known:
        APAi = np.sum([varAi[i]*mdot(Ai[i].T, P, Ai[i]) for i in range(q)], axis=0)
        BPBj = np.sum([varBj[j]*mdot(Bj[j].T, P, Bj[j]) for j in range(r)], axis=0)
        CPCk = np.sum([varCk[k]*mdot(Ck[k].T, P, Ck[k]) for k in range(s)], axis=0)

        Qxx = Q + mdot(A.T, P, A) + APAi
        Quu = R + mdot(B.T, P, B) + BPBj
        Qvv = -S + mdot(C.T, P, C) + CPCk
        Qux = mdot(B.T, P, A)
        Qvx = mdot(C.T, P, A)
        Qvu = mdot(C.T, P, B)
    else:
        nr = sim_options['nr']
        nt = sim_options['nt']
        qfun_estimator = sim_options['qfun_estimator']

        if qfun_estimator == 'direct':
            # Simulation data_files collection
            x0, u0, v0, Qval = rollout(problem_data, K, L, sim_options)

            # Dimensions
            Qpart_shapes = [[n, n], [m, m], [p, p], [m, n], [p, n], [p, m]]
            Qvec_part_lengths = [np.prod(shape) for shape in Qpart_shapes]

            # Least squares estimation
            xuv_data = np.zeros([nr, np.sum(Qvec_part_lengths)])
            for i in range(nr):
                x = x0[i]
                u = u0[i]
                v = v0[i]
                xuv_data[i] = np.hstack([kron(x.T, x.T), kron(u.T, u.T), kron(v.T, v.T),
                                         2*kron(x.T, u.T), 2*kron(x.T, v.T), 2*kron(u.T, v.T)])

            # Solve the least squares problem
            Qvec = la.lstsq(xuv_data, Qval, rcond=None)[0]

            # Split and reshape the solution vector into the appropriate matrices
            idxi = [0]
            Qvec_parts = []
            Q_parts = []
            for i, part_length in enumerate(Qvec_part_lengths):
                idxi.append(idxi[i] + part_length)
                Qvec_parts.append(Qvec[idxi[i]:idxi[i+1]])
                Q_parts.append(np.reshape(Qvec_parts[i], Qpart_shapes[i]))

            Qxx, Quu, Qvv, Qux, Qvx, Qvu = Q_parts

        elif qfun_estimator == 'lsadp':
            # Simulation data_files collection
            x_hist, u_hist, v_hist, c_hist = rollout(problem_data, K, L, sim_options)

            # Form the data_files matrices
            ns = nr*(nt-1)
            nz = int(((n+m+p+1)*(n+m+p))/2)
            mu_hist = np.zeros([nr, nt, nz])
            nu_hist = np.zeros([nr, nt, nz])

            def phi(x):
                return svec2(np.outer(x, x))
            
            for i in range(nr):
                for j in range(nt):
                    z = np.concatenate([x_hist[i, j], u_hist[i, j], v_hist[i, j]])
                    w = np.concatenate([x_hist[i, j], np.dot(K, x_hist[i, j]), np.dot(L, x_hist[i, j])])
                    mu_hist[i, j] = phi(z)
                    nu_hist[i, j] = phi(w)
            Y = np.zeros(ns)
            Z = np.zeros([ns, nz])
            for i in range(nr):
                lwr = i*(nt-1)
                upr = (i+1)*(nt-1)
                Y[lwr:upr] = c_hist[i, 0:-1]
                Z[lwr:upr] = mu_hist[i, 0:-1] - nu_hist[i, 1:]

            # Solve the least squares problem
            # H_svec = la.lstsq(Z, Y, rcond=None)[0]
            # H = smat(H_svec)
            H_svec2 = la.lstsq(Z, Y, rcond=None)[0]
            H = smat2(H_svec2)

            Qxx = H[0:n, 0:n]
            Quu = H[n:n+m, n:n+m]
            Qvv = H[n+m:, n+m:]
            Qux = H[n:n+m, 0:n]
            Qvx = H[n+m:, 0:n]
            Qvu = H[n+m:, n:n+m]

        elif qfun_estimator == 'lstdq':
            # Simulation data_files collection
            x_hist, u_hist, v_hist, c_hist = rollout(problem_data, K, L, sim_options)

            # Form the data_files matrices
            nz = int(((n+m+p+1)*(n+m+p))/2)
            mu_hist = np.zeros([nr, nt, nz])
            nu_hist = np.zeros([nr, nt, nz])

            def phi(x):
                return svec2(np.outer(x, x))

            for i in range(nr):
                for j in range(nt):
                    z = np.concatenate([x_hist[i, j], u_hist[i, j], v_hist[i, j]])
                    w = np.concatenate([x_hist[i, j], np.dot(K, x_hist[i, j]), np.dot(L, x_hist[i, j])])
                    mu_hist[i, j] = phi(z)
                    nu_hist[i, j] = phi(w)

            Y = np.zeros(nr*nz)
            Z = np.zeros([nr*nz, nz])
            for i in range(nr):
                lwr = i*nz
                upr = (i+1)*nz
                for j in range(nt-1):
                    Y[lwr:upr] += mu_hist[i, j]*c_hist[i, j]
                    Z[lwr:upr] += np.outer(mu_hist[i, j], mu_hist[i, j] - nu_hist[i, j+1])

            H_svec2 = la.lstsq(Z, Y, rcond=None)[0]
            H = smat2(H_svec2)

            Qxx = H[0:n, 0:n]
            Quu = H[n:n+m, n:n+m]
            Qvv = H[n+m:, n+m:]
            Qux = H[n:n+m, 0:n]
            Qvx = H[n+m:, 0:n]
            Qvu = H[n+m:, n:n+m]

    if output_format == 'list':
        outputs = Qxx, Quu, Qvv, Qux, Qvx, Qvu
    elif output_format == 'matrix':
        outputs = np.block([[Qxx, Qux.T, Qvx.T],
                            [Qux, Quu, Qvu.T],
                            [Qvx, Qvu, Qvv]])
        # ABC = np.hstack([A, B, C])
        # X = sla.block_diag(Q, R, -S)
        # Y = mdot(ABC.T, P, ABC)
        # Z = sla.block_diag(APAi, BPBj, CPCk)
        # outputs = X + Y + Z
    return outputs


def policy_iteration(problem_data, problem_data_known, K0, L0, sim_options=None, num_iterations=100,
                     print_iterates=True):
    """Policy iteration"""
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n, m, p = [M.shape[1] for M in [A, B, C]]
    K, L = np.copy(K0), np.copy(L0)

    # Check initial policies are stabilizing
    if specrad(A + B.dot(K0) + C.dot(L0)) > 1:
        raise Exception("Initial policies are not stabilizing!")

    P_history, K_history, L_history = [np.zeros([num_iterations, dim, n]) for dim in [n, m, p]]
    H_history = np.zeros([num_iterations, n+m+p, n+m+p])
    c_history = np.zeros(num_iterations)

    print('Policy iteration')
    for i in range(num_iterations):
        # Record history
        K_history[i] = K
        L_history[i] = L

        # Policy evaluation
        P = gdlyap(problem_data, K, L)
        Qxx, Quu, Qvv, Qux, Qvx, Qvu = qfun(problem_data, problem_data_known, P, K, L, sim_options)
        QuvQvvinv = solveb(Qvu.T, Qvv)
        QvuQuuinv = solveb(Qvu, Quu)
        H = np.block([[Qxx, Qux.T, Qvx.T],
                      [Qux, Quu, Qvu.T],
                      [Qvx, Qvu, Qvv]])

        # Policy improvement
        K = -la.solve(Quu-QuvQvvinv.dot(Qvu), Qux-QuvQvvinv.dot(Qvx))
        L = -la.solve(Qvv-QvuQuuinv.dot(Qvu.T), Qvx-QvuQuuinv.dot(Qux))

        # Record history
        P_history[i] = P
        H_history[i] = H
        c_history[i] = np.trace(P)
        if print_iterates:
            print('iteration %3d / %3d' % (i+1, num_iterations))
            print(P)

    print('')
    return P, K, L, H, P_history, K_history, L_history, c_history, H_history


def value_iteration(problem_data, P0=None, num_iterations=100):
    """Value iteration"""
    # Only the case of known dynamics is supported
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n, m, p = [M.shape[1] for M in [A, B, C]]

    if P0 is None:
        P0 = np.copy(Q)
    P = np.copy(P0)

    P_history = np.zeros([num_iterations, n, n])
    c_history = np.zeros(num_iterations)

    for i in range(num_iterations):
        # Record history
        P_history[i] = P
        c_history[i] = np.trace(P)

        # Value improvement
        Qxx, Quu, Qvv, Qux, Qvx, Qvu = qfun(problem_data, P=P)
        QxuQxv = np.vstack([Qux, Qvx])
        QuuQuvQvuQvv = np.block([[Quu, Qvu.T],
                                 [Qvu, Qvv]])
        P = Qxx - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv))

    # Policy synthesis
    QuvQvvinv = solveb(Qvu.T, Qvv)
    QvuQuuinv = solveb(Qvu, Quu)
    K = -la.solve(Quu-QuvQvvinv.dot(Qvu), Qux-QuvQvvinv.dot(Qvx))
    L = -la.solve(Qvv-QvuQuuinv.dot(Qvu.T), Qvx-QvuQuuinv.dot(Qux))

    return P, K, L, P_history, c_history


def verify_gare(problem_data, P, algo_str=None):
    """Verify that the GARE is solved by the solution P"""
    if algo_str is None:
        algo_str = ''

    Qxx, Quu, Qvv, Qux, Qvx, Qvu = qfun(problem_data, P=P)
    QxuQxv = np.vstack([Qux, Qvx])
    QuuQuvQvuQvv = np.block([[Quu, Qvu.T],
                             [Qvu, Qvv]])

    print(algo_str)
    print('-'*len(algo_str))
    LHS = P
    RHS = Qxx - np.dot(QxuQxv.T, la.solve(QuuQuvQvuQvv, QxuQxv))
    print(' Left-hand side of the GARE: Positive definite = %s' % is_pos_def(LHS))
    print(LHS)
    print('')
    print('Right-hand side of the GARE: Positive definite = %s' % is_pos_def(RHS))
    print(RHS)
    print('')
    print('Difference')
    print(LHS-RHS)
    print('\n')
    return


def get_initial_gains(problem_data, initial_gain_method=None):
    """Get initial gains"""
    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n, m, p = [M.shape[1] for M in [A, B, C]]

    if initial_gain_method is None:
        initial_gain_method = 'zero'

    if initial_gain_method == 'zero':
        K0 = np.zeros([m, n])
        L0 = np.zeros([p, n])
    elif initial_gain_method == 'dare':
        Pare, Kare = dare_gain(A, B, Q, R)
        K0 = Kare
        L0 = np.zeros([p, n])

    return K0, L0


def compare_qfun(problem_data, sim_options, K, L):
    """Compare single Q-function evaluation in cases of known and unknown dynamics"""
    Qxx, Quu, Qvv, Qux, Qvx, Qvu = qfun(problem_data, problem_data_known=False, K=K, L=L, sim_options=sim_options)
    Q_parts = [Qxx, Quu, Qvv, Qux, Qvx, Qvu]

    # Calculate error
    Q_part_strings = ['Qxx', 'Quu', 'Qvv', 'Qux', 'Qvx', 'Qvu']

    Q_parts_true = qfun(problem_data, problem_data_known=True, K=K, L=L)
    for Q_part, Q_part_true, Q_part_string in zip(Q_parts, Q_parts_true, Q_part_strings):
        print(Q_part_string)
        print('true')
        print(Q_part_true)
        print('est')
        print(Q_part)
        print('diff')
        print(Q_part-Q_part_true)
        print('\n')
    return
