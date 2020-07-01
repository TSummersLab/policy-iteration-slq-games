import numpy as np
import numpy.random as npr
import numpy.linalg as la
from scipy import signal

from matrixmath import specrad


def gen_ex2_ABC():
    A = np.array([[0.7, 0.2],
                  [0.0, 0.5]])
    B = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    C = np.array([[1.0, 0.0],
                  [0.2, 0.7]])
    return A, B, C


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


def gen_rand_ABC(n=4, m=3, p=2, rho=None, seed=1):
    npr.seed(seed)
    if rho is None:
        rho = 0.9
    A = npr.randn(n, n).round(1)
    A = A*(rho/specrad(A))
    B = npr.rand(n, m).round(1)
    C = npr.rand(n, p).round(1)
    return A, B, C


def gen_rand_problem_data(n=4, m=3, p=2, rho=None, seed=1):
    npr.seed(seed)

    A, B, C = gen_rand_ABC(n, m, p, rho, seed)

    q = np.copy(n)
    r = np.copy(m)
    s = np.copy(p)

    Ai = npr.randn(q, n, n).round(1)
    Bj = npr.randn(r, n, m).round(1)
    Ck = npr.randn(s, n, p).round(1)
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


def gen_inv_pendulum():
    m = 1
    dt = 0.1

    A = np.array([[1,    dt],
                  [m*dt,  1]])
    B = np.array([[0],[dt]])
    C = 0.1*np.array([[0.1],[0.1]])

    n = 2
    m = 1
    p = 1

    q = np.copy(n)
    r = np.copy(m)
    s = np.copy(p)

    Ai = np.array([[0,0],
                  [1,0]])
    # npr.randn(q, n, n).round(1)
    Bj = npr.randn(r, n, m).round(1)
    Ck = npr.randn(s, n, p).round(1)
    # Ai = Ai / la.norm(Ai, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    Bj = Bj / la.norm(Bj, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    Ck = Ck / la.norm(Ck, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    varAi = (10**(-3))*np.ones(q)
    varBj = (10**(-3))*np.ones(r)
    varCk = (10**(-3))*np.ones(s)

    Q = np.eye(n)
    R = np.eye(m)
    S = np.eye(p)

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    problem_data_values = [A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S]
    problem_data = dict(zip(problem_data_keys, problem_data_values))
    # print(data_files)
    return problem_data


def gen_inv_pendulum_cart(): #Refer http://www2.ensc.sfu.ca/people/faculty/saif/ctm/examples/pend/digINVSS.html
    M = 0.5     #Mass of Cart - kg
    m = 0.2     #Mass of Pendulum - kg
    b = 0.1     #Friction of cart - N/(m sec)
    l = 0.3     #Pendulum COM from cart length - m
    I = 0.006   #Pendulum inertia - kg m^2
    g = 9.81    #Accln due to gravity - m/(sec^2)
    #Continuous time model: states:x = [cart position, cart velocity, pendulum angle, pendulum angular velocity]'
    # x(t+1) = A_c x(t) + B_c [u(t);v(t)] where B_c = [B|C]
    # y(t) = O_c x(t) + D_c [u(t);v(t)]
    den = I*(M+m) + M*m*(l**2)
    A_c = np.array([[0,1,0,0],
                    [0,(-(I+ m*(l**2))*b)/den,(g*(m*l)**2)/den,0],
                    [0,0,0,1],
                    [0,(-m*l*b)/den,(m*g*l*(M+m))/den,0]])
    B_c = np.array([[0],
                    [(I+m*(l**2))/den],
                    [0],
                    [(m*l)/den]])    #Control actuator
    C_c = np.array([[0],
                    [1],
                    [0],
                    [1]])

    Bin_c = np.concatenate((B_c,C_c), axis=1)
    O_c = np.array([[1,0,0,0],
                    [0,0,1,0]])                         #State Observer matrix
    D_c = np.array([[0,0],                              #Input Observer matrix
                    [0,0]])
    s_c = signal.StateSpace(A_c,Bin_c,O_c,D_c)

    dt = 0.1    #Sampling time interval
    s_d = s_c.to_discrete(dt)

    n = 4
    m = 1
    p = 1
    A = np.array(s_d.A.tolist())
    # print(A)
    B = np.array(s_d.B[:,0].tolist())
    B = B.reshape(n,m)
    C = np.array(s_d.B[:,1].tolist())
    C = C.reshape(n,p)
    # print(B)
    # print(C)
    # B = np.copy(np.array(s_d.B[:][1]).transpose())
    # C = np.copy(np.array(s_d.B[:][2]).transpose())
    # # s = dict(zip(['A','B','C'],[A,B,C]))
    # return B

    q = np.copy(n)
    r = np.copy(m)
    s = np.copy(p)

    Ai = npr.randn(q, n, n).round(1)
    Bj = npr.randn(r, n, m).round(1)
    Ck = npr.randn(s, n, p).round(1)
    Ai = Ai / la.norm(Ai, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    Bj = Bj / la.norm(Bj, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    Ck = Ck / la.norm(Ck, ord=2, axis=(1,2))[:, np.newaxis, np.newaxis]
    varAi = (10**(-3))*np.ones(q)
    varBj = (10**(-3))*np.ones(r)
    varCk = (10**(-1))*np.ones(s)

    Q = np.eye(n)
    R = np.eye(m)
    S = np.eye(p)

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    problem_data_values = [A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S]
    problem_data = dict(zip(problem_data_keys, problem_data_values))
    # print(data_files)
    return problem_data


def gen_double_spring_mass():
    """
    Generate problem data_files for a general double spring-mass system.
    Return problem data_files for:
    1) The full system which has perfect dynamics i.e. the true system
    2) A reduced system which ignores dynamics of the second mass and underestimates the first mass
    """

    # Full dynamics, true system
    # Model parameters
    # Spring constants are for 'negative springs' i.e. spring < 0 is a restoring force
    spring1 = +1.00
    spring2 = -1.00
    spring3 = -0.10

    # Masses
    mass1 = 0.6
    mass2 = 100.0

    # Friction constants are for 'negative friction' i.e. friction < 0 is a retarding force
    friction1 = -0.10
    friction2 = -10.0

    # Actuator strength
    b = 1.0

    # Discretization time duration
    dt = 0.50

    # Continuous time dynamics
    Ac = np.array([[0, 1, 0, 0],
                   [(spring1+spring3)/mass1, friction1/mass1, spring1/mass1, 0],
                   [0, 0, 0, 1],
                   [spring2/mass2, 0, (spring2+spring3)/mass2, friction2/mass2]])
    Bc = np.array([[0], [b], [0], [0]])

    # Discrete time dynamics under forward Euler discretization
    A = np.eye(4) + Ac*dt
    B = Bc*dt
    C = np.zeros([4, 1])

    Ai = np.zeros([1, 4, 4])
    Bj = np.zeros([1, 4, 1])
    Ck = np.zeros([1, 4, 1])

    varAi = np.array([0])
    varBj = np.array([0])
    varCk = np.array([0])

    Q = np.eye(4)
    R = np.eye(1)
    S = np.eye(1)

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    problem_data_values = [A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S]
    problem_data_full = dict(zip(problem_data_keys, problem_data_values))


    # Reduced dynamics, nominal system + uncertainty

    # Model parameters
    # Use same spring constants, mass2, discretization time as in true system
    # Use mis-specified parameters
    spring1 = 0.7*spring1 # smaller than true spring1
    b = 1.2*b # larger than true actuator strength b

    # Continuous time dynamics
    Ac = np.array([[0, 1,],
                   [(spring1+spring3)/mass1, friction1/mass1]])
    Bc = np.array([[0], [b]])
    Cc = np.array([[0], [1]])

    # Discrete time dynamics under forward Euler discretization
    A = np.eye(2) + Ac*dt
    B = Bc*dt
    C = Cc*dt

    Ai = np.array([[[0, 0],
                    [1, 0]]])*dt
    Bj = np.array([[[0], [1]]])*dt
    Ck = np.array([[[0], [1]]])*dt

    varAi = np.array([0.8])
    varBj = np.array([0.4])
    varCk = np.array([0.0])

    Q = np.eye(2)
    R = np.eye(1)
    S = 7*np.eye(1)

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    problem_data_values = [A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S]
    problem_data_reduced = dict(zip(problem_data_keys, problem_data_values))

    return problem_data_full, problem_data_reduced


def example_system_erdos_renyi(n, m, p, diffusion_constant=1.0, leakiness_constant=0.2, time_constant=0.1,
                               leaky=True, seed=None, show_graph=False):
    npr.seed(seed)
    # ER probability
    # crp = 7.0
    # erp = (np.log(n+1)+crp)/(n+1)  # almost surely connected prob=0.999

    mean_degree = 4.0 # should be > 1 for giant component to exist
    erp = mean_degree/(n-1.0)

    n_edges = 0
    # Create random Erdos-Renyi graph
    # Adjacency matrix
    adjacency = np.zeros([n, n])
    for i in range(n):
        for j in range(i+1, n):
            if npr.rand() < erp:
                n_edges += 1
                adjacency[i, j] = npr.randint(low=1, high=4)
                adjacency[j, i] = np.copy(adjacency[i, j])

    # Degree matrix
    degree = np.diag(adjacency.sum(axis=0))
    # Graph Laplacian
    laplacian = degree-adjacency
    # Continuous-time dynamics matrices
    Ac = -laplacian*diffusion_constant


    Bc = np.zeros([n, m])
    Cc = np.zeros([n, p])
    B_idx = np.sort(npr.permutation(n)[0:m])
    C_idx = np.sort(npr.permutation(n)[0:p])

    for i in range(m):
        Bc[B_idx[i], i] = npr.randint(low=1, high=5)/time_constant
    for i in range(p):
        Cc[C_idx[i], i] = npr.randint(low=1, high=5)/time_constant

    if leaky:
        Fc = leakiness_constant*np.eye(n)
        Ac = Ac - Fc

    # Plot
    if show_graph:
        visualize_graph_ring(adjacency, n)

    # Forward Euler discretization
    A = np.eye(n) + Ac*time_constant
    B = Bc*time_constant
    C = Cc*time_constant

    # Multiplicative noises
    varAi = 0.01*npr.randint(low=1, high=5, size=n_edges)*np.ones(n_edges)
    Ai = np.zeros([n_edges, n, n])
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j] > 0:
                Ai[k, i, i] = 1
                Ai[k, j, j] = 1
                Ai[k, i, j] = -1
                Ai[k, j, i] = -1
                k += 1

    varBj = 0.01*npr.randint(low=1, high=5, size=m)*np.ones(m)
    varCk = 0.01*npr.randint(low=1, high=5, size=p)*np.ones(p)
    Bj = np.zeros([m, n, m])
    Ck = np.zeros([p, n, p])

    for i in range(m):
        Bj[i, B_idx[i], i] = 1
    for i in range(p):
        Ck[i, C_idx[i], i] = 1

    Q = np.eye(n)
    R = np.eye(m)
    S = 200*np.eye(p)


    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    problem_data_values = [A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S]
    problem_data = dict(zip(problem_data_keys, problem_data_values))

    return problem_data


def visualize_graph_ring(adj, n):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(4, 4))

    # Scatter plot all the center points
    t = np.arange(0,2*np.pi,2*np.pi/n)
    x = np.cos(t)
    y = np.sin(t)
    plt.scatter(x, y, s=600, alpha=1.0, zorder=21)
    # plt.scatter(x[0],y[0],s=60,alpha=1.0,zorder=110,marker='s') # Highlight the reference node
    # Draw edge lines
    linecolor = (0.1,0.1,0.1)
    lines = []
    linewidths = []

    for i in range(n):
        for j in range(i+1,n):
            if adj[i,j] > 0:
                line = ((x[i],y[i]),(x[j],y[j]))
                lines.append(line)
                linewidths.append(4*adj[i,j])

    linecol = LineCollection(lines,linewidths=linewidths,alpha=0.5,colors=linecolor,zorder=10)
    ax.add_collection(linecol)

    # Plot options
    plt.axis('scaled')
    plt.axis('equal')
    plt.axis('off')
    plt.ion()
    plt.tight_layout()
    plt.show()