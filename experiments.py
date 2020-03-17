import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt
import control as ctrl
import copy

from problem_data_gen import gen_double_spring_mass, gen_double_spring_mass2, example_system_erdos_renyi

from policy_iteration import policy_iteration, value_iteration, get_initial_gains, get_sim_options, verify_gare

from ltimult_lqm import gdlyap

import sys
sys.path.insert(0,'../utility')
from matrixmath import mdot, specrad, dlyap, dare_gain, is_pos_def, kron, vec


def model_check(A, B, Q, R, K, op = True, model_type=None):
    """Check stability of system (A,B) with costs Q, R under the feedback gain K """
    # A, B, C = sample_ABCrand(problem_data_true)
    Qbar = Q + mdot(K.T, R, K)
    Abar = A + mdot(B,K)
    # print("Feedback gains test ")
    if model_type != None:
        print(model_type)
    if is_pos_def(dlyap(Abar,Qbar))==1: #if pos_def => gain stabilizes
        if op == True:
            print("Policy Iteration on model stabilizes the system - D-Lyap is Positive Definite.\n")
        ret = True
    else:
        if op == True:
            print("Policy Iteration on model does not stabilize the system - D-Lyap is NOT Positive Definite.\n")
        ret = False
    return ret


def mtpl(A, B, K_pi, Kn_pi, Km_pi, Q, R, init = 1, multiplier_accuracy = 10**(-8), multiplier_bound = 10**(5), iteration_steps = 10**10):
    """ Calculate a multiplying/scaling factor on the dynamics such that the system is not stabilized by gains from nominal game or LQR with multiplicative noise
     and only by gains from game with multiplicative noise """
    At_base = copy.deepcopy(A)
    Bt = copy.deepcopy(B)
    mpl = copy.deepcopy(init)

    #standard controllability check
    if np.linalg.matrix_rank(ctrl.ctrb(At_base,Bt)) < len(At_base):
        print("Given A,B matrix pair for the true system is not controllable.")
        return 0

    iter_check = 0
    check_gains = 0
    mpl_his = 0

    while check_gains == 0:
        iter_check += 1
        At = copy.deepcopy(At_base)
        At *= mpl
        c_gm = model_check(At, Bt, K_pi, np.eye(n,n), np.eye(m,m), False)
        c_n = model_check(At, Bt, Kn_pi, np.eye(n,n), np.eye(m,m), False)
        c_m = model_check(At, Bt, Km_pi, np.eye(n,n), np.eye(m,m), False)
        # print(c_gm,":",c_n,":",c_m,":",mpl,"\n")
        if c_gm == 0 and c_n + c_m >0:
            check_gains = 2
            break
        elif c_gm ==1 and c_n + c_m ==0:
            check_gains = 1
            break
        elif abs(mpl-mpl_his)<multiplier_accuracy:
            check_gains = 3
            break
        elif mpl>multiplier_bound:
            check_gains = 4
            break
        elif iter_check>iteration_steps:
            check_gains = 5
            break
        else:
            mpl_his = mpl
            if c_gm == 1 and c_n + c_m > 0:
                mpl = mpl*3/2
            elif c_gm + c_n + c_m == 0:
                mpl = mpl/2

    print("\tMultiplier:",mpl)
    # print("check_gains:",check_gains)
    if check_gains == 1:
        print("\tGain of game w/ noise stabilizes while other two gains fail")
    elif check_gains == 2:
        print("\tGains of nominal game w/o m-noise or lqr w/ m-noise stabilize the true system, game w/ m-noise fails")
    elif check_gains ==3:
        print("\tFailed check. Multiplier out of threshold or too many iterations.")
    return mpl



def robust_stabilization():
    seed = 1
    npr.seed(seed)

    problem_data_true, problem_data = gen_double_spring_mass()

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    n, m, p = [M.shape[1] for M in [A, B, C]]
    q, r, s = [M.shape[0] for M in [Ai, Bj, Ck]]

    # Synthesize controllers using various uncertainty modeling terms

    # Modify problem data
    # LQR w/ game adversary
    # Setting varAi, varBj, varCk = 0 => no multiplicative noise on the game
    problem_data_model_n = copy.deepcopy(problem_data)
    problem_data_model_n['varAi'] *= 0
    problem_data_model_n['varBj'] *= 0
    problem_data_model_n['varCk'] *= 0

    # LQR w/ multiplicative noise
    # Setting C = 0 and varCk = 0 => no game adversary
    problem_data_model_m = copy.deepcopy(problem_data)
    problem_data_model_m['C'] *= 0
    problem_data_model_m['varCk'] *= 0

    # Simulation options
    sim_options = get_sim_options()
    num_iterations = 50
    problem_data_known = True

    # Policy iteration on LQR w/ game adversary and multiplicative noise
    K0, L0 = get_initial_gains(problem_data, initial_gain_method='dare')
    print("LQR w/ game adversary and multiplicative noise")
    P_pi, K_pi, L_pi, H_pi, P_history_pi, K_history_pi, L_history_pi, c_history_pi, H_history_pi = policy_iteration(problem_data, problem_data_known, K0, L0, sim_options, num_iterations)
    verify_gare(problem_data, P_pi, algo_str='Policy iteration - Game w/ Multiplicative noise')

    # Check concavity condition
    Qvv_pi = -S + mdot(C.T, P_pi, C) + np.sum([varCk[k]*mdot(Ck[k].T, P_pi, Ck[k]) for k in range(s)], axis=0)
    if not is_pos_def(-Qvv_pi):
        raise Exception('Problem fails the concavity condition, adjust adversary strength')

    # Check positive definiteness condition
    QKL_pi = Q + mdot(K_pi.T, R, K_pi) - mdot(L_pi.T, S, L_pi)
    if not is_pos_def(QKL_pi):
        raise Exception('Problem fails the positive definiteness condition, adjust adversary strength')
    print(QKL_pi)

    # Policy Iteration on LQR w/ game adversary
    K0n, L0n = get_initial_gains(problem_data_model_n, initial_gain_method='dare')
    print("LQR w/ game adversary")
    Pn_pi, Kn_pi, Ln_pi, Hn_pi, Pn_history_pi, Kn_history_pi, Ln_history_pi, cn_history_pi, Hn_history_pi = policy_iteration(problem_data_model_n, problem_data_known, K0n, L0n, sim_options, num_iterations)
    verify_gare(problem_data_model_n, Pn_pi, algo_str='Policy iteration - Game w/o Multiplicative noise')

    # Policy Iteration on LQR w/ multiplicative noise
    K0m, L0m = get_initial_gains(problem_data_model_m, initial_gain_method='dare')
    print("LQR w/ multiplicative noise")
    Pm_pi, Km_pi, Lm_pi, Hm_pi, Pm_history_pi, Km_history_pi, Lm_history_pi, cm_history_pi, Hm_history_pi = policy_iteration(problem_data_model_m, problem_data_known, K0m, L0m, sim_options, num_iterations)
    verify_gare(problem_data_model_m, Pm_pi, algo_str='Policy iteration - LQR w/ Multiplicative noise')

    # LQR on true system
    A_true, B_true, Q_true, R_true = [problem_data_true[key] for key in ['A', 'B', 'Q', 'R']]
    n_true, m_true = [M.shape[1] for M in [A_true, B_true]]
    Pare_true, Kare_true = dare_gain(A_true, B_true, Q_true, R_true)

    # LQR on nominal system, no explicit robust control design
    Pce, Kce = dare_gain(A, B, Q, R)

    # Check if synthesized controllers stabilize the true system
    K_pi_true = np.hstack([K_pi, np.zeros([m, n])])
    Kn_pi_true = np.hstack([Kn_pi, np.zeros([m, n])])
    Km_pi_true = np.hstack([Km_pi, np.zeros([m, n])])
    Kce_true = np.hstack([Kce, np.zeros([m, n])])
    Kol_true = np.zeros_like(Kce_true)

    control_method_strings = ['open-loop     ', 'cert equiv    ', 'noise         ', 'game          ', 'noise + game  ', 'optimal       ']
    K_list = [Kol_true, Kce_true, Km_pi_true, Kn_pi_true, K_pi_true, Kare_true]
    AK_list = [A_true + np.dot(B_true, K) for K in K_list]
    QK_list = [Q_true + mdot(K.T, R_true, K) for K in K_list]
    specrad_list = [specrad(AK) for AK in AK_list]
    cost_list = [np.trace(dlyap(AK.T, QK)) if sr < 1 else np.inf for AK, QK, sr in zip(AK_list, QK_list, specrad_list)]

    print('method       |  specrad  |   cost   |  gains')
    for control_method_string, sr, cost, K in zip(control_method_strings, specrad_list, cost_list, K_list):
        print('%s  %f %8s    %s' % (control_method_string, sr, '%10.2f' % cost, K))


def robust_stabilization2():
    seed = 1
    npr.seed(seed)

    problem_data_true, problem_data = gen_double_spring_mass2()

    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]

    n, m, p = [M.shape[1] for M in [A, B, C]]
    q, r, s = [M.shape[0] for M in [Ai, Bj, Ck]]

    # Synthesize controllers using various uncertainty modeling terms

    # Modify problem data
    # LQR w/ game adversary
    # Setting varAi, varBj, varCk = 0 => no multiplicative noise on the game
    problem_data_model_n = copy.deepcopy(problem_data)
    problem_data_model_n['varAi'] *= 0
    problem_data_model_n['varBj'] *= 0
    problem_data_model_n['varCk'] *= 0

    # LQR w/ multiplicative noise
    # Setting C = 0 and varCk = 0 => no game adversary
    problem_data_model_m = copy.deepcopy(problem_data)
    problem_data_model_m['C'] *= 0
    problem_data_model_m['varCk'] *= 0

    # Simulation options
    sim_options = get_sim_options()
    num_iterations = 50
    problem_data_known = True

    # Policy iteration on LQR w/ game adversary and multiplicative noise
    K0, L0 = get_initial_gains(problem_data, initial_gain_method='dare')
    print("LQR w/ game adversary and multiplicative noise")
    P_pi, K_pi, L_pi, H_pi, P_history_pi, K_history_pi, L_history_pi, c_history_pi, H_history_pi = policy_iteration(problem_data, problem_data_known, K0, L0, sim_options, num_iterations)
    verify_gare(problem_data, P_pi, algo_str='Policy iteration - Game w/ Multiplicative noise')

    # Check concavity condition
    Qvv_pi = -S + mdot(C.T, P_pi, C) + np.sum([varCk[k]*mdot(Ck[k].T, P_pi, Ck[k]) for k in range(s)], axis=0)
    if not is_pos_def(-Qvv_pi):
        raise Exception('Problem fails the concavity condition, adjust adversary strength')

    # Check positive definiteness condition
    QKL_pi = Q + mdot(K_pi.T, R, K_pi) - mdot(L_pi.T, S, L_pi)
    if not is_pos_def(QKL_pi):
        raise Exception('Problem fails the positive definiteness condition, adjust adversary strength')
    print(QKL_pi)

    # Policy Iteration on LQR w/ game adversary
    K0n, L0n = get_initial_gains(problem_data_model_n, initial_gain_method='dare')
    print("LQR w/ game adversary")
    Pn_pi, Kn_pi, Ln_pi, Hn_pi, Pn_history_pi, Kn_history_pi, Ln_history_pi, cn_history_pi, Hn_history_pi = policy_iteration(problem_data_model_n, problem_data_known, K0n, L0n, sim_options, num_iterations)
    verify_gare(problem_data_model_n, Pn_pi, algo_str='Policy iteration - Game w/o Multiplicative noise')

    # Policy Iteration on LQR w/ multiplicative noise
    K0m, L0m = get_initial_gains(problem_data_model_m, initial_gain_method='dare')
    print("LQR w/ multiplicative noise")
    Pm_pi, Km_pi, Lm_pi, Hm_pi, Pm_history_pi, Km_history_pi, Lm_history_pi, cm_history_pi, Hm_history_pi = policy_iteration(problem_data_model_m, problem_data_known, K0m, L0m, sim_options, num_iterations)
    verify_gare(problem_data_model_m, Pm_pi, algo_str='Policy iteration - LQR w/ Multiplicative noise')

    # LQR on true system
    A_true, B_true, Q_true, R_true = [problem_data_true[key] for key in ['A', 'B', 'Q', 'R']]
    n_true, m_true = [M.shape[1] for M in [A_true, B_true]]
    Pare_true, Kare_true = dare_gain(A_true, B_true, Q_true, R_true)

    # LQR on nominal system, no explicit robust control design
    Pce, Kce = dare_gain(A, B, Q, R)

    # Check if synthesized controllers stabilize the true system
    K_pi_true = np.hstack([K_pi, np.zeros([m, n])])
    Kn_pi_true = np.hstack([Kn_pi, np.zeros([m, n])])
    Km_pi_true = np.hstack([Km_pi, np.zeros([m, n])])
    Kce_true = np.hstack([Kce, np.zeros([m, n])])
    Kol_true = np.zeros_like(Kce_true)

    control_method_strings = ['open-loop     ', 'cert equiv    ', 'noise         ', 'game          ', 'noise + game  ', 'optimal       ']
    K_list = [Kol_true, Kce_true, Km_pi_true, Kn_pi_true, K_pi_true, Kare_true]
    AK_list = [A_true + np.dot(B_true, K) for K in K_list]
    QK_list = [Q_true + mdot(K.T, R_true, K) for K in K_list]
    specrad_list = [specrad(AK) for AK in AK_list]
    cost_list = [np.trace(dlyap(AK.T, QK)) if sr < 1 else np.inf for AK, QK, sr in zip(AK_list, QK_list, specrad_list)]


    def set_numpy_decimal_places(places, width=0):
        set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
        np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})

    set_numpy_decimal_places(1)

    print('method       |  specrad  |   cost   |  gains')
    for control_method_string, sr, cost, K in zip(control_method_strings, specrad_list, cost_list, K_list):
        print('%s  %.3f %8s    %s' % (control_method_string, sr, '%10.0f' % cost, K))




if __name__ == "__main__":
    # robust_stabilization2()

    # Paste / type code here while testing new experiments



    seed = 2
    npr.seed(seed)

    problem_data = example_system_erdos_renyi(n=3, m=2, p=2, seed=seed)
    # from data_io import load_problem_data
    # problem_data = load_problem_data(4)


    problem_data_keys = ['A', 'B', 'C', 'Ai', 'Bj', 'Ck', 'varAi', 'varBj', 'varCk', 'Q', 'R', 'S']
    A, B, C, Ai, Bj, Ck, varAi, varBj, varCk, Q, R, S = [problem_data[key] for key in problem_data_keys]
    n, m, p = [M.shape[1] for M in [A, B, C]]
    q, r, s = [M.shape[0] for M in [Ai, Bj, Ck]]

    # Initial gains
    K0, L0 = get_initial_gains(problem_data, initial_gain_method='zero')

    # Simulation options
    # Std deviation for initial state, defender inputs, attacker inputs, and additive noise
    xstd, ustd, vstd, wstd = 1.0, 1.0, 1.0, 0.0

    # Rollout length
    nt = 4000

    # Number of rollouts
    nr = 1

    # Rollout computation type
    group_option = 'group'

    # Q-function estimation scheme
    # qfun_estimator = 'direct'
    # qfun_estimator = 'lsadp'
    qfun_estimator = 'lstdq'

    sim_options_keys = ['xstd', 'ustd', 'vstd', 'wstd', 'nt', 'nr', 'group_option', 'qfun_estimator']
    sim_options_values = [xstd, ustd, vstd, wstd, nt, nr, group_option, qfun_estimator]
    sim_options = dict(zip(sim_options_keys, sim_options_values))

    num_iterations = 10

    # Policy iteration
    problem_data_known = True
    all_data = policy_iteration(problem_data, problem_data_known, K0, L0, sim_options, num_iterations)
    P, K, L, H, P_history, K_history, L_history, c_history, H_history = all_data
    verify_gare(problem_data, P, algo_str='Policy iteration')




    all_data_list = []
    problem_data_known = False
    num_trials = 3
    for i in range(num_trials):
        all_data_list.append(policy_iteration(problem_data, problem_data_known, K0, L0, sim_options, num_iterations))


    def norm_history_plotter(ax, M_history, M_history_list, ylabel_str):
        ax.plot(la.norm(M_history-M_history[-1], ord=2, axis=(1, 2)))
        for i in range(num_trials):
            ax.plot(la.norm(M_history_list[i]-M_history[-1], ord=2, axis=(1, 2)), color='k', alpha=0.5)
        ax.set_ylabel(ylabel_str, rotation=0)


    P_history_list = [all_data_list[i][4] for i in range(num_trials)]
    K_history_list = [all_data_list[i][5] for i in range(num_trials)]
    L_history_list = [all_data_list[i][6] for i in range(num_trials)]
    H_history_list = [all_data_list[i][8] for i in range(num_trials)]
    M_history_list_list = []

    fig, ax = plt.subplots(nrows=4)
    data_idx_list = [4, 5, 6, 8]
    ylabel_list = ['P', 'K', 'L', 'H']
    for i in range(4):
        j = data_idx_list[i]
        norm_history_plotter(ax[i], all_data[j], [all_data_list[i][j] for i in range(num_trials)], ylabel_list[i])
    # plt.tight_layout()
    ax[0].set_title('Relative norm of error vs iteration')
    plt.xlabel('Iteration')




    #
    # # Plotting
    # # plt.close('all')
    # t_history = np.arange(num_iterations)+1
    #
    # # Cost-to-go matrix
    # fig, ax = plt.subplots(ncols=2)
    # plt.suptitle('Value matrix (P)')
    # ax[0].imshow(P_pi)
    # ax[1].imshow(P_vi)
    # ax[0].set_title('Policy iteration')
    # ax[1].set_title('Value iteration')
    # # Specific stuff for Ben's monitor
    # fig.canvas.manager.window.setGeometry(3600, 600, 800, 600)
    # fig.canvas.manager.window.showMaximized()
    # plt.show()
    #
    # # Cost over iterations
    # fig, ax = plt.subplots()
    # ax.plot(t_history, c_history_pi)
    # ax.plot(t_history, c_history_vi)
    # plt.legend(['Policy iteration', 'Value iteration'])
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # if num_iterations <= 20:
    #     plt.xticks(np.arange(num_iterations)+1)
    #
    # # Specific stuff for Ben's monitor
    # fig.canvas.manager.window.setGeometry(3600, 600, 800, 600)
    # fig.canvas.manager.window.showMaximized()
    # plt.show()





    # # Plot closed-loop response of true and nominal models
    # nt = 100
    # x_hist = np.zeros([nt, n])
    # x_hist_true = np.zeros([nt, n_true])
    # u0 = np.ones(m)
    #
    # for i in range(nt-1):
    #     if i == 0:
    #         u = u0
    #         u_true = u0
    #     else:
    #         u = np.dot(Kare_true[:,0:n], x_hist[i])
    #         u_true =  np.dot(Kare_true, x_hist_true[i])
    #
    #     x_hist[i+1] = np.dot(A, x_hist[i]) + np.dot(B, u)
    #     x_hist_true[i+1] = np.dot(A_true, x_hist_true[i]) + np.dot(B_true, u_true)
    #
    #
    #
    # fig, ax = plt.subplots(nrows=2)
    #
    # ax[0].plot(x_hist)
    # ax[1].plot(x_hist_true)
    #
    #
    # t_history = np.arange(num_iterations)+1
    #
    #
    # # Cost over iterations
    # fig, ax = plt.subplots()
    # ax.plot(t_history, c_history_pi)
    # ax.plot(t_history, cn_history_pi)
    # ax.plot(t_history, cm_history_pi)
    # plt.legend(['Combined', 'Adversary only', 'Mult. noise only'])
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # if num_iterations <= 20:
    #     plt.xticks(np.arange(num_iterations)+1)
