import numpy as np
import numpy.linalg as la
import numpy.random as npr
import matplotlib.pyplot as plt
import control as ctrl
import copy

from problem_data_gen import gen_double_spring_mass

from policy_iteration import policy_iteration, get_initial_gains, get_sim_options, verify_gare

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
    P_pi, K_pi, L_pi, P_history_pi, K_history_pi, L_history_pi, c_history_pi = policy_iteration(problem_data, problem_data_known, K0, L0, sim_options, num_iterations)
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
    Pn_pi, Kn_pi, Ln_pi, Pn_history_pi, Kn_history_pi, Ln_history_pi, cn_history_pi = policy_iteration(problem_data_model_n, problem_data_known, K0n, L0n, sim_options, num_iterations)
    verify_gare(problem_data_model_n, Pn_pi, algo_str='Policy iteration - Game w/o Multiplicative noise')

    # Policy Iteration on LQR w/ multiplicative noise
    K0m, L0m = get_initial_gains(problem_data_model_m, initial_gain_method='dare')
    print("LQR w/ multiplicative noise")
    Pm_pi, Km_pi, Lm_pi, Pm_history_pi, Km_history_pi, Lm_history_pi, cm_history_pi = policy_iteration(problem_data_model_m, problem_data_known, K0, L0, sim_options, num_iterations)
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





if __name__ == "__main__":
    robust_stabilization()

    # Paste / type code here while testing new experiments