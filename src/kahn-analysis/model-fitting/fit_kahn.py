import pickle
import os
import numpy as np
from cbm.individual_fit import individual_fit, Config
from cbm.model_selection import bms

def policy(model, D, r, state, V, parameters):
    n_states = 11
    T = np.zeros((n_states, n_states))
    T[0, 1:3] = 0.5
    T[1, 3:5] = 0.5
    T[2, 5:7] = 0.5
    T[3, 7] = 1.0
    T[4, 8] = 1.0
    T[5, 9] = 1.0
    T[6, 10] = 1.0
    T[7:11, 7:11] = np.eye(4)        
    terminals =  np.diag(T) == 1
    succ_states = np.where(T[state] > 0)[0]
    if model == 'linrl':
        gamma = parameters[0]            
        lambda_ =parameters[1]    
        beta = parameters[2]
        const = 1e-10
        
        D = gamma * np.linalg.inv(np.eye(n_states-4) - gamma * T[~terminals][:,~terminals])            

        t = np.zeros(n_states-4) # P @ np.exp(r[terminals] / lambda_)
        rmax = max(r[terminals])
        # tmax = np.exp( rmax / lambda_)
        t[3:] = np.exp( (r[terminals] - rmax) / lambda_)
        Z = np.zeros(n_states)
        Z[~terminals] = D @ t
        Z[terminals] = t[3:]
        Z[Z < const] = const  # prevent log(0)
        
        log_Z = np.log(Z)            
        log_Z += rmax / lambda_                   
        V = log_Z[succ_states] * lambda_

        V /= beta            
    
    if model == 'sr_is':
        gamma = parameters[0]
        lambda_ = parameters[1]
        beta = parameters[2]
        const = 1e-10            

        t = np.zeros(n_states-4) # P @ np.exp(r[terminals] / lambda_)
        rmax = max(r[terminals])
        # tmax = np.exp( rmax / lambda_)
        t[3:] = np.exp( (r[terminals] - rmax) / lambda_)
        Z = np.zeros(n_states)
        Z[~terminals] = (gamma * D[~terminals][:,~terminals]) @ t
        Z[terminals] = t[3:]
        Z[Z < const] = const  # prevent log(0)
        
        log_Z = np.log(Z)            
        log_Z += rmax / lambda_                   
        V = log_Z[succ_states] * lambda_

        V /= beta
    elif model == 'sr':
        beta = parameters[0]

        V = D @ r
        V = V[succ_states]
        V /= beta
    elif model == 'mb':
        beta = parameters[0]
        V = np.zeros(n_states)
        V[terminals] = r[terminals]
        V[3:7] = r[terminals]
        V[2] = np.max(V[5:7])
        V[1] = np.max(V[3:5])
        V[0] = np.max(V[1:3])
        V[1:3] = V[1:3]/beta
        V[3:7] = V[3:7]/beta
        V = V[succ_states]
    elif model == 'hybrid':
        beta1 = parameters[0]
        beta2 = parameters[1]
        V_sr = D @ r
        V_mb = np.zeros(n_states)
        V_mb[terminals] = r[terminals]
        V_mb[3:7] = r[terminals]
        V_mb[2] = np.max(V_mb[5:7])
        V_mb[1] = np.max(V_mb[3:5])
        V_mb[0] = np.max(V_mb[1:3])
        V_net = V_mb/beta1 + V_sr/beta2
        V = V_net[succ_states]      
    
    probs = np.zeros(n_states)
    exp_Q = np.exp(V - np.max(V))
    probs[succ_states] = exp_Q / (np.sum(exp_Q))
    return probs

def define_model_parameters(model, parameters=None):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    exp_func = lambda x: np.exp(x)
    
    if parameters is None:
        parameters = np.zeros(5)

    alpha_r = 0.0
    alpha_D = 0.0
    gamma = 0.0  
    if model == 'linrl':
        alpha_r = sigmoid(parameters[0])  
        gamma = sigmoid(parameters[1])        
        lambda_ = -1/np.log(gamma + 1e-10) # assume c=1, so lambda = -1/log(gamma)
        beta = lambda_
        policy_params = [gamma, lambda_, beta]
        pnames = ['alpha_r', 'gamma']                
    if model == 'sr_is':
        alpha_r = sigmoid(parameters[0])        
        alpha_D = sigmoid(parameters[1])
        gamma = sigmoid(parameters[2])
        lambda_ = exp_func(parameters[3])
        beta = lambda_
        policy_params = [gamma, lambda_, beta]
        pnames = ['alpha_r', 'alpha_D', 'gamma', 'lambda']
    if model == 'sr':
        alpha_r = sigmoid(parameters[0])        
        alpha_D = sigmoid(parameters[1])
        gamma = sigmoid(parameters[2])
        beta = exp_func(parameters[3])
        policy_params = [beta]
        pnames = ['alpha_r', 'alpha_D', 'gamma', 'beta']
    if model == 'hybrid':
        alpha_r = sigmoid(parameters[0])        
        alpha_D = sigmoid(parameters[1])
        gamma = sigmoid(parameters[2])
        beta1 = exp_func(parameters[3])        
        beta2 = exp_func(parameters[4])
        policy_params = [beta1, beta2]
        pnames = ['alpha_r', 'alpha_D', 'gamma', 'beta1', 'beta2']  
    if model == 'mb':
        alpha_r = sigmoid(parameters[0])        
        beta = exp_func(parameters[1])        
        policy_params = [beta]
        pnames = ['alpha_r', 'beta']

    return pnames, alpha_r, alpha_D, gamma, policy_params

def model(parameters, data, model, simulate=False):
    _, alpha_r, alpha_D, gamma, policy_params = define_model_parameters(model, parameters)

    n_states = 11
    terminals = np.zeros(n_states, dtype=bool)
    terminals[7:] = True

    # inialize M
    M = np.eye(n_states)
    M[~terminals, :] = 1    
    np.fill_diagonal(M, (1e-10+1 - gamma)**-1)

    r = np.zeros(n_states)

    loglik = 0.0    
    V = np.zeros(n_states)
    for t in range(len(data)):
        states = data[t]['states']
        reward = data[t]['reward']
        simulated_states = []
        for i in range(len(states)-1):
            state = states[i]-1
            next_state = states[i+1]-1            
            
            probs = policy(model, M, r, state, V, policy_params)
            if simulate:
                simulated_states.append(np.random.choice(np.arange(n_states)+1, p=probs))
            loglik += np.log(probs[next_state] + 1e-10)

            if model == 'sr_is':
                w = 0.5/(max(probs[next_state], 1e-2))
            else:
                w = 1
            
            target = np.eye(n_states)[state] + gamma * M[next_state]
            M[state] = (1 - alpha_D) * M[state] + alpha_D * target * w
            V[state] = (1 - alpha_r) * V[state] + alpha_r * gamma * V[next_state] 

        state = states[-1]-1
        M[state] = (1 - alpha_D) * M[state] + alpha_D * np.eye(n_states)[state]        
        r[state] = (1 - alpha_r) * r[state] + alpha_r * reward

    return loglik 

def define_prior(model):
    pnames, _, _, _, _ = define_model_parameters(model, parameters=None)
    num_dim = len(pnames)
    prior_mean = np.zeros(num_dim)
    prior_variance = np.zeros(num_dim)
    for i in range(num_dim):
        if pnames[i].startswith('beta'):
            prior_variance[i] = 100.0
        else:
            prior_variance[i] = 6.25
    return prior_mean, prior_variance


def transformed_parameter_dict(model, parameters):
    """Map raw fitted parameters to transformed, named parameters."""
    pnames, alpha_r, alpha_D, gamma, policy_params = define_model_parameters(model, parameters)

    if model == 'linrl':
        return {'alpha_r': alpha_r,
                'lambda': policy_params[1]}    
    if model == 'sr_is':
        return {
            'alpha_r': alpha_r,
            'alpha_D': alpha_D,
            'gamma': gamma,
            'lambda': policy_params[1],
        }
    if model == 'sr':
        return {
            'alpha_r': alpha_r,
            'alpha_D': alpha_D,
            'gamma': gamma,
            'beta': policy_params[0],
        }
    if model == 'hybrid':
        return {
            'alpha_r': alpha_r,
            'alpha_D': alpha_D,
            'gamma': gamma,
            'beta1': policy_params[0],
            'beta2': policy_params[1],
        }
    if model == 'mb':
        return {'alpha_r': alpha_r, 'beta': policy_params[0]}

    # Fallback: keep original parameter ordering if model is unknown.
    return {pnames[j]: parameters[j] for j in range(min(len(pnames), len(parameters)))}

def run_fitting(data, data_str):

    model_names = ['sr_is', 'sr', 'mb', 'hybrid', 'linrl']
    im = [0, 1, 2, 3, 4]
    lme = []  # Will build list of columns, then stack at end
    for i in range(len(im)):
        i_model = im[i]
        np.random.seed(42)
        model_name = model_names[i_model]        
        cbm_file = f'{data_str}/{model_name}.pkl'
        print(f"Model is: {model_name}")
        prior_mean, prior_variance = define_prior(model_name)
        cfg = Config(num_init=1, num_init_med=3, num_init_up=5)
        mdl = lambda p, d: model(p, d, model=model_name)
        if not os.path.exists(cbm_file):
            os.makedirs(os.path.dirname(cbm_file), exist_ok=True)
            cbm = individual_fit(data, mdl, prior_mean, prior_variance, cbm_file, cfg)
        with open(cbm_file, 'rb') as f:
            cbm = pickle.load(f)

        # Submit each subject's fit to define_model_parameters before summary stats.
        fitted_params = np.asarray(cbm.output.parameters)
        transformed_rows = [
            transformed_parameter_dict(model_name, np.asarray(row, dtype=float))
            for row in fitted_params
        ]

        if transformed_rows:
            ordered_keys = list(transformed_rows[0].keys())
            print("Transformed parameter summaries:")
            for key in ordered_keys:
                values = np.asarray([row[key] for row in transformed_rows], dtype=float)
                mean = np.mean(values)
                std = np.std(values)
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                print(
                    f"  {key}: mean={mean:.3f}, std={std:.3f}, "
                    f"Q1={q1:.3f}, median={median:.3f}, Q3={q3:.3f}"
                )
        
        lme.append(cbm.output.log_evidence)

    lme = np.column_stack(lme)  # Convert list to 2D array: subjects × models
    bms_result = bms(lme)    
    print(f"\nExpected model frequencies: {np.array2string(bms_result.model_frequency, suppress_small=True, precision=3)}")
    print(f"Protected exceedance probabilities: {np.array2string(bms_result.protected_exceedance_prob, suppress_small=True, precision=3)}")
    print(f"\nBor: {np.array2string(bms_result.bor, suppress_small=True, precision=7)}")


if __name__ == "__main__":
    all_subjects_data_file = 'data/kahn/all_subjects_data.pkl'
    with open(all_subjects_data_file, 'rb') as f:        
        data = pickle.load(f); data_str = 'fit_kahn'

    print(f"Number of subjects: {len(data)}")
    run_fitting(data, data_str)
