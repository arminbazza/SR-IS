import os

import numpy as np
from scipy.stats import sem
import gymnasium as gym

import gym_env
from utils import decision_policy, update_terminal_reward, policy_reval, softmax
from utils_render import plot_decision_prob
from models import SR_IS, SR_TD

# Hyperparams
reward = -0.1
alpha = 0.05
beta = 1.0
num_steps = [40000, 60000, 150000, 250000, 350000, 450000]
num_iterations = 10
term_reward = 10
old_reward = 10
new_reward = 20
env = "maze-11x11-two-goal"

# For plotting
colors = [1, 9]
g1 = 4
g2 = 69
prob_point = 36
prob_locs = [29, 43]

# High level
save_figs = False
load_checkpoints = False

# Save dir
save_dir = os.path.join('..', 'figures/')
checkpoint_dir = os.path.join('..', 'checkpoints/policy_reval/')


if not load_checkpoints:
    for num_step in num_steps:
        step_label = f"{num_step // 1000}k"
        print(f"Running analysis for {num_step} steps:")
        prob_old_sr_is, prob_new_sr_is = [[], []], [[], []]
        for i in range(num_iterations):
            print(f"  Iteration {i+1}")
            np.random.seed(i)
            sr_is = SR_IS(env_name=env, reward=reward, term_reward=reward, alpha=alpha, beta=beta, 
                        num_steps=num_step, policy="softmax", imp_samp=True)

            update_terminal_reward(sr_is, loc=0, r=old_reward)

            ## Learn current terminal reward structure and get current policy
            sr_is.learn(seed=int(i))

            pii_old_sr_is = decision_policy(sr_is, sr_is.Z)

            ## Update terminal states and get new policy
            update_terminal_reward(sr_is, loc=1, r=new_reward)
            V_new, Z_new = policy_reval(sr_is)
            pii_new_sr_is = decision_policy(sr_is, Z_new)

            probs_old_sr_is = pii_old_sr_is[prob_point, prob_locs]
            probs_old_sr_is = probs_old_sr_is / np.sum(probs_old_sr_is)

            probs_new_sr_is = pii_new_sr_is[prob_point, prob_locs]
            probs_new_sr_is = probs_new_sr_is / np.sum(probs_new_sr_is)

            prob_old_sr_is[0].append(probs_old_sr_is[0])
            prob_old_sr_is[1].append(probs_old_sr_is[1])
            prob_new_sr_is[0].append(probs_new_sr_is[0])
            prob_new_sr_is[1].append(probs_new_sr_is[1])
            
        ## Save fig
        prob_train_mean = [np.mean(prob_old_sr_is[0]), np.mean(prob_old_sr_is[1])]
        prob_test_mean = [np.mean(prob_new_sr_is[0]), np.mean(prob_new_sr_is[1])]
        std_train = [sem(prob_old_sr_is[0]), sem(prob_old_sr_is[1])]
        std_test = [sem(prob_new_sr_is[0]), sem(prob_new_sr_is[1])]
        
        # Save checkpoints
        np.save(checkpoint_dir+'train_mean_' + step_label + '.npy', prob_train_mean)
        np.save(checkpoint_dir+'test_mean_' + step_label + '.npy', prob_test_mean)
        np.save(checkpoint_dir+'train_std_' + step_label + '.npy', std_train)
        np.save(checkpoint_dir+'test_std_' + step_label + '.npy', std_test)

        save_path = save_dir + 'policy_reval_sr_is_' + step_label + '.png' if save_figs else None
        ylabel = 'Probabilities' if num_step in [40000, 250000] else None
        
        plot_decision_prob(
            probs_train=prob_train_mean,
            probs_test=prob_test_mean,
            colors=colors,
            title=f'SR-IS {step_label}',
            ylabel=ylabel,
            leg_loc="upper center",
            save_path=save_path,
            std=[std_train, std_test],
            ymax=1.0,
            show=False,
            remove_spine=True
        )

# Load checkpoints
else:
    for num_step in num_steps:
        step_label = f"{num_step // 1000}k"
        prob_train_mean = np.load(checkpoint_dir+'train_mean_' + step_label + '.npy')
        prob_test_mean = np.load(checkpoint_dir+'test_mean_' + step_label + '.npy')
        std_train = np.load(checkpoint_dir+'train_std_' + step_label + '.npy', std_train)
        std_test = np.load(checkpoint_dir+'test_std_' + step_label + '.npy', std_test)
        save_path = save_dir + 'policy_reval_sr_is_' + step_label + '.png' if save_figs else None
        ylabel = 'Probabilities' if num_step in [40000, 250000] else None
        plot_decision_prob(
                probs_train=prob_train_mean,
                probs_test=prob_test_mean,
                colors=colors,
                title=f'SR-IS {step_label}',
                ylabel=ylabel,
                leg_loc="upper center",
                save_path=save_path,
                std=[std_train, std_test],
                ymax=1.0,
                show=False,
                remove_spine=True
            )