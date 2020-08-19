import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import matplotlib.pyplot as plt
txt_path = "/home/ubuntu_1/DDPG/resource_aware_control_rl_24_test" #the file used to store the txt

def train(env_id, env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, my_render=True, eg_explore=True,reward_param_scaling=1.0,
    reward_param_thr = 70, reward_param_type='const'):
    
    #save data
    ####################################
    full_path = txt_path + '_etc_RL.txt'
    file = open(full_path,'w')
    print('Start training for env: '+env_id)
    #change to your dir of choice for saving
    save_path = os.getcwd() 
    print('Save data at '+save_path+'. Change to your desired path.')

    
    dump_name = 'sav_ddpg_'+env_id+'.reward_'+reward_param_type+'_'+str(reward_param_scaling)+'.pkl'
    append_num = 0
    while os.path.exists(os.path.join(save_path,dump_name)):
        dump_name = 'sav_ddpg_'+env_id+'.reward_'+reward_param_type+'_'+str(reward_param_scaling)+'.'+str(append_num)+'.pkl'
        append_num+=1

    rank = MPI.COMM_WORLD.Get_rank()
    print('second rank is ',rank)
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.##############danny
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver(max_to_keep = 1)
    else:
        saver = None

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0
        
        
        epoch = 0
        start_time = time.time()

        epoch_com_sav = []

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        count_fwj = 0 # used to count how many step() is played
        for epoch in range(nb_epochs):
            print(nb_epochs)
            # collect data for saving plot
            save_data = {'act': [],
                         'obs': [],
                         'qpos':[],
                         'rew':[], # reward for this episode
                         'freq_com':[], # communication frequency
                         'act_ts': [],
                         'obs_ts': [],
                         'qpos_ts': [],
                         'rew_ts': [],  # reward for this episode
                         'freq_com_ts': [],  # communication frequency
                         'comm_r_factor':reward_param_scaling,
                         'eplen_ts':[] # len of test episodes
                         }

            # decay the exploration
            e_greed = 0.5 - 0.1 * np.log10( (t%10000) + 1)
            explore_switch = (t < 20000 and eg_explore and e_greed > 0)
            print('total steps: '+str(t)+', eps greedy rate: '+str(e_greed)+', explore is '+str(explore_switch))
            
            for cycle in range(nb_epoch_cycles):
                
                # Perform rollouts.

                # init u_old, don't forget to change test also
                u_old = 1.0 * env.action_space.sample() / max_action

                num_no_com = 0
                for t_rollout in range(nb_rollout_steps):

                    count_fwj+=1
                    print ('test played is ###################',count_fwj)
                    # Predict next action.
                    # edit this to be param version
                    if len(obs) is not 6:
                        obs.append(obs[2]-obs[0])
                        obs.append(obs[3]-obs[1])
                        a_raw, q = agent.pi(np.concatenate([obs,u_old],axis=0), apply_noise=False, compute_Q=True)

                    else: 
                        #a_1 = np.zeros()
                        a_raw, q = agent.pi(np.concatenate([obs,u_old],axis=0), apply_noise=False, compute_Q=True)
                        print('value of q is',q)
                        if count_fwj % 50000 == 0:
                            saver.save(sess,'./home/test_3_ar.ckpt',global_step =1)
                    a0 = a_raw[0]
                    a1 = a_raw[1]

                    # eps greedy, flip the coin
                    # make eps decay first 10k updates
                    dice_greed = np.random.uniform()
                    if explore_switch and dice_greed < e_greed:
                        com = ( np.random.uniform() > 0.5 )
                    else:
                        com = (a0 > a1)

                    # action according to com switch
                    if com:
                        r_com = 0.0
                        action = np.copy(a_raw[2:]) #No communication
                        num_no_com += 1             #No communication
                    else:
                        if reward_param_type=='const':
                            r_com = 1. # const reward
                        elif reward_param_type=='linear':
                            r_com = (1.0 / (nb_rollout_steps - reward_param_thr)) * (nb_rollout_steps - num_no_com) # linear interp reward
                        elif reward_param_type=='inv':
                            r_com = 1.0 / (1.0 + (np.maximum(num_no_com - reward_param_thr, 0)))  # inv decay reward
                        else:
                            print('no such reward type!')
                            assert 1==0

                        r_com = reward_param_scaling * r_com
                        #action = np.copy(u_old)
                        action = np.copy(a_raw[2:])
                        num_no_com += 1

                    assert action.shape == env.action_space.shape

                    assert max_action.shape == action.shape
                    new_obs, r, done, info = env.step(action) 

                    print(done)

                    file.write(str(new_obs)+',q_value_is,'+str(q)+',step_reward,'+str(r)+',action used,' + str(max_action*action)+'\n')


                    t += 1
                    if rank == 0 and render:
                        pass

                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(a_raw)
                    epoch_qs.append(q)

                    agent.store_transition(np.concatenate([obs,u_old],axis=0), a_raw, r+r_com, np.concatenate([np.squeeze(new_obs), action],axis=0) , done)
                    obs = np.squeeze(new_obs)

                    save_data['act'].append(np.array(action))
                    save_data['obs'].append(np.array(obs))
                    if hasattr(env.unwrapped, 'data'):
                        save_data['qpos'].append(np.array(env.unwrapped.data.qpos))

                    u_old = np.copy(action)

                    if done:


                        # Episode done.
                        epoch_com_sav.append(np.asarray(1.0*num_no_com/episode_step))

                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        print('one game finished, count is =================================',env.count)
                        file.write('count is,'+str(env.count))
                        file.write('done is,' + str(done))
                        file.write('long term reward is,' + str(env.long_term_reward))
                        file.write('#'*12+'one game finished\n')

                        agent.reset()
                        obs = env.reset()

                #end of loop nb_rollout
                print('communication savings: ' + str(num_no_com)) # check com number
                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                eval_episode_rewards = []
                eval_qs = []
                if eval_env is not None:
                    eval_episode_reward = 0.
                    for t_rollout in range(nb_eval_steps):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  
                        if render_eval:
                            eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_obs = eval_env.reset()
                            eval_episode_rewards.append(eval_episode_reward)
                            eval_episode_rewards_history.append(eval_episode_reward)
                            eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()

            # log stuff
            save_data['rew'].append(np.mean(epoch_episode_rewards))
            save_data['freq_com'].append(np.mean(epoch_com_sav))

            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)

            # Evaluation statistics.
            if eval_env is not None:
                combined_stats['eval/return'] = eval_episode_rewards
                combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
                combined_stats['eval/Q'] = eval_qs
                combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

        ###===============================================
        # test the fully-trained agent
        env = env.unwrapped

 
