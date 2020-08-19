#!/bin/bash
#To start run the test, type this in the Terminal: ./run_etc.sh

# set param
n_epoch=500
REW_TYPE="const" # const, inv, linear
param_val=0.1 # this parameter control the trade-off between the communication freq and performance
env_name="CarRacing-v0"
#env_name="Pendulum-v0"
seed=0 # use seed to reproduce results

# run script to train the resource aware agent
python -m baselines.ddpg.main_ddpg --reward_param_scaling ${param_val} --reward_param_type ${REW_TYPE} --env-id $env_name --nb-epochs ${n_epoch} --no-my_render --seed ${seed}
