## A Pareto Actor-Critic Framework for multiple FL Providers Co-optimization

This repository contains the source code for the paper: [Pareto Actor-Critic for Communication and
Computation Co-Optimization in Non-Cooperative
Federated Learning Services](https://arxiv.org/abs/2508.16037). It demonstrates the use of Multi-Agent Reinforcement Learning (MARL) for joint optimization of communication and computation resources in a non-cooperative, multi-service provider Federated Learning (FL) environment.

## Files Overview
*run.py*: The main script to start the MARL training and evaluation process.

*fed.py*: Contains the core logic for the Federated Learning simulation, including the FLServer and FLClient classes.

*env_runner.py*: Defines the MultiAgentEnv that wraps the FL simulation and provides the MDP interface (state, action, reward) for the agents.

*control.py*: Implements the multi-agent controller (MAC) which holds the agent policies and selects actions.

*Learner/pareto_learner.py*: The implementation of the PAC learning algorithm for updating agent parameters.

*config.py*: The script for fetching all hyperparameters and settings for the experiments.

*episode_replaybuffer.py*: The replay buffer for storing and sampling transition data.

## Requirements
See the requirements.txt for the required python packages and run ```python 
pip install -r requirements.txt``` to install them.


## Usage
1. Train the PAC-MARL agent using the ```run.py``` script.
2. Test traditional FL performance using the ```fed.py``` script.