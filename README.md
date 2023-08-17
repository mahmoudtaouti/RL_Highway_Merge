# Multi-Agent Reinforcement Learning for Autonomous Vehicles (Use case: Highway on-Ramp Merging)

applying multi-agent techniques and approaches to the on-Ramp Merging Scenario, currently its based on the single agent reinforcement learning. Mainly value-based deep Q-learning and policy-based Advantage Actor-Critic algorithms.

## Algorithms

All the MARL algorithms are extended from the single-agent RL with parameter sharing and following centralized training with decentralized execution (CTDE) paradigm by using a centralized controller.

- [x] MADQN: independent learning and centralized learning (Currently there is a problem with QMIX network).
- [x] MAA2C: independant and centralized mode.


## Installation
- create an python virtual environment: `conda create -n marl_cav python=3.6 -y`
- active the virtul environment: `conda activate marl_cav`
- install pytorch (torch>=1.2.0): `pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`
- install the requirements: `pip install -r requirements.txt`


## Demo
[see the site]()

## Usage
To run the training, just run it via `python run_xxxx.py`.
To run the evaluation for existing models, just run it via `python test_env.py`.

## Training curves
<p align="center">
     <img src="eva/training_perfermance_4.png" alt="output_example" width="90%" height="90%">
     <br>Fig.2 Performance comparison between the implemented methods.
</p>


## Reference
