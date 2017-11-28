# PPOC
This repository implements the Proximal Policy Option-Critic algorithm. It is based on the [baselines](https://github.com/openai/baselines) from OpenAI. In order to be able to run the code, you should first install it directly from their official repository. After that, you simply have to replace the some of the files contained in the [ppo1](https://github.com/openai/baselines/tree/master/baselines/ppo1) folder on your machine, with the ones in this repository.

To train a model and save the results:

`python run_mujoco.py --saves --wsaves --opt 2 --env Walker2d-v1 --seed 777 --app savename --dc 0.1`

where the most important parameter is "dc", the deliberation cost.


# Results
It is possible to view some of our results in this [video](https://www.youtube.com/watch?v=XI_txkRnKjU). 

<p align="center" size="width 150">
  <img src="https://github.com/mklissa/PPOC/blob/master/score3.png"/>
</p>



