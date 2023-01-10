# DQN-with-MCTS-on-Adaptive-Bitrate
Official codes for ["Training Deep Q-Network via Monte Carlo Tree Search for Adaptive Bitrate Control in Video Delivery"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4297671)

## Create Environment
The codes are implementable with Python 3.6.5 and tensorflow2

## How To Train
Run
``` Bash
python train.py
```
You can see the following console outputs
```Bash
Episode 19:   4%|██▊                                                                  | 20/500 [00:06<02:40,  3.00it/s, episode_reward=32.6, running_reward=-195]
```
Log for training rewards is saved as [./log.txt](https://github.com/xiaotianliu01/DQN-with-MCTS-on-Adaptive-Bitrate/blob/main/log.txt). Best model during training is saved as [./model_best.h5](https://github.com/xiaotianliu01/DQN-with-MCTS-on-Adaptive-Bitrate/blob/main/model_best.h5).

## How To Test
Run
``` Bash
python test.py
```
You can see the following console outputs which are overal average reward and bitrate action statistics
``` Bash
 start testing...
119.0951
[508, 10109, 943]
```
Logs for detalis of all decision processes are saved under folder [./test_log/](https://github.com/xiaotianliu01/DQN-with-MCTS-on-Adaptive-Bitrate/tree/main/test_log).
