from gym.envs.registration import register

register(
    id = 'video_player-v0',
    entry_point = 'env.myenv:MyEnv'
)
