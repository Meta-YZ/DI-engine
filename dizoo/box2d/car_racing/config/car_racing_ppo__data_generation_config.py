from ding.entry import serial_pipeline
from easydict import EasyDict

car_racing_ppo_data_generation_config = dict(
    exp_name='ppo_offpolicy_car_racing',
    env=dict(
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='CarRacing-v0',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        random_collect_size=2048,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=24,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0002,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            clip_ratio=0.1,
            learner=dict(
                hook=dict(save_ckpt_after_iter=1000)
            ),
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=1024,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(replay_buffer=dict(
            replay_buffer_size=100000,
            max_use=5,
        ), ),
    ),
)
car_racing_ppo_data_generation_config = EasyDict(car_racing_ppo_data_generation_config)
main_config = car_racing_ppo_data_generation_config

car_racing_ppo_data_generation_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.box2d.car_racing.envs.car_racing_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
)
car_racing_ppo_data_generation_create_config = EasyDict(car_racing_ppo_data_generation_create_config)
create_config = car_racing_ppo_data_generation_create_config