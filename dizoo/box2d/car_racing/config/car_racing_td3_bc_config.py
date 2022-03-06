from easydict import EasyDict

car_racing_td3_bc_config = dict(
    exp_name='car_racing_td3_bc',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
        frame_stack=4,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=800,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            twin_critic=True,
            action_space='regression',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            train_epoch=30000,
            batch_size=128,
            learning_rate_actor=1e-4,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
            alpha=2.5,
        ),
        collect=dict(
            noise_sigma=0.1,
            data_type='hdf5',
            data_path='./td3/expert_demos.hdf5',
            normalize_states=True,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=20000, ), ),
    ),
)
car_racing_td3_bc_config = EasyDict(car_racing_td3_bc_config)
main_config = car_racing_td3_bc_config

car_racing_td3_bc_create_config = dict(
    env=dict(
        type='car_racing',
        import_names=['dizoo.box2d.car_racing.envs.car_racing_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='td3_bc',
        import_names=['ding.policy.td3_bc'],
    ),
)
car_racing_td3_bc_create_config = EasyDict(car_racing_td3_bc_create_config)
create_config = car_racing_td3_bc_create_config