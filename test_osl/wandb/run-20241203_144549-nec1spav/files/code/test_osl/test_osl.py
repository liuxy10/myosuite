from myosuite.utils import gym
import numpy as np
from utils import *
from in_callbacks import *
from stable_baselines3.common.callbacks import CheckpointCallback

IS_WnB_enabled = False
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    IS_WnB_enabled = True
    
except ImportError as e:
    pass 

def train(env_name, policy_name, timesteps, seed):
    """
    Trains a policy using sb3 implementation of SAC.
    
    env_name: str; name of gym env.
    policy_name: str; choose unique identifier of this policy
    timesteps: int; how long you want to train your policy for
    seed: str (not int); relevant if you want to train multiple policies with the same params
    """
    
    config = {
        "policy_type": policy_name,
        "total_timesteps": timesteps,
        "env_name": env_name,
    }
    if IS_WnB_enabled:
        run = wandb.init(
            project="{policy_name}_model_{env_name}_{seed}",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    log = configure(f'results_{policy_name}_{env_name}')
    env = gym.make(env_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))
    freq = 1000
    
    if policy_name == 'SAC':
        model = SAC('MlpPolicy', env, learning_rate=linear_schedule(.001), buffer_size=int(3e5),
                learning_starts=1000, batch_size=256, tau=.02, gamma=.98, train_freq=(1, "episode"),
                gradient_steps=-1,policy_kwargs=policy_kwargs, verbose=1,
                tensorboard_log=f"wandb/{run.id}")
    
    
    if IS_WnB_enabled:
        callback = [WandbCallback(
                model_save_path=f"models/{policy_name}_model_{env_name}_{seed}",
                verbose=2,
            )]
    else:
        print('Wandb not enabled')
        callback = []
    
    callback += [EvalCallback(freq, env)]
    callback += [InfoCallback()]
    callback += [FallbackCheckpoint(freq)]
    callback += [CheckpointCallback(save_freq=freq, save_path=f'logs/',
                                            name_prefix='rl_models')]
    
    # callback += SaveSuccesses(check_freq=1, env_name=env_name+'_'+seed, 
    #                          log_dir=f'{policy_name}_successes_{env_name}_{seed}') 
    
    # model.set_logger(configure(f'{policy_name}_results_{env_name}_{seed}'))
    model.learn(total_timesteps=int(timesteps), callback=callback, log_interval=4)
    model.save(f"{policy_name}_model_{env_name}_{seed}")
    env.save(f'{policy_name}_env_{env_name}_{seed}')

def zeroshot_test(name, test_env_name, env_name='myoHandReorient100-v0', seed='0', determ=True, ica=None, 
                  pca=None, normalizer=None, phi=.66, episodes=500, is_sar=False, syn_nosyn=False):
    """
    Check zero-shot performance of policies on the test environments.
    
    name: str; name of the policy to test
    env_name: str; name of gym env the policy to test was trained on (Reorient100-v0).
    seed: str; seed of the policy to test
    test_env_name: str; name of the desired test env
    ica: if testing SAR-RL, the ICA object
    pca: if testing SAR-RL, the PCA object
    normalizer: if testing SAR-RL, the normalizer object
    phi: float; blend parameter between synergistic and nonsynergistic activations
    episodes: int; number of episodes to run on the test environment
    """
    if is_sar:
        if syn_nosyn:
            env = SynNoSynWrapper(gym.make(test_env_name), ica, pca, normalizer, phi)
        else:
            # env = SynergyWrapper(gym.make(test_env_name), ica, pca, normalizer, phi)
            env = SynergyWrapper(gym.make(test_env_name), ica, pca, normalizer)
    else:
        env = gym.make(test_env_name)
    env.reset()

    model = SAC.load(f'{name}_model_{env_name}_{seed}')
    vec = VecNormalize.load(f'{name}_env_{env_name}_{seed}', DummyVecEnv([lambda: env]))
    solved = []
    for i,_ in enumerate(range(episodes)):
        is_solved = []
        env.reset()
        done = False
        while not done:
            o = env.get_obs()
            o = vec.normalize_obs(o)
            a, __ = model.predict(o, deterministic=determ)
            next_o, r, done, *_, info = env.step(a)
            is_solved.append(info['solved'])
        
        if sum(is_solved) > 0:
            solved.append(1)
        else:
            solved.append(0)

    env.close()
    return np.mean(solved)

def plot_results(folder_path, smth):
    a_df = pd.read_csv(folder_path + "/progress.csv")
    a_timesteps = a_df['time/total_timesteps'][:-smth]
    a_reward_mean = smooth(a_df['rollout/ep_rew_mean'], smth)[:-smth]
    plt.plot(a_timesteps, a_reward_mean, linewidth=3, label='SAR-RL')
    plt.grid()

    plt.xlabel('environment iterations', fontsize=14)
    plt.ylabel('success/reward metric', fontsize=14)

    plt.legend(fontsize=11, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f'{folder_path}.png')

# def get_vid(name, env_name, seed, episodes, vid_name):
    

if __name__ == '__main__':
    # training for 
    env_id = "myoLegWalk-v0"
    # env_id = "myoLegStandRandom-v0"
    train(env_id, 'SAC', 1e6, '1') 
    # suc = zeroshot_test('SAC', env_id, env_id, '1', episodes=5)
    
    # print(suc)
  
    # get_vid( 'SAC',env_id, '1', 1, f'SAC_{env_id}_video.mp4')

    # plot_results("SAC_results_myoLegWalk-v0_1", 10) 