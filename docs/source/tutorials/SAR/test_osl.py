from myosuite.utils import gym
import numpy as np
# imports for SAR
from SAR_tutorial_utils import *

def train(env_name, policy_name, timesteps, seed):
    """
    Trains a policy using sb3 implementation of SAC.
    
    env_name: str; name of gym env.
    policy_name: str; choose unique identifier of this policy
    timesteps: int; how long you want to train your policy for
    seed: str (not int); relevant if you want to train multiple policies with the same params
    """
    env = gym.make(env_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    net_shape = [400, 300]
    policy_kwargs = dict(net_arch=dict(pi=net_shape, qf=net_shape))
    
    model = SAC('MlpPolicy', env, learning_rate=linear_schedule(.001), buffer_size=int(3e5),
            learning_starts=1000, batch_size=256, tau=.02, gamma=.98, train_freq=(1, "episode"),
            gradient_steps=-1,policy_kwargs=policy_kwargs, verbose=1)
    
    succ_callback = SaveSuccesses(check_freq=1, env_name=env_name+'_'+seed, 
                             log_dir=f'{policy_name}_successes_{env_name}_{seed}')
    
    model.set_logger(configure(f'{policy_name}_results_{env_name}_{seed}'))
    model.learn(total_timesteps=int(timesteps), callback=succ_callback, log_interval=4)
    model.save(f"{policy_name}_model_{env_name}_{seed}")
    env.save(f'{policy_name}_env_{env_name}_{seed}')



if __name__ == '__main__':

    train('myoChallengeOslRunRandom-v0', 'SAC', 1e5, '1')     

    # Include the locomotion track environment, uncomment to select the manipulation challenge
    # env = gym.make('myoChallengeOslRunRandom-v0')
    #env = gym.make('myoChallengeBimanual-v0')


    # env.reset()



    # # Repeat 1000 time steps
    # for _ in range(1000):

    #     # Activate mujoco rendering window
    #     # env.mj_render(headless=True)

    #     # Select skin group
    #     geom_1_indices = np.where(env.sim.model.geom_group == 1)
    #     # Change the alpha value to make it transparent
    #     env.sim.model.geom_rgba[geom_1_indices, 3] = 0
        
    #     # print(env.get_proprioception())
    #     # print(env.get_exteroception())

    #     # Get observation from the envrionment, details are described in the above docs
    #     obs = env.get_obs()
    #     #print(current_time)
    #     print(obs.shape)


    #     # Take random actions
    #     action = env.action_space.sample()


    #     # Environment provides feedback on action
    #     next_obs, reward, terminated, truncated, info = env.step(action)


    #     # Reset training if env is terminated
    #     if terminated:
    #         next_obs, info = env.reset()