import gymnasium as gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered


def test_null_policy():
    env = gym.make("JSBSim-v0")

    obs, info = env.reset()
    done = False
    trunc = False
    step = 0
    for t in range(100):
        u = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(u)
        step += 1

    env.close()
    assert step == 100


def test_truncated_end():
    env = gym.make("JSBSim-v0", max_episode_steps=10)

    obs, info = env.reset()
    done = False
    trunc = False
    step = 0
    for t in range(100):
        u = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(u)
        step += 1
        if trunc:
            break

    env.close()
    assert step == 10
    assert trunc