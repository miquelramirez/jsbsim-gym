import numpy as np

import gymnasium as gym
import jsbsim_gym.jsbsim_gym
from time import sleep
import imageio as iio

def main() -> None:

    mp4_writer = iio.get_writer("video.mp4", format="ffmpeg", fps=30)
    gif_writer = iio.get_writer("video.gif", format="gif", fps=5)

    step: int = 0

    env = env = gym.make("JSBSim-v0", max_episode_steps=10, render_mode="rgb_array")
    obs, info = env.reset()
    render_data = env.render()


    for _ in range(300):
        mp4_writer.append_data(render_data)
        if step % 6 == 0:
            gif_writer.append_data(render_data[::2,::2,:])
        obs, reward, done, trunc, info = env.step(np.array([0.05, -0.2, 0, .5]))
        render_data = env.render()
        sleep(1/30)
        step += 1

    mp4_writer.close()
    gif_writer.close()
    env.close()

if __name__ == "__main__":
    main()