import gymnasium as gym
u
import imageio as iio
from features import JSBSimFeatureExtractor
from stable_baselines3 import SAC

def main() -> None:

    policy_kwargs = dict(
        features_extractor_class=JSBSimFeatureExtractor
    )

    env = gym.make("JSBSim-v0", render_mode="rgb_array")

    model = SAC.load("models/jsbsim_sac", env)

    mp4_writer = iio.get_writer("video.mp4", format="ffmpeg", fps=30)
    gif_writer = iio.get_writer("video.gif", format="gif", fps=5)
    obs, info = env.reset()
    done = False
    trunc = False
    step = 0
    while not done:
        render_data = env.render()
        mp4_writer.append_data(render_data)

        if step % 6 == 0:
            gif_writer.append_data(render_data[::2,::2,:])

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        step += 1

    mp4_writer.close()
    gif_writer.close()
    env.close()

if __name__ == '__main__':
    main()