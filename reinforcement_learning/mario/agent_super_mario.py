from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Flatten,
    Convolution2D,
    Permute,
    Dropout,
    Input,
    MaxPooling2D,
    Activation,
    AlphaDropout,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from PIL import Image
import numpy as np
import os
from datetime import datetime
import gym
from gym_environment import MarioWorld

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
now = datetime.now()
time = now.strftime("%H%M%S")

ENV_NAME = "mario_world"
env = MarioWorld()
env.seed(123)

run_id = ["dqn-bz", "dqn-eps", "dqn-double", "dqn-duel"]
LOG_DIR = f"logs/final/"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

weights_filename = f"{LOG_DIR}dqn_{ENV_NAME}_weights.h5f"
log_filename = f"{LOG_DIR}dqn_{ENV_NAME}_log.json"

np.random.seed(123)
INPUT_SHAPE = (64, 64)
WINDOW_LENGTH = 100
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
nb_actions = env.action_space.n
nb_steps = 60000


class ImageProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray((observation * 255).astype(np.uint8))
        img = img.resize(INPUT_SHAPE).convert("L")
        processed_observation = np.array(img)
        return processed_observation.astype("uint8")

    def process_state_batch(self, batch):
        processed_batch = batch.astype("float32") / 255.0
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)


model = Sequential()
processor = ImageProcessor()

model.add(Permute((2, 3, 1), input_shape=input_shape))
model.add(Convolution2D(64, (3, 3), strides=(2, 2), kernel_initializer="lecun_normal"))
model.add(Activation("selu"))
model.add(Convolution2D(128, (6, 6), strides=(2, 2), kernel_initializer="lecun_normal"))
model.add(Activation("selu"))
model.add(Convolution2D(32, (3, 3), strides=(1, 1), kernel_initializer="lecun_normal"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(AlphaDropout(0.25))
model.add(Activation("selu"))
model.add(Flatten())
model.add(Dense(256, kernel_initializer="lecun_normal"))
model.add(Activation("selu"))
model.add(AlphaDropout(0.4))
model.add(Dense(128, kernel_initializer="lecun_normal"))
model.add(Activation("selu"))
model.add(AlphaDropout(0.2))
model.add(Dense(nb_actions))
model.add(Activation("linear"))
print(model.summary())

memory = SequentialMemory(limit=nb_steps // 10, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr="eps",
    value_max=1.0,
    value_min=0.05,
    value_test=0.025,
    nb_steps=nb_steps,
)
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=nb_steps // 40,
    processor=processor,
    enable_double_dqn=True,
    enable_dueling_network=False,
    target_model_update=4,
    policy=policy,
    gamma=0.95,
    train_interval=2,
    delta_clip=1.0,
)
dqn.compile(Adam(lr=1e-4, epsilon=1e-08), metrics=["mae"])

callbacks = [ModelIntervalCheckpoint(weights_filename, interval=nb_steps // 20)]
callbacks += [FileLogger(log_filename, interval=nb_steps // 20)]
callbacks += [TensorBoard(log_dir=LOG_DIR)]

dqn.fit(
    env,
    action_repetition=1,
    nb_steps=nb_steps,
    visualize=False,
    callbacks=callbacks,
    verbose=2,
)
dqn.save_weights(weights_filename, overwrite=True)

dqn.load_weights(weights_filename)
dqn.test(env, nb_episodes=100, visualize=True)
