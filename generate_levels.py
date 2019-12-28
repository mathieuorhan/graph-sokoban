import os
import uuid
import random

import tqdm
import gym
import gym_sokoban
import numpy as np
import matplotlib.pyplot as plt


# Parameters
ENV_LEVEL = "Sokoban-small-v0"
OUT_PATH = "levels/very_easy_1"
SEED = 123
TRAIN_SIZE = 1
TEST_SIZE = 1
RENDER_MODE = "tiny_rgb_array"

# Other constants
TRAIN_PATH = os.path.join(OUT_PATH, "train")
TEST_PATH = os.path.join(OUT_PATH, "test")

# Seed
np.random.seed(SEED)
random.seed(SEED)

# Create env
env = gym.make(ENV_LEVEL)


def hash_array(x):
    return hash(x.tostring())


"""
As the process of generation is random, 
we hash the initial states to remove duplicates, and
prevent data leak. 
Note that two differents images could correspond
to the same graph embedding...
"""
train_hash = set()
test_hash = set()

for path, size, hash_set in [
    (TEST_PATH, TEST_SIZE, test_hash),
    (TRAIN_PATH, TRAIN_SIZE, train_hash),
]:
    # Create output folder
    os.makedirs(path, exist_ok=True)

    # Populate folder
    for _ in tqdm.tqdm(range(size)):
        uname = str(uuid.uuid4()) + ".png"
        fname = os.path.join(path, uname)
        init_state = env.reset(render_mode=RENDER_MODE)
        token = hash_array(init_state)
        if (token not in test_hash) and (token not in train_hash):
            hash_set.add(token)
            plt.imsave(fname, init_state)

print("Total train levels: ", len(train_hash))
print("Total test levels: ", len(test_hash))
assert train_hash.intersection(test_hash) == set()
