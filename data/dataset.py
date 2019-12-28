import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


class InMemorySokobanDataset(torch.utils.data.Dataset):
    def __init__(self, root, embedding, device=None):
        super().__init__()
        self.root = root
        self.embedding = embedding
        files = glob.glob(os.path.join(root, "*.png"))

        def load_image(fname):
            return (255 * plt.imread(fname)[:, :, :3]).astype(np.uint8)

        init_images = [load_image(f) for f in files]
        self.init_states = [self.embedding(img) for img in init_images]
        if device is not None:
            for s in self.init_states:
                s.to(device)

    def __getitem__(self, idx):
        return self.init_states[idx]

    def __len__(self):
        return len(self.init_states)

