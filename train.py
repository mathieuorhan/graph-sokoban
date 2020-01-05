import pickle
import time
import os

import torch.nn.functional as F

import options
import logger
from rl.qlearning_trainer import QLearningTrainer
from rl.qlearning_trainer_gc import QLearningGraphCenteredTrainer
from data.embedding import DirectionalPositionalEmbedding

opt = options.parse_options()
opt.embedding = DirectionalPositionalEmbedding()
opt.training_id = str(int(time.time()))

# Setup logger
logger.setup_logger(opt.logs, training_id=opt.training_id)

# trainer = QLearningTrainer(opt)
trainer = QLearningGraphCenteredTrainer(opt)
history = dict()

for epoch in range(opt.epochs):
    epoch_info = trainer.train_one_epoch()
    # monitor the information about training
    for info in epoch_info:
        if info not in history:
            history[info] = [epoch_info[info]]
        else:
            history[info].append(epoch_info[info])

    trainer.eval_one_epoch()

    if opt.render and epoch % opt.render_every == 0:
        trainer.render_one_episode(0)

    # Save weights
    trainer.save_model()

# Saving history in the logs
with open(os.path.join(opt.logs, opt.training_id, "history.pkl"), "wb") as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
