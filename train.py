import torch
import torch.nn.functional as F
import pickle
import os

import options
import logger
from rl.qlearning_trainer_gc import (
    QLearningGraphCenteredTrainer,
    QLearningPrioritizedBufferGraphCenteredTrainer,
)
from model.graph_centered import GraphCenteredNet, GraphCenteredNetV2
from data.embedding import DirectionalPositionalEmbedding, DirectionalEmbedding


opt = options.parse_options()
opt.embedding = DirectionalPositionalEmbedding()

# Setup logger
logger.setup_logger(opt.logs, training_id=opt.training_id)
history_train = dict()
history_eval = dict()
if opt.use_prioritised_replay:
    trainer = QLearningPrioritizedBufferGraphCenteredTrainer(opt)
else:
    trainer = QLearningGraphCenteredTrainer(opt)

# Training loop
for epoch in range(opt.epochs):
    # Train
    train_epoch_info = trainer.train_one_epoch()
    # monitor the information about training
    for info in train_epoch_info:
        if info not in history_train:
            history_train[info] = [train_epoch_info[info]]
        else:
            history_train[info].append(train_epoch_info[info])

    # Evaluate
    eval_epoch_info = trainer.eval_one_epoch()
    # monitor the information about training
    for info in eval_epoch_info:
        if info not in history_eval:
            history_eval[info] = [eval_epoch_info[info]]
        else:
            history_eval[info].append(eval_epoch_info[info])

    # Save weights and history
    if epoch % opt.save_every == 0:
        trainer.save_model()
        with open(
            os.path.join(opt.logs, opt.training_id, "train_history.pkl"), "wb"
        ) as f:
            pickle.dump(history_train, f, pickle.HIGHEST_PROTOCOL)
        with open(
            os.path.join(opt.logs, opt.training_id, "eval_history.pkl"), "wb"
        ) as f:
            pickle.dump(history_eval, f, pickle.HIGHEST_PROTOCOL)

trainer.save_model()
with open(os.path.join(opt.logs, opt.training_id, "train_history.pkl"), "wb") as f:
    pickle.dump(history_train, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(opt.logs, opt.training_id, "eval_history.pkl"), "wb") as f:
    pickle.dump(history_eval, f, pickle.HIGHEST_PROTOCOL)
