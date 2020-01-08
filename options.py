import time
import argparse

from easydict import EasyDict


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_id", type=str, default=str(int(time.time())))
    parser.add_argument("--logs", type=str, default="./logs/")
    parser.add_argument("--render", default=True, action="store_true")
    parser.add_argument("--render_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=10)

    # Data paths
    parser.add_argument("--train_path", type=str, default="levels/dummy_small_100")
    parser.add_argument("--test_path", type=str, default="levels/dummy_small_100")

    parser.add_argument("--target_update", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps_max", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.1)
    parser.add_argument("--eps_stop_step", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=10000)

    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--max_steps_eval", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--walls_prob", type=float, default=0)
    parser.add_argument("--static_prob", type=float, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", default=False, action="store_true")

    parser.add_argument("--hiddens", type=int, default=32)

    # Opt
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--rms_alpha", type=float, default=0.95)
    parser.add_argument("--rms_eps", type=float, default=0.01)
    parser.add_argument("--no_clamp_gradient", default=False, action="store_true")

    # Should improve performances, but degrades performances in experiments
    parser.add_argument(
        "--no_sensible_moves_gc",
        dest="sensible_moves_gc",
        default=True,
        action="store_false",
    )

    # Deadlocks # BUG with deadlocks : disabled !
    parser.add_argument("--early_stop_deadlocks", default=False, action="store_true")
    parser.add_argument("--no_penalize_deadlocks", default=True, action="store_true")
    parser.add_argument("--go_back_after_deadlocks", default=False, action="store_true")
    parser.add_argument("--reward_deadlocks", default=-1, type=float)

    # Prio replay
    # alpha_prioritised_replay
    # beta_prioritised_replay
    # incremental_td_error
    parser.add_argument("--alpha_prioritised_replay", default=0.6, type=float)
    parser.add_argument("--beta_prioritised_replay", default=0.1, type=float)
    parser.add_argument("--incremental_td_error", default=1e-8, type=float)

    args = EasyDict(parser.parse_args().__dict__)
    return args


if __name__ == "__main__":
    opt = parse_options()
