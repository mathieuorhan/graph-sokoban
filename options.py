import argparse
from easydict import EasyDict


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs", type=str, help="output dir to save the logs", default="./logs/",
    )
    # Data paths
    parser.add_argument("--train_path", type=str, default="levels/dummy_10")
    parser.add_argument("--test_path", type=str, default="levels/dummy_10")

    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eps_max", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.1)
    parser.add_argument("--eps_stop_step", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=5000)

    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--max_steps_eval", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--walls_prob", type=float, default=0)
    parser.add_argument("--static_prob", type=float, default=0)

    # Opt
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--rms_alpha", type=float, default=0.95)
    parser.add_argument("--rms_eps", type=float, default=0.01)

    # Deadlocks # BUG with deadlocks : disabled !
    parser.add_argument("--early_stop_deadlocks", default=False, action="store_true")
    parser.add_argument("--no_penalize_deadlocks", default=True, action="store_true")
    parser.add_argument("--go_back_after_deadlocks", default=False, action="store_true")
    parser.add_argument("--reward_deadlocks", default=-1, type=float)

    args = EasyDict(parser.parse_args().__dict__)
    return args


if __name__ == "__main__":
    opt = parse_options()
