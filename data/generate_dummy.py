import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from data.constants import SokobanElements as se
from data.utils import ascii_to_img


def generate_dummy(width, height, n_box, n_levels, savedir):
    """Generates rectangular levels without interior walls.
    """
    os.makedirs(savedir, exist_ok=True)

    for lvl in range(n_levels):
        # Create the template
        w, h = width - 2, height - 2
        lines = [width * [se.WALL]]
        for _ in range(h):
            lines.append(list(se.WALL + w * se.FLOOR + se.WALL))
        lines.append(width * [se.WALL])

        nb_random_try = 0
        max_random_try = 10

        for _ in range(n_box):
            # Place the boxes
            crit = True
            while crit and nb_random_try < max_random_try:
                x, y = np.random.randint(2, w), np.random.randint(2, h)
                crit = lines[y][x] != se.FLOOR
                nb_random_try += 1
            lines[y][x] = se.BOX

            # Place the targets
            crit = True
            while crit and nb_random_try < max_random_try:
                x, y = np.random.randint(1, w + 1), np.random.randint(1, h + 1)
                crit = lines[y][x] != se.FLOOR
                nb_random_try += 1
            lines[y][x] = se.BOX_TARGET

        # Place the player
        crit = True
        while crit and nb_random_try < max_random_try:
            x, y = np.random.randint(1, w + 1), np.random.randint(1, h + 1)
            crit = lines[y][x] != se.FLOOR
            nb_random_try += 1
        lines[y][x] = se.PLAYER
        # Save the level as txt
        os.makedirs(savedir, exist_ok=True)
        txt_fname = os.path.join(
            savedir, "dummy_{}x{}_{}box_{}.txt".format(width, height, n_box, lvl)
        )
        with open(txt_fname, "w") as f:
            for line in lines:
                f.write("".join(line) + "\n")

        img_fname = os.path.join(
            savedir, "dummy_{}x{}_{}box_{}.png".format(width, height, n_box, lvl)
        )
        img = ascii_to_img(txt_fname)
        plt.imsave(img_fname, img)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="output dir to save the levels",
        default="./levels/",
    )
    parser.add_argument("-w", "--width", type=int, default=7, help="number of columns")
    parser.add_argument("-r", "--height", type=int, default=7, help="number of rows")
    parser.add_argument(
        "-b", "--boxes", type=int, default=1, help="number of boxes / targets"
    )
    parser.add_argument(
        "-l", "--levels", type=int, default=10, help="number of levels to generate"
    )

    args = parser.parse_args()
    savedir, width, height, n_box, n_levels = (
        args.dir,
        args.width,
        args.height,
        args.boxes,
        args.levels,
    )

    # Generate levels
    generate_dummy(width, height, n_box, n_levels, savedir)
