import numpy as np 
import matplotlib.pyplot as plt 
from constants import SokobanElements as se


def generate_dummy(width, height, n_box, n_levels, savedir):
    """Generates rectangular levels without interior walls."""
    for lvl in range(n_levels):
        # Create the template
        w, h = width - 2, height - 2
        lines = [width * [se.WALL]]
        for _ in range(h):
            lines.append(list(se.WALL + w * se.FLOOR + se.WALL))
        lines.append(width * [se.WALL])

        for _ in range(n_box):
            # Place the targets
            crit = True
            while crit:
                x, y = np.random.randint(1, w + 1), np.random.randint(1, h + 1)
                crit = lines[y][x] != se.FLOOR
            lines[y][x] = se.BOX_TARGET

            # Place the boxes
            crit = True
            while crit:
                x, y = np.random.randint(2, w), np.random.randint(2, h)
                crit = lines[y][x] != se.FLOOR
            lines[y][x] = se.BOX

        # Place the player
        crit = True
        while crit:
            x, y = np.random.randint(1, w + 1), np.random.randint(1, h + 1)
            crit = lines[y][x] != se.FLOOR
        lines[y][x] = se.PLAYER

        # Save the level
        with open(savedir + 'dummy_{}x{}_{}box_{}.txt'.format(width, height, n_box, lvl), 'w') as f:
            for line in lines:
                f.write(''.join(line) + '\n')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="output dir to save the levels", default="./levels/")
    parser.add_argument("-w", "--width", type=int, default=7, help="number of columns")
    parser.add_argument("-r", "--height", type=int, default=7, help="number of rows")
    parser.add_argument("-b", "--boxes", type=int, default=1, help="number of boxes / targets")
    parser.add_argument("-l", "--levels", type=int, default=10, help="number of levels to generate")

    args = parser.parse_args()
    savedir, width, height, n_box, n_levels = args.dir, args.width, args.height, args.boxes, args.levels

    # Generate levels
    generate_dummy(width, height, n_box, n_levels, savedir)
