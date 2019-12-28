import argparse
from tkinter import Canvas, Tk, Button
from constants import SokobanElements


LEVEL_CHARS = {
    0: SokobanElements.EMPTY,
    1: SokobanElements.WALL,
    2: SokobanElements.BOX,
    3: SokobanElements.PLAYER,
    4: SokobanElements.TARGET
}


class Cell():
    N_STATES = 5
    BORDER_COLOR = "black"
    WALL_COLOR = "black"
    EMPTY_COLOR = "white"
    BOX_COLOR = "orange"
    PLAYER_COLOR = "blue"
    TARGET_COLOR = "green"
    COLORS = {
        0: EMPTY_COLOR,
        1: WALL_COLOR,
        2: BOX_COLOR,
        3: PLAYER_COLOR,
        4: TARGET_COLOR
    }

    def __init__(self, master, x, y, size):
        self.master = master
        self.x = x
        self.y = y
        self.size = size
        self.state = 0

    def _switch(self):
        self.state = (self.state + 1) % Cell.N_STATES

    def draw(self):
        if self.master != None:
            state_color = Cell.COLORS[self.state]
            xmin = self.x * self.size
            xmax = xmin + self.size
            ymin = self.y * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(
                xmin, ymin, xmax, ymax, 
                fill=state_color, outline=Cell.BORDER_COLOR
            )


class CellGrid(Canvas):
    def __init__(self, master, n_row, n_col, cell_size, path, *args, **kwargs):
        Canvas.__init__(
            self, master, 
            width=cell_size * n_col,
            height=cell_size * n_row,
            *args, **kwargs
        )
        self.path = path  # Path to save the level
        self.cell_size = cell_size
        self.grid = []

        for row in range(n_row):
            line = []

            for col in range(n_col):
                line.append(
                    Cell(self, col, row, cell_size)
                )

            self.grid.append(line)

        # memorize the cells that have been modified to avoid many switching of state during mouse motion.
        self.switched = []

        # bind click action
        self.bind("<Button-1>", self.handleMouseClick) 

        # bind moving while clicking
        self.bind("<B1-Motion>", self.handleMouseMotion)

        # bind release button action - clear the memory of midified cells.
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())

        self.draw()

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()
        
    def _event_coords(self, event):
        row = int(event.y / self.cell_size)
        col = int(event.x / self.cell_size)
        return row, col

    def handleMouseClick(self, event):
        row, col = self._event_coords(event)
        cell = self.grid[row][col]
        cell._switch()
        cell.draw()

        # Add the cell to the list of switched cells list
        self.switched.append(cell)

    def handleMouseMotion(self, event):
        row, column = self._event_coords(event)
        cell = self.grid[row][column]

        if cell not in self.switched:
            cell._switch()
            cell.draw()
            self.switched.append(cell)

    def save(self):
        """Save the built level in an ASCII string."""
        with open(self.path, 'w') as f:
            for line in self.grid:
                for cell in line:
                    f.write(LEVEL_CHARS[cell.state])
                f.write("\n")
                    

def build_level(path, width, height, size):
    """Launch the tkinter level builder and save it."""
    app = Tk()

    # Add the grid
    grid = CellGrid(app, width, height, size, path)
    grid.pack()

    # Add a save button
    save_button = Button(app, text="Save", command=grid.save)
    save_button.pack()

    app.mainloop()


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()

    parser.add_argument("filename", type=str, help="name of the level")
    parser.add_argument("-d", "--dir", type=str, help="output dir to save the level", default="./levels/")
    parser.add_argument("-w", "--width", type=int, default=7, help="number of columns")
    parser.add_argument("-r", "--height", type=int, default=7, help="number of rows")
    parser.add_argument("-s", "--size", type=int, default=50, help="cell size")

    args = parser.parse_args()

    path = args.dir + args.filename
    width, height, size = args.width, args.height, args.size

    # Launch the level builder
    build_level(path, width, height, size)
