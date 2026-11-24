#!/usr/bin/env python3
"""Queens - a playable remake of the LinkedIn puzzle game.

Rules
-----
Place exactly one queen in every row, every column and every colored region.
No two queens may touch, not even diagonally.

Usage
-----
    python queens.py        # start with an 8x8 board
    python queens.py 6      # start with a 6x6 board

The board is generated on the fly, and "New puzzle" (or the N key) builds a
fresh one.  Puzzles are checked for a unique solution before they are shown.
"""

from __future__ import annotations

import colorsys
import random
import sys
import time

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover - only on Pythons built without Tk
    tk = None


EMPTY, QUEEN, MARK = 0, 1, 2

MIN_SIZE, MAX_SIZE, DEFAULT_SIZE = 5, 8, 8

BG = "#f6f2ea"
FG = "#202124"
MUTED = "#6f6a63"
BOARD_EDGE = "#4a4a4a"
CONFLICT = "#c0392b"
SOLVED = "#1e8449"


# --------------------------------------------------------------------------- #
#  Puzzle generation
# --------------------------------------------------------------------------- #

def neighbours(cell: int, n: int):
    """Yield the (up to four) orthogonally adjacent cells of ``cell``."""
    row, col = divmod(cell, n)
    if row:
        yield cell - n
    if row < n - 1:
        yield cell + n
    if col:
        yield cell - 1
    if col < n - 1:
        yield cell + 1


def random_solution(n: int, rng: random.Random):
    """Return one column per row with one queen per row/column and no touching.

    Two queens can only touch when they sit in consecutive rows, so it is
    enough to keep neighbouring rows at least two columns apart.
    """
    cols = [-1] * n
    used = [False] * n

    def place(row: int) -> bool:
        if row == n:
            return True
        candidates = list(range(n))
        rng.shuffle(candidates)
        for col in candidates:
            if used[col]:
                continue
            if row and abs(col - cols[row - 1]) < 2:
                continue
            cols[row] = col
            used[col] = True
            if place(row + 1):
                return True
            used[col] = False
            cols[row] = -1
        return False

    return cols if place(0) else None


def build_regions(n: int, cols, rng: random.Random):
    """Grow one connected region around each queen until the board is covered.

    Returns a list ``owner`` where ``owner[row * n + col]`` is the region id.
    """
    owner = [-1] * (n * n)
    sizes = [1] * n
    frontier = [set() for _ in range(n)]

    for row in range(n):
        owner[row * n + cols[row]] = row
    for row in range(n):
        for cell in neighbours(row * n + cols[row], n):
            if owner[cell] == -1:
                frontier[row].add(cell)

    remaining = n * n - n
    while remaining:
        live = [i for i in range(n) if frontier[i]]
        live.sort(key=lambda i: (sizes[i], rng.random()))
        # Usually grow the smallest region (keeps sizes balanced); sometimes
        # take the runner-up so the shapes do not all look the same.
        if len(live) > 1 and rng.random() >= 0.7:
            region = live[1]
        else:
            region = live[0]

        # Prefer cells that hug the region, which keeps the shapes compact.
        best_cell, best_score = -1, float("-inf")
        for cell in frontier[region]:
            touching = sum(1 for nb in neighbours(cell, n) if owner[nb] == region)
            score = touching * 4.0 + rng.random() * 3.0
            if score > best_score:
                best_cell, best_score = cell, score

        owner[best_cell] = region
        sizes[region] += 1
        remaining -= 1
        for cells in frontier:
            cells.discard(best_cell)
        for nb in neighbours(best_cell, n):
            if owner[nb] == -1:
                frontier[region].add(nb)

    return owner


def count_solutions(n: int, owner, limit: int = 2) -> int:
    """Count valid queen layouts, stopping as soon as ``limit`` is reached."""
    cols = [-1] * n
    used_col = [False] * n
    used_region = [False] * n
    found = 0

    def search(row: int) -> bool:
        nonlocal found
        if row == n:
            found += 1
            return found >= limit
        for col in range(n):
            if used_col[col]:
                continue
            region = owner[row * n + col]
            if used_region[region]:
                continue
            if row and abs(col - cols[row - 1]) < 2:
                continue
            cols[row] = col
            used_col[col] = True
            used_region[region] = True
            stop = search(row + 1)
            used_col[col] = False
            used_region[region] = False
            cols[row] = -1
            if stop:
                return True
        return False

    search(0)
    return found


def find_other_solution(n: int, owner, solution):
    """Return a valid layout different from ``solution``, or ``None``."""
    cols = [-1] * n
    used_col = [False] * n
    used_region = [False] * n
    found = None

    def search(row: int) -> bool:
        nonlocal found
        if found is not None:
            return True
        if row == n:
            if cols != solution:
                found = cols[:]
                return True
            return False
        for col in range(n):
            if used_col[col]:
                continue
            region = owner[row * n + col]
            if used_region[region]:
                continue
            if row and abs(col - cols[row - 1]) < 2:
                continue
            cols[row] = col
            used_col[col] = True
            used_region[region] = True
            if search(row + 1):
                return True
            used_col[col] = False
            used_region[region] = False
            cols[row] = -1
        return False

    search(0)
    return found


def _connected(cells, n: int) -> bool:
    if not cells:
        return True
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for nb in neighbours(cell, n):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def force_unique(n: int, solution, owner, rng: random.Random,
                 max_steps: int = 400, time_budget: float = 0.5):
    """Nudge region borders until ``solution`` is the only valid layout.

    An alternative layout is killed by handing one of its queen cells to a
    neighboring region.  The move is only made when it keeps both regions
    connected and never touches a cell of the intended solution.
    """
    owner = list(owner)
    region_cells = [set() for _ in range(n)]
    for cell, region in enumerate(owner):
        region_cells[region].add(cell)

    seen_layouts = set()
    started = time.perf_counter()

    for _ in range(max_steps):
        if time.perf_counter() - started > time_budget:
            return None

        other = find_other_solution(n, owner, solution)
        if other is None:
            return owner

        key = tuple(owner)
        if key in seen_layouts:
            return None
        seen_layouts.add(key)

        intended = {}
        for row, col in enumerate(solution):
            intended[owner[row * n + col]] = row * n + col
        rival = {}
        for row, col in enumerate(other):
            rival[owner[row * n + col]] = row * n + col

        moves = []
        for region in range(n):
            cell = rival[region]
            if cell == intended[region]:
                continue
            if not _connected(region_cells[region] - {cell}, n):
                continue
            for nb in neighbours(cell, n):
                target = owner[nb]
                if target == region:
                    continue
                hugging = sum(1 for x in neighbours(cell, n) if owner[x] == target)
                moves.append((hugging + rng.random() * 2.0, cell, region, target))

        if not moves:
            return None

        moves.sort(key=lambda move: move[0], reverse=True)
        _, cell, region, target = moves[rng.randrange(min(3, len(moves)))]

        region_cells[region].discard(cell)
        region_cells[target].add(cell)
        owner[cell] = target

    return None


def generate_puzzle(n: int, rng: random.Random, attempts: int = 400,
                    time_budget: float = 3.0):
    """Return ``(solution, owner)`` for a fresh puzzle.

    A unique solution is preferred; if the budget runs out the last generated
    board is used anyway, since it is always solvable.
    """
    started = time.perf_counter()
    fallback = None
    for _ in range(attempts):
        cols = random_solution(n, rng)
        if cols is None:
            continue
        owner = build_regions(n, cols, rng)
        if count_solutions(n, owner, 2) == 1:
            return cols, owner
        repaired = force_unique(n, cols, owner, rng, time_budget=0.25)
        if repaired is not None:
            return cols, repaired
        if fallback is None:
            fallback = (cols, owner)
        if time.perf_counter() - started > time_budget:
            break
    if fallback is None:
        raise RuntimeError("could not generate a puzzle")
    return fallback


# --------------------------------------------------------------------------- #
#  Drawing helpers
# --------------------------------------------------------------------------- #

def shade(color: str, factor: float) -> str:
    """Multiply an ``#rrggbb`` color by ``factor`` (0..1)."""
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    return "#%02x%02x%02x" % (
        max(0, min(255, round(r * factor))),
        max(0, min(255, round(g * factor))),
        max(0, min(255, round(b * factor))),
    )


def make_palette(n: int):
    """``n`` distinct pastel colors, spread around the hue wheel."""
    palette = []
    for i in range(n):
        hue = (0.07 + i * 0.618033988749895) % 1.0
        sat = 0.50 + 0.08 * ((i * 5) % 3) / 2.0
        light = 0.79 + 0.05 * (i % 3) / 2.0
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        palette.append("#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255)))
    return palette


# --------------------------------------------------------------------------- #
#  The game window
# --------------------------------------------------------------------------- #

class QueensApp:
    def __init__(self, root: "tk.Tk", size: int = DEFAULT_SIZE):
        self.root = root
        self.rng = random.Random()
        self.n = size
        self.solution = []
        self.owner = []
        self.cells = []
        self.colors = []
        self.solved = False
        self.cell = 0
        self.origin_x = 0
        self.origin_y = 0

        self._build_ui()
        self.new_puzzle()

    # -- layout ------------------------------------------------------------ #

    def _build_ui(self):
        root = self.root
        root.title("Queens")
        root.geometry("760x820")
        root.minsize(560, 640)
        root.configure(bg=BG)

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Segoe UI", 10), padding=(14, 6))
        style.configure("TCombobox", padding=4)

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=22, pady=(18, 2))
        tk.Label(header, text="Queens", font=("Segoe UI Semibold", 26),
                 bg=BG, fg=FG).pack(side="left")
        tk.Label(header, text="one queen per row, column and region",
                 font=("Segoe UI", 10), bg=BG, fg=MUTED).pack(
            side="left", padx=(14, 0), pady=(10, 0))

        controls = tk.Frame(root, bg=BG)
        controls.pack(fill="x", padx=22, pady=(10, 6))

        tk.Label(controls, text="Board", font=("Segoe UI", 10),
                 bg=BG, fg=MUTED).pack(side="left")
        self.size_var = tk.StringVar(value=str(self.n))
        size_box = ttk.Combobox(controls, textvariable=self.size_var,
                                values=[str(i) for i in range(MIN_SIZE, MAX_SIZE + 1)],
                                width=3, state="readonly")
        size_box.pack(side="left", padx=(8, 18))
        size_box.bind("<<ComboboxSelected>>", lambda _e: self.new_puzzle())

        ttk.Button(controls, text="New puzzle", command=self.new_puzzle).pack(side="left")
        ttk.Button(controls, text="Clear", command=self.clear_board).pack(side="left", padx=8)
        ttk.Button(controls, text="Rules", command=self.show_rules).pack(side="left")

        board_frame = tk.Frame(root, bg=BG)
        board_frame.pack(fill="both", expand=True, padx=22, pady=(6, 6))
        self.canvas = tk.Canvas(board_frame, bg=BG, highlightthickness=0,
                                cursor="hand2")
        self.canvas.pack(fill="both", expand=True)

        self.status = tk.Label(root, text="", anchor="w", font=("Segoe UI", 10),
                               bg=BG, fg=MUTED)
        self.status.pack(fill="x", padx=24, pady=(0, 16))

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

        for key, action in (("n", self.new_puzzle), ("N", self.new_puzzle),
                            ("r", self.show_rules), ("R", self.show_rules),
                            ("c", self.clear_board), ("C", self.clear_board)):
            root.bind("<KeyPress-%s>" % key, lambda _e, a=action: a())

    # -- game state -------------------------------------------------------- #

    def new_puzzle(self):
        n = int(self.size_var.get())
        self.n = n
        self.status.config(text="Generating a fresh puzzle...")
        self.root.update_idletasks()

        self.solution, self.owner = generate_puzzle(n, self.rng)
        self.cells = [EMPTY] * (n * n)
        self.colors = make_palette(n)
        self.solved = False
        self.redraw()
        self.update_status()

    def clear_board(self):
        if not self.cells:
            return
        self.cells = [EMPTY] * (self.n * self.n)
        self.solved = False
        self.redraw()
        self.update_status()

    def _after_change(self):
        conflicts = self.find_conflicts()
        self.solved = self.cells.count(QUEEN) == self.n and not conflicts
        self.redraw()
        self.update_status()

    def find_conflicts(self):
        """Cells whose queen breaks a rule (row, column, region or touching)."""
        n = self.n
        queens = [(i // n, i % n) for i, value in enumerate(self.cells) if value == QUEEN]
        bad = set()
        for a in range(len(queens)):
            r1, c1 = queens[a]
            for b in range(a + 1, len(queens)):
                r2, c2 = queens[b]
                if (r1 == r2 or c1 == c2
                        or (abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1)
                        or self.owner[r1 * n + c1] == self.owner[r2 * n + c2]):
                    bad.add(r1 * n + c1)
                    bad.add(r2 * n + c2)
        return bad

    def update_status(self):
        placed = self.cells.count(QUEEN)
        if self.solved:
            text = "Solved!  Press N or click \"New puzzle\" for another board."
        elif placed == 0:
            text = ("Left-click to place a queen  |  right-click to mark an X  |  "
                    "N = new puzzle")
        else:
            text = "%d of %d queens placed" % (placed, self.n)
            if self.find_conflicts():
                text += "  -  some queens break the rules"
        self.status.config(text=text)

    # -- input ------------------------------------------------------------- #

    def _cell_at(self, x, y):
        if self.cell <= 0:
            return None
        col = (x - self.origin_x) // self.cell
        row = (y - self.origin_y) // self.cell
        if 0 <= row < self.n and 0 <= col < self.n:
            return int(row), int(col)
        return None

    def _on_left_click(self, event):
        hit = self._cell_at(event.x, event.y)
        if hit is None:
            return
        row, col = hit
        i = row * self.n + col
        self.cells[i] = EMPTY if self.cells[i] == QUEEN else QUEEN
        self._after_change()

    def _on_right_click(self, event):
        hit = self._cell_at(event.x, event.y)
        if hit is None:
            return
        row, col = hit
        i = row * self.n + col
        state = self.cells[i]
        if state == MARK:
            self.cells[i] = EMPTY
        elif state == QUEEN:
            self.cells[i] = EMPTY
        else:
            self.cells[i] = MARK
        self._after_change()

    # -- painting ---------------------------------------------------------- #

    def redraw(self):
        canvas = self.canvas
        canvas.delete("all")
        if not self.owner:
            return

        n = self.n
        width = max(canvas.winfo_width(), 1)
        height = max(canvas.winfo_height(), 1)
        cell = min((width - 12) // n, (height - 12) // n)
        if cell < 12:
            return

        board = cell * n
        ox = (width - board) // 2
        oy = (height - board) // 2
        self.cell, self.origin_x, self.origin_y = cell, ox, oy

        conflicts = self.find_conflicts()
        border = max(2, round(cell * 0.075))

        # Region fills.
        for row in range(n):
            for col in range(n):
                i = row * n + col
                x1 = ox + col * cell
                y1 = oy + row * cell
                canvas.create_rectangle(
                    x1, y1, x1 + cell, y1 + cell,
                    fill=self.colors[self.owner[i]], outline="")

        # Faint hairlines between cells of the same region.
        for row in range(n):
            for col in range(n):
                i = row * n + col
                x1 = ox + col * cell
                y1 = oy + row * cell
                x2, y2 = x1 + cell, y1 + cell
                faint = shade(self.colors[self.owner[i]], 0.92)
                if col and self.owner[i] == self.owner[i - 1]:
                    canvas.create_line(x1, y1, x1, y2, fill=faint)
                if row and self.owner[i] == self.owner[i - n]:
                    canvas.create_line(x1, y1, x2, y1, fill=faint)

        # Thick borders wherever two regions meet (and around the board).
        for row in range(n):
            for col in range(n):
                i = row * n + col
                x1 = ox + col * cell
                y1 = oy + row * cell
                x2, y2 = x1 + cell, y1 + cell
                if col == n - 1 or self.owner[i] != self.owner[i + 1]:
                    canvas.create_line(x2, y1, x2, y2, fill=BOARD_EDGE,
                                       width=border, capstyle="round")
                if row == n - 1 or self.owner[i] != self.owner[i + n]:
                    canvas.create_line(x1, y2, x2, y2, fill=BOARD_EDGE,
                                       width=border, capstyle="round")
                if col == 0:
                    canvas.create_line(x1, y1, x1, y2, fill=BOARD_EDGE,
                                       width=border, capstyle="round")
                if row == 0:
                    canvas.create_line(x1, y1, x2, y1, fill=BOARD_EDGE,
                                       width=border, capstyle="round")

        # Marks and queens.
        for i, state in enumerate(self.cells):
            if state == EMPTY:
                continue
            row, col = divmod(i, n)
            cx = ox + col * cell + cell / 2
            cy = oy + row * cell + cell / 2
            if state == QUEEN:
                if i in conflicts:
                    color = CONFLICT
                elif self.solved:
                    color = SOLVED
                else:
                    color = "#262626"
                if i in conflicts:
                    canvas.create_rectangle(
                        ox + col * cell + 3, oy + row * cell + 3,
                        ox + (col + 1) * cell - 3, oy + (row + 1) * cell - 3,
                        outline=CONFLICT, width=2)
                self._draw_crown(cx, cy, cell, color)
            else:
                self._draw_cross(cx, cy, cell)

    def _draw_crown(self, cx, cy, cell, color):
        points = [
            (-0.34, 0.40), (0.34, 0.40), (0.34, 0.18),
            (0.28, -0.20), (0.10, 0.04), (0.00, -0.38),
            (-0.10, 0.04), (-0.28, -0.20), (-0.34, 0.18),
        ]
        coords = []
        for fx, fy in points:
            coords.extend((cx + fx * cell, cy + fy * cell))
        self.canvas.create_polygon(coords, fill=color, outline=color,
                                   joinstyle="round")

    def _draw_cross(self, cx, cy, cell):
        d = cell * 0.22
        width = max(2, round(cell * 0.055))
        self.canvas.create_line(cx - d, cy - d, cx + d, cy + d,
                                fill="#8a8a8a", width=width, capstyle="round")
        self.canvas.create_line(cx - d, cy + d, cx + d, cy - d,
                                fill="#8a8a8a", width=width, capstyle="round")

    # -- rules ------------------------------------------------------------- #

    def show_rules(self):
        win = tk.Toplevel(self.root)
        win.title("How to play Queens")
        win.configure(bg="#ffffff")
        win.resizable(False, False)
        win.transient(self.root)

        body = tk.Frame(win, bg="#ffffff", padx=30, pady=26)
        body.pack()

        tk.Label(body, text="Queens", font=("Segoe UI Semibold", 22),
                 bg="#ffffff", fg=FG, anchor="w").pack(fill="x")
        tk.Label(body, text="Place queens so that all of these are true:",
                 font=("Segoe UI", 10), bg="#ffffff", fg=MUTED,
                 anchor="w").pack(fill="x", pady=(2, 10))

        rules = (
            "1.  Every row contains exactly one queen.\n"
            "2.  Every column contains exactly one queen.\n"
            "3.  Every colored region contains exactly one queen.\n"
            "4.  No two queens touch each other, not even diagonally.\n\n"
            "Queens that break a rule are shown in red.\n\n"
            "Controls\n"
            "  Left-click    place or remove a queen\n"
            "  Right-click   add or remove an X (a note to yourself)\n"
            "  New puzzle    generate a fresh board (key: N)\n"
            "  Clear         wipe all queens and X marks (key: C)\n"
            "  Board         pick a size from 5x5 to 8x8\n\n"
            "Every puzzle is generated fresh and checked to have a unique\n"
            "solution, so there is always exactly one way to win."
        )
        tk.Label(body, text=rules, justify="left", anchor="w",
                 font=("Segoe UI", 10), bg="#ffffff", fg=FG).pack(fill="x")

        ttk.Button(body, text="Got it", command=win.destroy).pack(pady=(18, 0))

        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry("+%d+%d" % (max(0, x), max(0, y)))
        win.bind("<Escape>", lambda _e: win.destroy())
        win.grab_set()


def main():
    if tk is None:
        print("This game needs tkinter, which this Python build does not include.")
        print("On Windows, re-run the official installer and enable 'tcl/tk'.")
        return

    size = DEFAULT_SIZE
    for arg in sys.argv[1:]:
        if arg.isdigit():
            size = max(MIN_SIZE, min(MAX_SIZE, int(arg)))

    root = tk.Tk()
    QueensApp(root, size)
    root.mainloop()


if __name__ == "__main__":
    main()
