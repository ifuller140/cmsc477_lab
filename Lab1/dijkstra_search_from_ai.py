import csv
import glob
import heapq
import os
import sys
from typing import List, Tuple, Optional, Dict

# dijkstra_search_from_ai.py
# Finds best path from start (2) to finish (3) in a CSV grid.
# Uses Dijkstra where node-weight = cell value, and ties broken by fewer steps.


Coord = Tuple[int, int]

def load_grid(csv_path: Optional[str] = None) -> List[List[int]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Weighted_Map.csv')
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        grid = [[int(cell) for cell in row] for row in reader]
    return grid

def find_values(grid: List[List[int]], targets: List[int]) -> Dict[int, Coord]:
    found = {}
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val in targets:
                found[val] = (r, c)
                if len(found) == len(targets):
                    return found
    return found

def dijkstra(grid: List[List[int]]) -> Tuple[List[Coord], int]:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    vals = find_values(grid, [2, 3])
    if 2 not in vals or 3 not in vals:
        raise ValueError("Grid must contain a start (2) and finish (3).")
    start = vals[2]
    goal = vals[3]

    # priority queue: (total_cost, steps, r, c)
    pq = []
    heapq.heappush(pq, (0, 0, start[0], start[1]))
    dist: Dict[Coord, Tuple[int, int]] = {start: (0, 0)}  # coord -> (cost, steps)
    parent: Dict[Coord, Coord] = {}

    def neighbors(r: int, c: int):
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    while pq:
        cost, steps, r, c = heapq.heappop(pq)
        if (r, c) == goal:
            # reconstruct path
            path = []
            cur = goal
            while cur != start:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            path.reverse()
            return path, cost
        # stale entry check
        if dist.get((r, c), (1<<60, 1<<60)) < (cost, steps):
            continue
        for nr, nc in neighbors(r, c):
            cell_val = grid[nr][nc]
            # treat start/goal as zero-cost cells when entered
            enter_cost = 0 if cell_val in (2, 3) else cell_val
            new_cost = cost + enter_cost
            new_steps = steps + 1
            old = dist.get((nr, nc))
            if old is None or (new_cost, new_steps) < old:
                dist[(nr, nc)] = (new_cost, new_steps)
                parent[(nr, nc)] = (r, c)
                heapq.heappush(pq, (new_cost, new_steps, nr, nc))

    raise ValueError("No path from start to finish found.")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    grid = load_grid(csv_file)
    path, cost = dijkstra(grid)
    print("cost:", cost)
    print("path (r,c):")
    for p in path:
        print(p)