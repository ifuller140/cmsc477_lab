import csv
import os

class Point:
    def __init__(self):
        self.dist_from_start = float('inf')
        self.cost = 0
        self.trips = float('inf')
        self.visited = False
        self.previous = None
        
# def draw_map_dist(points):
#     for r, row in enumerate(grid):
#         for c, val in enumerate(row):
#             point = points[(r, c)]
#             if point.dist_from_start == float('inf'):
#                 print("  ∞  ", end="")
#             else:
#                 print(f"{int(point.dist_from_start):4}", end=" ")
#         print()
#     print("\n" + "-" * 30 + "\n")

# def draw_map_trips(points):
#     for r, row in enumerate(grid):
#         for c, val in enumerate(row):
#             point = points[(r, c)]
#             if point.trips == float('inf'):
#                 print("  ∞  ", end="")
#             else:
#                 print(f"{int(point.trips):4}", end=" ")
#         print()
#     print("\n" + "-" * 30 + "\n")
            

def define_grid():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Weighted_Map_2.csv')
    with open(file_path) as f:
        reader = csv.reader(f)
        grid = [[int(cell) for cell in row] for row in reader]
    return grid 

def find_value(target_val):
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            if val == target_val:
                return [r, c]

def find_neighbors(x,y, points):
    neighbors = []
    if (x-1, y) in points:
        neighbors.append(points[(x-1,y)])
    if (x, y-1) in points:            
        neighbors.append(points[(x, y-1)])
    if (x, y+1) in points:
        neighbors.append(points[(x, y+1)])
    if (x+1, y) in points:
        neighbors.append(points[(x+1,y)])
    return neighbors

def make_path(points, start, fin):
    path = []
    curr = fin
    path.append(curr)
    while True:
        if curr == start:
            break
        path.append(points[curr[0], curr[1]].previous)
        curr = points[curr[0], curr[1]].previous
    path.reverse()
    path_rebased = [(r+1, c+1) for r, c in path]
    return(path_rebased)

def redefine_coords(grid, path):
    col_num = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
             col_num.append(c)
    col_num = int(len(col_num)/len(row))
    path_redefined = [(col_num - 4 - (r-1), c - 4) for r, c in path]
    total_cell_length = 0.266
    specific_cell_length = total_cell_length/3
    path_mm = [(c*specific_cell_length, r*specific_cell_length, ) for r, c in path_redefined]
    return path_mm
    


def dijkstra():
    start = tuple(find_value(2))
    fin = tuple(find_value(3))


    points = {}
    unvisited = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            points[(r,c)] = Point()
            points[(r,c)].cost = val
            unvisited.append(points[(r,c)])

    points[start].dist_from_start = 0
    points[start].trips = 0
    points[start].cost = 0
    points[fin].cost = 0

    while True:
        points_unvisited = []

        for p in points:
            if points[p].visited == False:
                points_unvisited.append(p)

        if not points_unvisited:
                break   

        current = min(points_unvisited, key=lambda p: points[p].dist_from_start)
        
        if points[current].dist_from_start == float('inf'):
                        break

        points[current].visited = True
        curr_x, curr_y = current
        curr_dist = points[current].dist_from_start
        curr_trips = points[current].trips

        for n in find_neighbors(curr_x, curr_y, points):
            if n.visited == False:
                if n.dist_from_start > curr_dist + n.cost:
                    n.dist_from_start = curr_dist + n.cost
                    n.trips = curr_trips + 1
                    n.previous = current
                elif n.dist_from_start == curr_dist + n.cost:
                    if n.trips > curr_trips + 1:
                        n.trips = curr_trips + 1
                        n.previous = current
    # draw_map_dist(points)
    # draw_map_trips(points)
    path = make_path(points, start, fin)
    return path


        

if __name__ == "__main__":
    grid = define_grid()
    path = dijkstra()
    path_mm = redefine_coords(grid, path)
    print(path_mm)


