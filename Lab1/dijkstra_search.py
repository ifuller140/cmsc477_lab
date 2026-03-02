import csv
import os
import time

class Point:
    def __init__(self):
        self.dist_from_start = float('inf')
        self.cost = 0
        self.trips = float('inf')
        self.visited = False
        
def print_map(points):
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            point = points[(r, c)]
            if point.dist_from_start == float('inf'):
                print("  ∞  ", end="")
            else:
                print(f"{int(point.dist_from_start):4}", end=" ")
        print()
    print("\n" + "-" * 30 + "\n")
            

def define_grid():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Weighted_Map.csv')
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
    for i in range(3):
        if (x-1, y-1+i) in points:
            neighbors.append(points[(x-1,y-1 + i)])
    if (x, y-1) in points:            
        neighbors.append(points[(x, y-1)])
    if (x, y+1) in points:
        neighbors.append(points[(x, y+1)])
    for i in range(3):
        if (x+1, y-1+i) in points:
            neighbors.append(points[(x+1,y-1 + i)])
    return neighbors

def dijkstra():
    start = find_value(2)
    fin = find_value(3)


    points = {}
    unvisited = []
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            points[(r,c)] = Point()
            points[(r,c)].cost = val
            unvisited.append(points[(r,c)])

    points[start[0], start[1]].dist_from_start = 0
    points[start[0], start[1]].trips = 0

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

        print(current)
        for n in find_neighbors(curr_x, curr_y, points):
            if n.visited == False:
                if n.dist_from_start > curr_dist + n.cost:
                    n.dist_from_start = curr_dist + n.cost
                elif n.dist_from_start == curr_dist + n.cost:
                    if n.trips > curr_trips + 1:
                        n.trips = curr_trips + 1

    print_map(points)



        

if __name__ == "__main__":
    grid = define_grid()
    dijkstra()

