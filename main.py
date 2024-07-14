import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData

class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def find_nearest_node(tree, point):
    distances = [euclidean_distance(node.point, point) for node in tree]
    nearest_node_index = np.argmin(distances)
    return tree[nearest_node_index]

def generate_random_point(bounds):
    return [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

def rrt(start, goal, obstacles, points, max_iter=1000, step_size=0.1):
    tree = [Node(start)]
    bounds = [(min(points[:,0]), max(points[:,0])), (min(points[:,1]), max(points[:,1])), (min(points[:,2]), max(points[:,2]))]
    for _ in range(max_iter):
        random_point = generate_random_point(bounds)
        nearest_node = find_nearest_node(tree, random_point)
        new_point = np.clip(nearest_node.point + step_size * (random_point - nearest_node.point), 0, 10)
        if all(euclidean_distance(new_point, p) > step_size for p in obstacles[0]):  # Sprawdź kolizję z przeszkodą
            new_node = Node(new_point)
            new_node.parent = nearest_node
            tree.append(new_node)
            if euclidean_distance(new_point, goal) < step_size:
                final_node = Node(goal)
                final_node.parent = new_node
                tree.append(final_node)
                return tree
    return None

def prune_tree(tree, goal):
    pruned_tree = []
    current_node = tree[-1]  # Rozpoczynamy od ostatniego węzła (cela)
    while current_node is not None:
        pruned_tree.insert(0, current_node)  # Dodajemy węzeł na początek listy
        current_node = current_node.parent
    return pruned_tree

def visualize_rrt(tree, goal, points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Dodanie chmury punktów z pliku .ply
    ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o', label='Chmura punktów')
    if tree:
        for node in tree:
            if node.parent:
                ax.plot([node.point[0], node.parent.point[0]], [node.point[1], node.parent.point[1]], [node.point[2], node.parent.point[2]], c='b')
    ax.scatter(*goal, c='g', marker='o', label='Goal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def load_ply_file(filename):
    plydata = PlyData.read(filename)
    vertices = plydata['vertex']
    points = np.array([(vertex[0], vertex[1], vertex[2]) for vertex in vertices])
    return points

# Przykładowe dane wejściowe
start = (-5, 15, 1)
goal = (-18, 15, 1)

# Wczytanie chmury punktów z pliku .ply
points = load_ply_file('scena_test.PLY')
obstacles = [points]  # Zastąpienie przeszkód wczytaną chmurą punktów

# Znajdowanie trajektorii za pomocą algorytmu RRT
tree = rrt(start, goal, obstacles, points)

# Wizualizacja ścieżki po znalezieniu trajektorii
visualize_rrt(tree, goal, points)
