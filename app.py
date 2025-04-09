from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import heapq
from collections import deque
import random

app = Flask(
    __name__,
    static_folder='static',
    template_folder='templates'
)
CORS(app)


class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return (self.f, self.h) < (other.f, other.h)

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


def heuristic(a: Node, b: Node) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def reconstruct_path(node: Node):
    path = []
    while node:
        path.append({'x': node.x, 'y': node.y})
        node = node.parent
    return path[::-1]


def a_star(grid, start: Node, goal: Node):
    if not grid or not grid[0]:
        return None, []

    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, start)
    closed = set()
    visited = []

    while open_set:
        current = heapq.heappop(open_set)
        visited.append({'x': current.x, 'y': current.y})

        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current), visited

        if (current.x, current.y) in closed:
            continue
        closed.add((current.x, current.y))

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0:
                h = heuristic(Node(x,y), goal)
                neighbor = Node(x, y, current.g + 1, h, current)
                if (neighbor.x, neighbor.y) in closed:
                    continue

                # if in open_set with worse f, update it
                for node in open_set:
                    if node == neighbor:
                        if neighbor.f < node.f:
                            node.g, node.h, node.f, node.parent = neighbor.g, neighbor.h, neighbor.f, neighbor.parent
                            heapq.heapify(open_set)
                        break
                else:
                    heapq.heappush(open_set, neighbor)

    return None, visited


def ao_star(grid, start: Node, goal: Node):
    if not grid or not grid[0]:
        return None, []

    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, start)
    closed = set()
    visited = []

    while open_set:
        current = heapq.heappop(open_set)
        visited.append({'x': current.x, 'y': current.y})

        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current), visited

        if (current.x, current.y) in closed:
            continue
        closed.add((current.x, current.y))

        # generate and sort neighbors by heuristic
        nbrs = []
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0:
                nbrs.append((x,y))
        nbrs.sort(key=lambda pos: heuristic(Node(*pos), goal))

        for x, y in nbrs:
            neighbor = Node(x, y, current.g + 1, heuristic(Node(x, y), goal), current)
            if (neighbor.x, neighbor.y) in closed:
                continue

            for node in open_set:
                if node == neighbor:
                    if neighbor.f < node.f:
                        node.g, node.h, node.f, node.parent = neighbor.g, neighbor.h, neighbor.f, neighbor.parent
                        heapq.heapify(open_set)
                    break
            else:
                heapq.heappush(open_set, neighbor)

    return None, visited


def dijkstra(grid, start: Node, goal: Node):
    # A* with zero heuristic
    if not grid or not grid[0]:
        return None, []

    rows, cols = len(grid), len(grid[0])
    open_set = []
    start.h = 0
    start.f = 0
    heapq.heappush(open_set, start)
    closed = set()
    visited = []

    while open_set:
        current = heapq.heappop(open_set)
        visited.append({'x': current.x, 'y': current.y})

        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current), visited

        if (current.x, current.y) in closed:
            continue
        closed.add((current.x, current.y))

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0:
                neighbor = Node(x, y, current.g + 1, 0, current)
                if (neighbor.x, neighbor.y) in closed:
                    continue

                for node in open_set:
                    if node == neighbor:
                        if neighbor.g < node.g:
                            node.g = neighbor.g
                            node.f = neighbor.g
                            node.parent = neighbor.parent
                            heapq.heapify(open_set)
                        break
                else:
                    heapq.heappush(open_set, neighbor)

    return None, visited


def bfs(grid, start: Node, goal: Node):
    if not grid or not grid[0]:
        return None, []

    rows, cols = len(grid), len(grid[0])
    queue = deque([start])
    visited_set = {(start.x, start.y)}
    visited = []

    while queue:
        current = queue.popleft()
        visited.append({'x': current.x, 'y': current.y})

        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current), visited

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0 and (x,y) not in visited_set:
                neighbor = Node(x, y, current.g + 1, 0, current)
                visited_set.add((x, y))
                queue.append(neighbor)

    return None, visited


def dfs(grid, start: Node, goal: Node):
    if not grid or not grid[0]:
        return None, []

    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited_set = {(start.x, start.y)}
    visited = []

    while stack:
        current = stack.pop()
        visited.append({'x': current.x, 'y': current.y})

        if (current.x, current.y) == (goal.x, goal.y):
            return reconstruct_path(current), visited

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0 and (x,y) not in visited_set:
                neighbor = Node(x, y, current.g + 1, 0, current)
                visited_set.add((x, y))
                stack.append(neighbor)

    return None, visited


def find_multiple_paths(grid, start: Node, goal: Node, algorithm: str, max_alternates=3):
    # pick primary
    alg = algorithm.lower()
    funcs = {
        'astar': a_star,
        'aostar': ao_star,
        'dijkstra': dijkstra,
        'bfs': bfs,
        'dfs': dfs
    }
    fn = funcs.get(alg, a_star)
    primary_path, visited = fn(grid, start, goal)
    if not primary_path:
        return None, visited

    paths = [primary_path]
    grid_copy = [row.copy() for row in grid]
    alternates = []
    attempts = 0

    while len(alternates) < max_alternates and attempts < 20:
        attempts += 1
        temp = [row.copy() for row in grid_copy]
        all_nodes = {
            (n['x'], n['y'])
            for path in paths + alternates
            for n in path[1:-1]
        }
        if not all_nodes:
            break

        bx, by = random.choice(list(all_nodes))
        temp[by][bx] = 1  # block one node

        new_path, _ = fn(temp, start, goal)
        if new_path and not any(p == new_path for p in paths + alternates):
            alternates.append(new_path)

    alternates.sort(key=len)
    paths.extend(alternates[:max_alternates])
    return paths, visited


@app.route('/')
def home():
    return render_template('index.html')


# serve CSS & JS at root so your <link href="styles.css"> still works
@app.route('/styles.css')
def styles():
    return app.send_static_file('styles.css')


@app.route('/script.js')
def script():
    return app.send_static_file('script.js')


@app.route('/api/find-path', methods=['POST'])
def find_path():
    data = request.get_json()
    if not data:
        return jsonify(success=False, message='No data received'), 400
    if 'grid' not in data or 'start' not in data or 'goal' not in data:
        return jsonify(success=False, message='Missing required fields'), 400
    if not isinstance(data['grid'], list) or not all(isinstance(r, list) for r in data['grid']):
        return jsonify(success=False, message='Invalid grid format'), 400

    try:
        sx, sy = int(data['start']['x']), int(data['start']['y'])
        gx, gy = int(data['goal']['x']), int(data['goal']['y'])
    except (KeyError, TypeError, ValueError):
        return jsonify(success=False, message='Invalid coordinates'), 400

    alg = data.get('algorithm', 'astar').lower()
    find_multi = bool(data.get('findMultiplePaths', False))

    start = Node(sx, sy)
    goal = Node(gx, gy)

    try:
        if find_multi:
            paths, visited = find_multiple_paths(data['grid'], start, goal, alg)
            if paths:
                return jsonify(success=True, paths=paths, visited=visited, message=f'Found {len(paths)} paths')
            else:
                return jsonify(success=False, message='No path exists', visited=visited)
        else:
            funcs = {
                'astar': a_star,
                'aostar': ao_star,
                'dijkstra': dijkstra,
                'bfs': bfs,
                'dfs': dfs
            }
            fn = funcs.get(alg)
            if not fn:
                return jsonify(success=False, message='Unknown algorithm'), 400

            path, visited = fn(data['grid'], start, goal)
            if path:
                return jsonify(success=True, path=path, visited=visited)
            else:
                return jsonify(success=False, message='No path exists', visited=visited)
    except Exception as e:
        return jsonify(success=False, message=f'Server error: {e}'), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
