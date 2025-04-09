from flask import Flask, request, jsonify
from flask_cors import CORS
import heapq
from collections import deque
import random

app = Flask(__name__)
CORS(app)

class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic (estimated cost from current to goal)
        self.f = g + h  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f < other.f or (self.f == other.f and self.h < other.h)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

def heuristic(a, b):
    # Manhattan distance
    return abs(a.x - b.x) + abs(a.y - b.y)

def reconstruct_path(current_node):
    path = []
    while current_node:
        path.append({'x': current_node.x, 'y': current_node.y})
        current_node = current_node.parent
    return path[::-1]

def a_star(grid, start, goal):
    if not grid or not grid[0]:
        return None, []
    
    rows = len(grid)
    cols = len(grid[0])
    
    open_set = []
    heapq.heappush(open_set, start)
    closed_set = set()
    visited = []
    
    while open_set:
        current = heapq.heappop(open_set)
        visited.append({'x': current.x, 'y': current.y})
        
        if current.x == goal.x and current.y == goal.y:
            return reconstruct_path(current), visited
        
        if (current.x, current.y) in closed_set:
            continue
            
        closed_set.add((current.x, current.y))
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-directional movement
            x, y = current.x + dx, current.y + dy
            
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0:
                neighbor = Node(x, y, current.g + 1, heuristic(Node(x, y), goal), current)
                
                if (neighbor.x, neighbor.y) in closed_set:
                    continue
                
                # Check if neighbor is in open set
                found = False
                for node in open_set:
                    if node == neighbor:
                        found = True
                        if neighbor.f < node.f:
                            node.f = neighbor.f
                            node.g = neighbor.g
                            node.parent = neighbor.parent
                            heapq.heapify(open_set)
                        break
                
                if not found:
                    heapq.heappush(open_set, neighbor)
    
    return None, visited

def ao_star(grid, start, goal):
    if not grid or not grid[0]:
        return None, []
    
    rows = len(grid)
    cols = len(grid[0])
    
    open_set = []
    heapq.heappush(open_set, start)
    closed_set = set()
    visited = []
    
    while open_set:
        current = heapq.heappop(open_set)
        visited.append({'x': current.x, 'y': current.y})
        
        if current.x == goal.x and current.y == goal.y:
            return reconstruct_path(current), visited
        
        if (current.x, current.y) in closed_set:
            continue
            
        closed_set.add((current.x, current.y))
        
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = current.x + dx, current.y + dy
            
            if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0:
                neighbors.append((x, y))
        
        # Sort neighbors based on heuristic to goal
        neighbors.sort(key=lambda pos: heuristic(Node(pos[0], pos[1]), goal))
        
        for x, y in neighbors:
            neighbor = Node(x, y, current.g + 1, heuristic(Node(x, y), goal), current)
            
            if (neighbor.x, neighbor.y) in closed_set:
                continue
                
            found = False
            for node in open_set:
                if node == neighbor:
                    found = True
                    if neighbor.f < node.f:
                        node.f = neighbor.f
                        node.g = neighbor.g
                        node.parent = neighbor.parent
                        heapq.heapify(open_set)
                    break
            
            if not found:
                heapq.heappush(open_set, neighbor)
    
    return None, visited

def find_multiple_paths(grid, start, goal, algorithm, max_alternates=3):
    # Find primary path
    if algorithm == 'aostar':
        primary_path, visited = ao_star(grid, start, goal)
    else:
        primary_path, visited = a_star(grid, start, goal)
    
    if not primary_path:
        return None, visited
    
    paths = [primary_path]
    grid_copy = [row.copy() for row in grid]
    
    # Find alternative paths
    alternate_paths = []
    attempts = 0
    max_attempts = 20  # Prevent infinite loops
    
    while len(alternate_paths) < max_alternates and attempts < max_attempts:
        attempts += 1
        temp_grid = [row.copy() for row in grid_copy]
        
        # Block a random node from existing paths (but not start/goal)
        all_path_nodes = set()
        for path in paths + alternate_paths:
            for node in path[1:-1]:  # Exclude start and end
                all_path_nodes.add((node['x'], node['y']))
        
        if not all_path_nodes:
            break
            
        block_x, block_y = random.choice(list(all_path_nodes))
        temp_grid[block_y][block_x] = 1
        
        # Find new path
        if algorithm == 'aostar':
            new_path, _ = ao_star(temp_grid, start, goal)
        else:
            new_path, _ = a_star(temp_grid, start, goal)
        
        if new_path and not any(p == new_path for p in paths + alternate_paths):
            alternate_paths.append(new_path)
    
    # Sort alternate paths by length and select the shortest alternates
    alternate_paths.sort(key=lambda p: len(p))
    paths.extend(alternate_paths[:max_alternates])
    
    return paths, visited

@app.route('/api/find-path', methods=['POST'])
def find_path():
    try:
        data = request.get_json()
        
        # Validation checks
        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400
            
        if 'grid' not in data or 'start' not in data or 'goal' not in data:
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
            
        if not isinstance(data['grid'], list) or not all(isinstance(row, list) for row in data['grid']):
            return jsonify({'success': False, 'message': 'Invalid grid format'}), 400
            
        try:
            start_x = int(data['start']['x'])
            start_y = int(data['start']['y'])
            goal_x = int(data['goal']['x'])
            goal_y = int(data['goal']['y'])
        except (KeyError, TypeError, ValueError):
            return jsonify({'success': False, 'message': 'Invalid coordinates format'}), 400
            
        # Find path(s)
        if data.get('findMultiplePaths', False):
            paths, visited = find_multiple_paths(
                data['grid'],
                Node(start_x, start_y),
                Node(goal_x, goal_y),
                data.get('algorithm', 'astar'),
                max_alternates=3
            )
            
            if paths:
                return jsonify({
                    'success': True,
                    'paths': paths,
                    'visited': visited,
                    'message': f'Found {len(paths)} paths'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No path exists between start and goal',
                    'visited': visited
                })
        else:
            # Single path finding
            if data.get('algorithm') == 'aostar':
                path, visited = ao_star(data['grid'], 
                                      Node(start_x, start_y), 
                                      Node(goal_x, goal_y))
            else:
                path, visited = a_star(data['grid'], 
                                     Node(start_x, start_y), 
                                     Node(goal_x, goal_y))
                
            if path:
                return jsonify({
                    'success': True,
                    'path': path,
                    'visited': visited
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No path exists between start and goal',
                    'visited': visited
                })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)