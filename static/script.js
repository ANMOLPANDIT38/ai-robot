document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const gridElement = document.getElementById('grid');
    const generateBtn = document.getElementById('generateGrid');
    const findPathBtn = document.getElementById('findPath');
    const resetBtn = document.getElementById('resetGrid');
    const gridWidthInput = document.getElementById('gridWidth');
    const gridHeightInput = document.getElementById('gridHeight');
    const algorithmSelect = document.getElementById('algorithm');
    const obstacleDensityInput = document.getElementById('obstacleDensity');
    const densityValue = document.getElementById('densityValue');
    const statsPanel = document.getElementById('stats');
    const pathLengthElement = document.getElementById('pathLength');
    const nodesExploredElement = document.getElementById('nodesExplored');
    const timeTakenElement = document.getElementById('timeTaken');

    // Grid state
    let grid = [];
    let gridWidth = 10;
    let gridHeight = 10;
    let startPos = null;
    let goalPos = null;
    let isDrawing = false;
    let isErasing = false;
    let currentAlgorithm = 'astar';

    // Initialize the grid
    function initializeGrid() {
        gridWidth = parseInt(gridWidthInput.value) || 10;
        gridHeight = parseInt(gridHeightInput.value) || 10;
        
        // Validate grid size
        gridWidth = Math.max(5, Math.min(20, gridWidth));
        gridHeight = Math.max(5, Math.min(20, gridHeight));
        
        gridWidthInput.value = gridWidth;
        gridHeightInput.value = gridHeight;

        // Clear the grid
        gridElement.innerHTML = '';
        gridElement.style.gridTemplateColumns = `repeat(${gridWidth}, 30px)`;
        
        // Initialize grid data structure
        grid = Array(gridHeight).fill().map(() => Array(gridWidth).fill(0));
        
        // Create grid cells
        for (let y = 0; y < gridHeight; y++) {
            for (let x = 0; x < gridWidth; x++) {
                const cell = document.createElement('div');
                cell.className = 'cell bg-white';
                cell.dataset.x = x;
                cell.dataset.y = y;
                
                cell.addEventListener('click', () => handleCellClick(x, y));
                cell.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                    handleCellRightClick(x, y);
                });
                cell.addEventListener('mousedown', () => {
                    isDrawing = true;
                    toggleObstacle(x, y);
                });
                cell.addEventListener('mouseenter', () => {
                    if (isDrawing) {
                        toggleObstacle(x, y);
                    }
                });
                cell.addEventListener('mouseup', () => {
                    isDrawing = false;
                });
                
                gridElement.appendChild(cell);
            }
        }
        
        // Reset start and goal positions
        startPos = null;
        goalPos = null;
        hideStats();
    }

    // Generate random obstacles
    function generateRandomObstacles() {
        const density = parseInt(obstacleDensityInput.value) / 100;
        const totalCells = gridWidth * gridHeight;
        const obstacleCount = Math.floor(totalCells * density);
        
        // Clear existing obstacles
        for (let y = 0; y < gridHeight; y++) {
            for (let x = 0; x < gridWidth; x++) {
                if (grid[y][x] === 1) {
                    grid[y][x] = 0;
                    updateCellAppearance(x, y);
                }
            }
        }
        
        // Place random obstacles
        let placed = 0;
        while (placed < obstacleCount) {
            const x = Math.floor(Math.random() * gridWidth);
            const y = Math.floor(Math.random() * gridHeight);
            
            // Don't overwrite start or goal
            if ((startPos && x === startPos.x && y === startPos.y) || 
                (goalPos && x === goalPos.x && y === goalPos.y)) {
                continue;
            }
            
            if (grid[y][x] === 0) {
                grid[y][x] = 1;
                updateCellAppearance(x, y);
                placed++;
            }
        }
    }

    // Handle cell click (place start)
    function handleCellClick(x, y) {
        // If we're drawing, skip
        if (isDrawing) return;
        
        // If cell is an obstacle, skip
        if (grid[y][x] === 1) return;
        
        // Remove previous start
        if (startPos) {
            updateCellAppearance(startPos.x, startPos.y);
        }
        
        // Set new start
        startPos = { x, y };
        updateCellAppearance(x, y);
    }

    // Handle cell right click (place goal)
    function handleCellRightClick(x, y) {
        // If cell is an obstacle, skip
        if (grid[y][x] === 1) return;
        
        // Remove previous goal
        if (goalPos) {
            updateCellAppearance(goalPos.x, goalPos.y);
        }
        
        // Set new goal
        goalPos = { x, y };
        updateCellAppearance(x, y);
    }

    // Toggle obstacle
    function toggleObstacle(x, y) {
        // Don't allow placing obstacles on start or goal
        if ((startPos && x === startPos.x && y === startPos.y) || 
            (goalPos && x === goalPos.x && y === goalPos.y)) {
            return;
        }
        
        grid[y][x] = grid[y][x] === 1 ? 0 : 1;
        updateCellAppearance(x, y);
    }

    // Update cell appearance based on its state
    function updateCellAppearance(x, y) {
        const cell = document.querySelector(`.cell[data-x="${x}"][data-y="${y}"]`);
        if (!cell) return;
        
        // Reset classes
        cell.className = 'cell';
        cell.textContent = '';
        
        // Add appropriate classes
        if (grid[y][x] === 1) {
            cell.classList.add('obstacle');
        } else if (startPos && x === startPos.x && y === startPos.y) {
            cell.classList.add('start');
            cell.textContent = 'S';
        } else if (goalPos && x === goalPos.x && y === goalPos.y) {
            cell.classList.add('goal');
            cell.textContent = 'G';
        } else {
            cell.classList.add('bg-white');
        }
    }

    // Update clearPathVisualization
function clearPathVisualization() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => {
        cell.classList.remove(
            'visited', 'path', 'considered', 'shortest-path', 
            'robot', 'person', 'rescued'
        );
        // Remove all alternate path classes
        for (let i = 0; i < 4; i++) {
            cell.classList.remove(`alternate-path-${i}`);
        }
    });
}

    // Add these new functions
function sortPathsByLength(paths) {
    return paths.sort((a, b) => a.length - b.length);
}

// Update visualizePaths function
async function visualizePaths(sortedPaths, visited) {
    // Show visited cells first
    for (const node of visited) {
        const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
            cell.classList.add('visited');
            await new Promise(resolve => setTimeout(resolve, 5));
        }
    }
    
    // Visualize shortest path first with robot animation
    if (sortedPaths.length > 0) {
        const shortestPath = sortedPaths[0];
        await animateRobotMovement(shortestPath);
        
        // Then show the path visualization
        for (const node of shortestPath) {
            const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
            if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                cell.classList.add('path', 'shortest-path');
            }
        }
    }
    
    // Visualize alternate paths with different colors
    for (let i = 1; i < sortedPaths.length; i++) {
        const path = sortedPaths[i];
        const pathClass = `alternate-path-${(i-1) % 4}`; // Cycle through 4 colors
        
        for (const node of path) {
            const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
            if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal') && 
                !cell.classList.contains('shortest-path')) {
                cell.classList.add(pathClass);
                await new Promise(resolve => setTimeout(resolve, 50));
            }
        }
    }
}

    // Visualize the pathfinding process
    async function visualizePathfinding(visited, path) {
        // Clear any previous visualization first
        clearPathVisualization();
        
        // Show visited cells with animation
        for (const node of visited) {
            const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
            if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                cell.classList.add('visited');
                await new Promise(resolve => setTimeout(resolve, 5)); // Faster animation
            }
        }
        
        // Show final path with different animation
        if (path && path.length > 0) {
            for (const node of path) {
                const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
                if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                    cell.classList.add('path');
                    await new Promise(resolve => setTimeout(resolve, 50)); // Slower for path
                }
            }
        }
    }

    // Update density value display
    function updateDensityValue() {
        densityValue.textContent = `${obstacleDensityInput.value}%`;
    }

    // Update showStats to show path count
function showStats(pathLength, nodesExplored, timeTaken, algorithm, pathCount = 1) {
    pathLengthElement.textContent = pathLength;
    nodesExploredElement.textContent = nodesExplored;
    timeTakenElement.textContent = timeTaken;
    algorithmUsed.textContent = algorithm;
    
    // Add path count display
    const pathCountElement = document.getElementById('pathCount') || 
        document.createElement('div');
    pathCountElement.id = 'pathCount';
    pathCountElement.className = 'flex justify-between';
    pathCountElement.innerHTML = `
        <span class="text-gray-700">Paths Found:</span>
        <span class="font-medium">${pathCount}</span>
    `;
    
    if (!document.getElementById('pathCount')) {
        statsPanel.querySelector('div').appendChild(pathCountElement);
    } else {
        document.getElementById('pathCount').innerHTML = pathCountElement.innerHTML;
    }
    
    statsPanel.classList.remove('hidden');
}

async function animateRobotMovement(path) {
    if (!path || path.length === 0) return;
    
    // Clear any existing robot class
    document.querySelectorAll('.cell.robot').forEach(cell => {
        cell.classList.remove('robot');
    });
    
    // Add person to goal cell
    const goalCell = document.querySelector(`.cell[data-x="${path[path.length-1].x}"][data-y="${path[path.length-1].y}"]`);
    if (goalCell) {
        goalCell.classList.add('person');
    }
    
    // Animate robot through each step of the path
    for (let i = 0; i < path.length; i++) {
        const node = path[i];
        const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
        
        if (cell) {
            // Remove robot from previous cell
            if (i > 0) {
                const prevCell = document.querySelector(`.cell[data-x="${path[i-1].x}"][data-y="${path[i-1].y}"]`);
                prevCell.classList.remove('robot');
            }
            
            // Add robot to current cell
            cell.classList.add('robot');
            
            // If this is the goal cell, show rescue effect
            if (i === path.length - 1) {
                await new Promise(resolve => setTimeout(resolve, 500));
                cell.classList.remove('person', 'robot');
                cell.classList.add('rescued');
            } else {
                await new Promise(resolve => setTimeout(resolve, 300));
            }
        }
    }
}

    // Hide stats
    function hideStats() {
        statsPanel.classList.add('hidden');
    }

    
// Find path using selected algorithm
// Update findPath function
async function findPath() {
    if (!startPos || !goalPos) {
        alert('Please set both start and goal positions');
        return;
    }
    
    clearPathVisualization();
    currentAlgorithm = algorithmSelect.value;
    
    const gridData = {
        grid: grid,
        start: startPos,
        goal: goalPos,
        algorithm: currentAlgorithm,
        findMultiplePaths: true
    };
    
    try {
        const startTime = performance.now();
        const response = await fetch('/api/find-path', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(gridData)
        });

        
        const result = await response.json();
        const endTime = performance.now();
        const timeTaken = Math.round(endTime - startTime);
        
        if (result.success) {
            if (result.paths && result.paths.length > 0) {
                const sortedPaths = sortPathsByLength(result.paths);
                await visualizePaths(sortedPaths, result.visited);
                
                showStats(
                    sortedPaths[0].length, 
                    result.visited.length, 
                    timeTaken, 
                    `${currentAlgorithm} (${result.paths.length} paths)`,
                    sortedPaths.length
                );
            } else {
                await visualizePathfinding(result.visited, result.path);
                showStats(
                    result.path.length, 
                    result.visited.length, 
                    timeTaken, 
                    currentAlgorithm,
                    1
                );
            }
        } else {
            alert(result.message || 'Pathfinding failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to find path. Please check console for details.');
    }
}

// Add this function to visualize multiple paths
async function visualizeMultiplePaths(paths, visited) {
    // Show visited cells
    for (const node of visited) {
        const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
        if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
            cell.classList.add('visited');
        }
        await new Promise(resolve => setTimeout(resolve, 10));
    }
    
    // Show each path with different colors
    if (paths && paths.length > 0) {
        for (let i = 0; i < paths.length; i++) {
            const path = paths[i];
            const pathClass = `path-${(i % 5) + 1}`; // Cycle through 5 colors
            
            for (const node of path) {
                const cell = document.querySelector(`.cell[data-x="${node.x}"][data-y="${node.y}"]`);
                if (cell && !cell.classList.contains('start') && !cell.classList.contains('goal')) {
                    cell.classList.add(pathClass);
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
            }
        }
    }
}

    // Event listeners
    generateBtn.addEventListener('click', () => {
        initializeGrid();
        generateRandomObstacles();
    });
    
    findPathBtn.addEventListener('click', findPath);
    
    resetBtn.addEventListener('click', () => {
        initializeGrid();
        hideStats();
    });
    
    obstacleDensityInput.addEventListener('input', updateDensityValue);
    
    // Initialize on load
    initializeGrid();
    updateDensityValue();
    
    // Prevent context menu on right-click
    document.addEventListener('contextmenu', (e) => {
        if (e.target.classList.contains('cell')) {
            e.preventDefault();
        }
    });
});
