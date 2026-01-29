"""
A* Pathfinding Algorithm - Clean Implementation
Finds the optimal path through a 2D grid with obstacles using the A* search algorithm.

Core Concept:
- Explores nodes based on f(n) = g(n) + h(n)
- g(n): actual cost from start
- h(n): estimated cost to goal (heuristic)
- Guarantees shortest path when heuristic is admissible
"""

import math
import sys
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
ENABLE_VISUALIZATION = True  # Toggle real-time search visualization
PROGRAM_RUNNING = True  # Flag for clean program exit


def handle_keyboard_input(key_event):
    """
    Callback for keyboard events - allows clean exit via ESC key.
    
    Args:
        key_event: matplotlib keyboard event object
    """
    global PROGRAM_RUNNING
    if key_event.key == 'escape':
        print("\n[EXIT] User pressed ESC - Terminating gracefully...")
        PROGRAM_RUNNING = False
        plt.close('all')
        sys.exit(0)


class GridCell:
    """
    Represents a single location in the search space.
    
    Each cell tracks:
    - Its position in grid coordinates
    - Cost to reach it from the start (g_value)
    - Link to the previous cell in the path (predecessor)
    """
    
    def __init__(self, grid_x, grid_y, g_value, predecessor):
        """
        Initialize a grid cell.
        
        Args:
            grid_x: X-coordinate in grid space (integer index)
            grid_y: Y-coordinate in grid space (integer index)
            g_value: Accumulated cost from start node
            predecessor: Index of parent cell (-1 if this is the start)
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.g_value = g_value  # Actual cost from start
        self.predecessor = predecessor  # Parent cell for path reconstruction

    def __repr__(self):
        """String representation for debugging purposes."""
        return f"Cell({self.grid_x},{self.grid_y},g={self.g_value:.2f},parent={self.predecessor})"


class AStarPathfinder:
    """
    A* search algorithm implementation for 2D grid-based pathfinding.
    
    Converts continuous obstacle coordinates into a discrete occupancy grid,
    then searches for the optimal path using A* with Euclidean heuristic.
    """
    
    def __init__(self, obstacle_x_coords, obstacle_y_coords, cell_size, safety_radius):
        """
        Set up the pathfinding environment.
        
        Args:
            obstacle_x_coords: List of x-coordinates for obstacles [meters]
            obstacle_y_coords: List of y-coordinates for obstacles [meters]
            cell_size: Size of each grid cell [meters]
            safety_radius: Robot radius for obstacle inflation [meters]
        """
        # Store parameters
        self.cell_size = cell_size
        self.safety_radius = safety_radius
        
        # Map boundaries (will be computed from obstacles)
        self.boundary_min_x = 0
        self.boundary_min_y = 0
        self.boundary_max_x = 0
        self.boundary_max_y = 0
        
        # Grid dimensions (number of cells)
        self.num_cells_x = 0
        self.num_cells_y = 0
        
        # 2D occupancy grid: True = blocked, False = free
        self.occupancy_grid = None
        
        # Define how the robot can move between cells
        self.movement_options = self._generate_movement_model()
        
        # Build the occupancy grid from obstacle positions
        self._construct_occupancy_grid(obstacle_x_coords, obstacle_y_coords)
    
    def _generate_movement_model(self):
        """
        Define the 8-connected movement model (like a chess king).
        
        Returns:
            List of [dx, dy, cost] for each possible move:
            - Cardinal moves (up/down/left/right): cost = 1.0
            - Diagonal moves: cost = sqrt(2) ≈ 1.414
        """
        sqrt_two = math.sqrt(2)
        
        movements = [
            [1, 0, 1.0],      # East
            [0, 1, 1.0],      # North
            [-1, 0, 1.0],     # West
            [0, -1, 1.0],     # South
            [-1, -1, sqrt_two],  # Southwest
            [-1, 1, sqrt_two],   # Northwest
            [1, -1, sqrt_two],   # Southeast
            [1, 1, sqrt_two]     # Northeast
        ]
        return movements
    
    def _construct_occupancy_grid(self, obs_x, obs_y):
        """
        Build a 2D boolean grid representing free space vs obstacles.
        
        Process:
        1. Determine map boundaries from obstacle extents
        2. Create grid based on cell_size
        3. Mark cells as occupied if any obstacle is within safety_radius
        
        Args:
            obs_x: List of obstacle x-coordinates
            obs_y: List of obstacle y-coordinates
        """
        # ===== COMPUTE MAP BOUNDARIES =====
        self.boundary_min_x = round(min(obs_x))
        self.boundary_min_y = round(min(obs_y))
        self.boundary_max_x = round(max(obs_x))
        self.boundary_max_y = round(max(obs_y))
        
        print(f"[MAP] Boundaries: X=[{self.boundary_min_x}, {self.boundary_max_x}], "
              f"Y=[{self.boundary_min_y}, {self.boundary_max_y}]")
        
        # ===== CALCULATE GRID DIMENSIONS =====
        # Number of cells = span / cell_size
        self.num_cells_x = round((self.boundary_max_x - self.boundary_min_x) / self.cell_size)
        self.num_cells_y = round((self.boundary_max_y - self.boundary_min_y) / self.cell_size)
        
        print(f"[GRID] Dimensions: {self.num_cells_x} x {self.num_cells_y} cells")
        
        # ===== INITIALIZE OCCUPANCY GRID =====
        # Start with all cells free (False)
        self.occupancy_grid = [[False for _ in range(self.num_cells_y)]
                               for _ in range(self.num_cells_x)]
        
        # ===== MARK OCCUPIED CELLS =====
        # For each cell, check if any obstacle is within safety_radius
        for cell_x in range(self.num_cells_x):
            world_x = self._grid_to_world(cell_x, self.boundary_min_x)
            
            for cell_y in range(self.num_cells_y):
                world_y = self._grid_to_world(cell_y, self.boundary_min_y)
                
                # Check distance to all obstacles
                for obstacle_x, obstacle_y in zip(obs_x, obs_y):
                    distance = math.hypot(obstacle_x - world_x, obstacle_y - world_y)
                    
                    # If obstacle too close, mark cell as occupied
                    if distance <= self.safety_radius:
                        self.occupancy_grid[cell_x][cell_y] = True
                        break  # No need to check other obstacles
    
    def find_path(self, start_x, start_y, goal_x, goal_y):
        """
        Execute A* search to find optimal path from start to goal.
        
        Algorithm:
        1. Initialize open_set with start node
        2. Loop:
           - Pick best node from open_set (lowest f-value)
           - If it's the goal, reconstruct path and return
           - Expand neighbors, update costs
           - Move current to closed_set
        3. Return path coordinates
        
        Args:
            start_x, start_y: Starting position in world coordinates [meters]
            goal_x, goal_y: Goal position in world coordinates [meters]
        
        Returns:
            path_x, path_y: Lists of coordinates forming the path
        """
        # ===== CONVERT WORLD COORDS TO GRID INDICES =====
        start_cell = GridCell(
            self._world_to_grid(start_x, self.boundary_min_x),
            self._world_to_grid(start_y, self.boundary_min_y),
            0.0,  # Zero cost to reach start
            -1    # No predecessor
        )
        
        goal_cell = GridCell(
            self._world_to_grid(goal_x, self.boundary_min_x),
            self._world_to_grid(goal_y, self.boundary_min_y),
            0.0,
            -1
        )
        
        # ===== INITIALIZE SEARCH STRUCTURES =====
        # open_dict: nodes to explore (frontier)
        # closed_dict: nodes already explored
        # Key = unique grid index, Value = GridCell object
        open_dict = {}
        closed_dict = {}
        
        # Add start to open set
        start_index = self._compute_unique_index(start_cell)
        open_dict[start_index] = start_cell
        
        # For efficient visualization - batch collection
        explored_x_batch = []
        explored_y_batch = []
        
        # ===== MAIN A* SEARCH LOOP =====
        while True:
            # Check if user requested exit
            global PROGRAM_RUNNING
            if not PROGRAM_RUNNING:
                print("[SEARCH] Interrupted by user")
                return [], []
            
            # Check if search failed (no more nodes to explore)
            if len(open_dict) == 0:
                print("[SEARCH] Failed - no path exists!")
                break
            
            # ===== SELECT BEST NODE =====
            # Find node with minimum f(n) = g(n) + h(n)
            best_index = min(
                open_dict,
                key=lambda idx: open_dict[idx].g_value + 
                               self._compute_heuristic(goal_cell, open_dict[idx])
            )
            current_cell = open_dict[best_index]
            
            # ===== VISUALIZE EXPLORATION =====
            if ENABLE_VISUALIZATION:
                # Add to batch for plotting
                explored_x_batch.append(self._grid_to_world(current_cell.grid_x, self.boundary_min_x))
                explored_y_batch.append(self._grid_to_world(current_cell.grid_y, self.boundary_min_y))
                
                # Update display every 50 nodes (batch rendering)
                if len(closed_dict) % 50 == 0 and explored_x_batch:
                    plt.plot(explored_x_batch, explored_y_batch, "xc", markersize=2)
                    explored_x_batch.clear()
                    explored_y_batch.clear()
                    plt.pause(0.001)
            
            # ===== CHECK IF GOAL REACHED =====
            if current_cell.grid_x == goal_cell.grid_x and current_cell.grid_y == goal_cell.grid_y:
                print("[SEARCH] Goal found!")
                goal_cell.predecessor = current_cell.predecessor
                goal_cell.g_value = current_cell.g_value
                break
            
            # ===== MOVE TO CLOSED SET =====
            del open_dict[best_index]
            closed_dict[best_index] = current_cell
            
            # ===== EXPAND NEIGHBORS =====
            for movement in self.movement_options:
                # Calculate neighbor position
                neighbor_cell = GridCell(
                    current_cell.grid_x + movement[0],
                    current_cell.grid_y + movement[1],
                    current_cell.g_value + movement[2],  # Accumulated cost
                    best_index  # Parent is current cell
                )
                
                neighbor_index = self._compute_unique_index(neighbor_cell)
                
                # Skip if invalid (out of bounds or obstacle)
                if not self._is_valid_cell(neighbor_cell):
                    continue
                
                # Skip if already fully explored
                if neighbor_index in closed_dict:
                    continue
                
                # ===== UPDATE OR ADD NEIGHBOR =====
                if neighbor_index not in open_dict:
                    # New discovery - add to open set
                    open_dict[neighbor_index] = neighbor_cell
                else:
                    # Already in open set - update if we found cheaper path
                    if open_dict[neighbor_index].g_value > neighbor_cell.g_value:
                        open_dict[neighbor_index] = neighbor_cell
        
        # Plot any remaining visualization batch
        if ENABLE_VISUALIZATION and explored_x_batch:
            plt.plot(explored_x_batch, explored_y_batch, "xc", markersize=2)
        
        # ===== RECONSTRUCT PATH =====
        path_x, path_y = self._reconstruct_path(goal_cell, closed_dict)
        return path_x, path_y
    
    def _reconstruct_path(self, goal, closed_dict):
        """
        Trace back from goal to start using predecessor links.
        
        Args:
            goal: Goal GridCell with predecessor chain
            closed_dict: Dictionary of explored cells
        
        Returns:
            path_x, path_y: Lists of world coordinates forming path
        """
        # Start with goal position
        path_x = [self._grid_to_world(goal.grid_x, self.boundary_min_x)]
        path_y = [self._grid_to_world(goal.grid_y, self.boundary_min_y)]
        
        # Follow predecessor chain back to start
        current_predecessor = goal.predecessor
        while current_predecessor != -1:
            cell = closed_dict[current_predecessor]
            path_x.append(self._grid_to_world(cell.grid_x, self.boundary_min_x))
            path_y.append(self._grid_to_world(cell.grid_y, self.boundary_min_y))
            current_predecessor = cell.predecessor
        
        return path_x, path_y
    
    def _compute_heuristic(self, target, current):
        """
        Calculate h(n) - estimated cost from current to target.
        
        Uses Euclidean distance (straight-line distance) as heuristic.
        This is admissible because no path can be shorter than straight-line.
        
        Args:
            target: Target GridCell
            current: Current GridCell
        
        Returns:
            Heuristic cost estimate
        """
        heuristic_weight = 1.0  # Standard A* weight
        distance = math.hypot(target.grid_x - current.grid_x,
                            target.grid_y - current.grid_y)
        return heuristic_weight * distance
    
    def _is_valid_cell(self, cell):
        """
        Check if a cell is valid (within bounds and not an obstacle).
        
        Args:
            cell: GridCell to validate
        
        Returns:
            True if cell is traversable, False otherwise
        """
        # Convert to world coordinates for bounds checking
        world_x = self._grid_to_world(cell.grid_x, self.boundary_min_x)
        world_y = self._grid_to_world(cell.grid_y, self.boundary_min_y)
        
        # Check boundaries
        if world_x < self.boundary_min_x or world_x >= self.boundary_max_x:
            return False
        if world_y < self.boundary_min_y or world_y >= self.boundary_max_y:
            return False
        
        # Check occupancy grid
        if self.occupancy_grid[cell.grid_x][cell.grid_y]:
            return False
        
        return True
    
    def _compute_unique_index(self, cell):
        """
        Generate unique integer index for a grid cell.
        
        Uses row-major ordering: index = row * width + col
        
        Args:
            cell: GridCell to index
        
        Returns:
            Unique integer identifier
        """
        return (cell.grid_y - self.boundary_min_y) * self.num_cells_x + \
               (cell.grid_x - self.boundary_min_x)
    
    def _grid_to_world(self, grid_index, origin):
        """
        Convert grid index to world coordinate.
        
        Formula: world = grid_index * cell_size + origin
        
        Args:
            grid_index: Integer grid coordinate
            origin: World coordinate of grid origin
        
        Returns:
            World coordinate in meters
        """
        return grid_index * self.cell_size + origin
    
    def _world_to_grid(self, world_coord, origin):
        """
        Convert world coordinate to grid index.
        
        Formula: grid = round((world - origin) / cell_size)
        
        Args:
            world_coord: Position in meters
            origin: World coordinate of grid origin
        
        Returns:
            Integer grid index
        """
        return round((world_coord - origin) / self.cell_size)


def execute_pathfinding_demo():
    """
    Main demonstration of A* pathfinding.
    
    Creates a simple maze with boundary walls and interior obstacles,
    then searches for a path from bottom-left to top-right.
    """
    print("=" * 60)
    print("A* PATHFINDING DEMONSTRATION")
    print("=" * 60)
    
    # ===== DEFINE PROBLEM =====
    starting_position = (10.0, 10.0)  # Start coordinates [m]
    target_position = (50.0, 50.0)    # Goal coordinates [m]
    resolution = 2.0                  # Grid cell size [m]
    robot_size = 1.0                  # Robot radius for clearance [m]
    
    # ===== BUILD ENVIRONMENT =====
    obstacles_x = []
    obstacles_y = []
    
    # Create rectangular boundary walls
    # Bottom edge
    for x in range(-10, 60):
        obstacles_x.append(x)
        obstacles_y.append(-10.0)
    
    # Right edge
    for y in range(-10, 60):
        obstacles_x.append(60.0)
        obstacles_y.append(y)
    
    # Top edge
    for x in range(-10, 61):
        obstacles_x.append(x)
        obstacles_y.append(60.0)
    
    # Left edge
    for y in range(-10, 61):
        obstacles_x.append(-10.0)
        obstacles_y.append(y)
    
    # Interior wall #1: Vertical barrier at x=20
    for y in range(-10, 40):
        obstacles_x.append(20.0)
        obstacles_y.append(y)
    
    # Interior wall #2: Vertical barrier at x=40 (partial)
    for y in range(0, 40):
        obstacles_x.append(40.0)
        obstacles_y.append(60.0 - y)
    
    # ===== SETUP VISUALIZATION =====
    if ENABLE_VISUALIZATION:
        figure = plt.figure()
        figure.canvas.mpl_connect('key_press_event', handle_keyboard_input)
        
        # Plot environment
        plt.plot(obstacles_x, obstacles_y, ".k", markersize=2, label="Obstacles")
        plt.plot(starting_position[0], starting_position[1], "og", 
                markersize=10, label="Start")
        plt.plot(target_position[0], target_position[1], "xb", 
                markersize=12, linewidth=3, label="Goal")
        
        plt.grid(True)
        plt.axis("equal")
        plt.legend()
        plt.title("A* Pathfinding (Press ESC to exit)")
        plt.xlabel("X [meters]")
        plt.ylabel("Y [meters]")
    
    # ===== INITIALIZE PATHFINDER =====
    print("\n[INIT] Creating pathfinder...")
    pathfinder = AStarPathfinder(obstacles_x, obstacles_y, resolution, robot_size)
    
    # ===== EXECUTE SEARCH =====
    print("\n[SEARCH] Finding path... (Press ESC to abort)")
    result_x, result_y = pathfinder.find_path(
        starting_position[0], starting_position[1],
        target_position[0], target_position[1]
    )
    
    # ===== DISPLAY RESULTS =====
    if not result_x or not result_y:
        print("\n[RESULT] No path found or search interrupted!")
        if ENABLE_VISUALIZATION:
            plt.close('all')
        return
    
    # Calculate path metrics
    total_distance = sum([
        math.hypot(result_x[i+1] - result_x[i], result_y[i+1] - result_y[i])
        for i in range(len(result_x) - 1)
    ])
    
    print(f"\n[RESULT] Path found successfully!")
    print(f"  - Path length: {total_distance:.2f} meters")
    print(f"  - Waypoints: {len(result_x)}")
    
    # Visualize final path
    if ENABLE_VISUALIZATION:
        plt.plot(result_x, result_y, "-r", linewidth=2, label="A* Path")
        plt.legend()
        print("\n[INFO] Close window or press ESC to exit")
        plt.pause(0.5)
        plt.show()


# ===== PROGRAM ENTRY POINT =====
if __name__ == '__main__':
    execute_pathfinding_demo()