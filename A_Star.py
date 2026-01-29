import math
import matplotlib
import matplotlib.pyplot as plt

# Flag to enable/disable visualization during pathfinding
show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for A* planning
        
        This constructor sets up the entire search space by converting continuous
        obstacle positions into a discrete grid map for efficient collision checking.

        ox: x position list of Obstacles [m] - List of obstacle x-coordinates in world frame
        oy: y position list of Obstacles [m] - List of obstacle y-coordinates in world frame
        resolution: grid resolution [m] - Size of each grid cell (e.g., 2.0m means 2m x 2m cells)
        rr: robot radius [m] - Safety buffer around obstacles (inflates obstacles by this amount)
        """

        # Store grid resolution (cell size in meters)
        self.resolution = resolution
        
        # Store robot radius for obstacle inflation
        self.rr = rr
        
        # Initialize boundary variables (will be calculated from obstacles)
        self.min_x, self.min_y = 0, 0  # Minimum x,y coordinates of the map
        self.max_x, self.max_y = 0, 0  # Maximum x,y coordinates of the map
        
        # Will store 2D boolean array: True = obstacle, False = free space
        self.obstacle_map = None
        
        # Number of grid cells in x and y directions
        self.x_width, self.y_width = 0, 0
        
        # Get the 8-connected motion model (how robot can move between cells)
        self.motion = self.get_motion_model()
        
        # Build the obstacle map from obstacle positions
        self.calc_obstacle_map(ox, oy)

    class Node:
        """
        Represents a single cell in the search grid.
        Stores position, cost, and parent information for path reconstruction.
        """
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # Grid x-index (not world coordinates)
            self.y = y  # Grid y-index (not world coordinates)
            self.cost = cost  # g(n): Accumulated cost from start to this node
            self.parent_index = parent_index  # Index of parent node in closed_set (-1 for start)

        def __str__(self):
            # String representation for debugging
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A* path search algorithm
        
        Uses f(n) = g(n) + h(n) to find optimal path:
        - g(n): actual cost from start
        - h(n): estimated cost to goal (heuristic)
        - f(n): total estimated cost through this node

        input:
            sx: start x position [m] - Starting x-coordinate in world frame
            sy: start y position [m] - Starting y-coordinate in world frame
            gx: goal x position [m] - Goal x-coordinate in world frame
            gy: goal y position [m] - Goal y-coordinate in world frame

        output:
            rx: x position list of the final path - List of x-coordinates from goal to start
            ry: y position list of the final path - List of y-coordinates from goal to start
        """

        # Convert world coordinates to grid indices and create start node
        # cost = 0.0 (no cost to reach start), parent_index = -1 (no parent)
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        
        # Create goal node (cost and parent will be updated when reached)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        # Initialize open and closed sets as dictionaries for O(1) lookup
        # open_set: nodes discovered but not yet explored (frontier)
        # closed_set: nodes already explored
        open_set, closed_set = dict(), dict()
        
        # Add start node to open set using its grid index as key
        open_set[self.calc_grid_index(start_node)] = start_node

        # Main A* search loop
        while True:
            # Check if search failed (no path exists)
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            # Select node with minimum f(n) = g(n) + h(n) from open set
            # This is the core of A*: pick most promising node to explore next
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            # Get the actual node object
            current = open_set[c_id]

            # Visualization: plot current node being explored
            if show_animation:  # pragma: no cover
                # Plot cyan 'x' marker at current node position
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                
                # Allow user to press ESC to exit simulation
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                
                # Update plot every 10 nodes to avoid slowdown
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.1)

            # Goal check: reached destination?
            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                # Store parent and cost information in goal node for path reconstruction
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove current node from open set (we're done considering it)
            del open_set[c_id]

            # Add current node to closed set (mark as explored)
            closed_set[c_id] = current

            # Expand search: check all 8 neighboring cells
            for i, _ in enumerate(self.motion):
                # Create neighbor node by applying motion model
                # new position = current position + motion offset
                # new cost = current cost + motion cost
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                
                # Calculate unique grid index for this neighbor
                n_id = self.calc_grid_index(node)

                # Skip if node is invalid (out of bounds or obstacle)
                if not self.verify_node(node):
                    continue

                # Skip if already explored (in closed set)
                if n_id in closed_set:
                    continue

                # Add to open set if newly discovered
                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    # If we've seen this node before, check if this path is better
                    if open_set[n_id].cost > node.cost:
                        # This path is cheaper, update the node
                        open_set[n_id] = node

        # Reconstruct path by following parent pointers from goal to start
        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        """
        Reconstruct the final path by backtracking through parent pointers.
        
        Starts at goal and follows parent_index links back to start.
        Returns path from goal to start (reverse order).
        """
        # Initialize path with goal position (convert grid index to world coordinates)
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        
        # Get parent of goal node
        parent_index = goal_node.parent_index
        
        # Follow parent pointers until reaching start (parent_index = -1)
        while parent_index != -1:
            # Get parent node from closed set
            n = closed_set[parent_index]
            
            # Add parent's world coordinates to path
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            
            # Move to next parent
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        """
        Calculate heuristic cost h(n) from node n2 to node n1.
        
        Uses Euclidean distance as heuristic. This is admissible (never overestimates)
        because straight-line distance is the shortest possible path.
        
        w: weight of heuristic (1.0 = standard A*, >1.0 = weighted A* for faster but suboptimal)
        """
        w = 1.0  # weight of heuristic
        # math.hypot calculates sqrt(dx^2 + dy^2) - Euclidean distance
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        Convert grid index to world coordinate position.
        
        Formula: world_pos = grid_index * cell_size + map_origin
        
        :param index: Grid cell index (integer)
        :param min_position: Origin of the map (minimum coordinate)
        :return: World coordinate position in meters
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        """
        Convert world coordinate to grid index.
        
        Formula: grid_index = round((world_pos - map_origin) / cell_size)
        
        :param position: World coordinate in meters
        :param min_pos: Origin of the map (minimum coordinate)
        :return: Grid cell index (integer)
        """
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        """
        Calculate unique 1D index for a node's 2D grid position.
        
        Uses row-major ordering: index = row * width + col
        This allows dictionary lookup of nodes by their grid position.
        
        :param node: Node object with x, y grid coordinates
        :return: Unique integer index for this grid cell
        """
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        """
        Check if a node is valid (within bounds and not colliding with obstacles).
        
        Returns True if node is safe to visit, False otherwise.
        """
        # Convert grid indices to world coordinates for boundary checking
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        # Check if node is outside left boundary
        if px < self.min_x:
            return False
        # Check if node is outside bottom boundary
        elif py < self.min_y:
            return False
        # Check if node is outside right boundary
        elif px >= self.max_x:
            return False
        # Check if node is outside top boundary
        elif py >= self.max_y:
            return False

        # Check for collision with obstacles
        # obstacle_map[x][y] = True means occupied
        if self.obstacle_map[node.x][node.y]:
            return False

        # Node is valid if we get here
        return True

    def calc_obstacle_map(self, ox, oy):
        """
        Build a 2D occupancy grid from obstacle point cloud.
        
        For each grid cell, checks if any obstacle point is within robot radius.
        This "inflates" obstacles so robot center stays safe distance away.
        """

        # Find boundaries of the map from obstacle positions
        self.min_x = round(min(ox))  # Leftmost obstacle
        self.min_y = round(min(oy))  # Bottommost obstacle
        self.max_x = round(max(ox))  # Rightmost obstacle
        self.max_y = round(max(oy))  # Topmost obstacle
        
        # Print map boundaries for debugging
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        # Calculate number of grid cells in each direction
        # Formula: num_cells = (max - min) / cell_size
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        
        # Print grid dimensions for debugging
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # Initialize 2D occupancy grid (all False = free space)
        # Structure: obstacle_map[x_index][y_index] = True/False
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        
        # For each grid cell, check if any obstacle is within robot radius
        for ix in range(self.x_width):
            # Convert grid x-index to world x-coordinate
            x = self.calc_grid_position(ix, self.min_x)
            
            for iy in range(self.y_width):
                # Convert grid y-index to world y-coordinate
                y = self.calc_grid_position(iy, self.min_y)
                
                # Check distance to each obstacle point
                for iox, ioy in zip(ox, oy):
                    # Calculate Euclidean distance to this obstacle
                    d = math.hypot(iox - x, ioy - y)
                    
                    # If obstacle is within robot radius, mark cell as occupied
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break  # No need to check other obstacles for this cell

    @staticmethod
    def get_motion_model():
        """
        Define how the robot can move between grid cells.
        
        Returns 8-connected movement model (like chess king):
        - 4 cardinal directions (up, down, left, right): cost = 1.0
        - 4 diagonal directions: cost = sqrt(2) ≈ 1.414 (actual Euclidean distance)
        
        Format: [dx, dy, cost]
        dx, dy: change in grid position
        cost: movement cost for this action
        """
        motion = [[1, 0, 1],              # Move right
                  [0, 1, 1],              # Move up
                  [-1, 0, 1],             # Move left
                  [0, -1, 1],             # Move down
                  [-1, -1, math.sqrt(2)], # Move down-left (diagonal)
                  [-1, 1, math.sqrt(2)],  # Move up-left (diagonal)
                  [1, -1, math.sqrt(2)],  # Move down-right (diagonal)
                  [1, 1, math.sqrt(2)]]   # Move up-right (diagonal)

        return motion


def main():
    """
    Main function to demonstrate A* path planning.
    
    Sets up a simple maze environment with:
    - Boundary walls forming outer rectangle
    - Two interior vertical walls creating a challenging path
    - Start point at (10, 10)
    - Goal point at (50, 50)
    """
    print(__file__ + " start!!")

    # Define start and goal positions in world coordinates [meters]
    sx = 10.0  # Start x position [m]
    sy = 10.0  # Start y position [m]
    gx = 50.0  # Goal x position [m]
    gy = 50.0  # Goal y position [m]
    
    # Grid and robot parameters
    grid_size = 2.0  # [m] - Each grid cell is 2m x 2m
    robot_radius = 1.0  # [m] - Safety buffer around obstacles

    # Build obstacle list (walls and barriers)
    ox, oy = [], []  # Initialize empty lists for obstacle x, y coordinates
    
    # Bottom boundary wall: horizontal line from x=-10 to x=60 at y=-10
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    
    # Right boundary wall: vertical line from y=-10 to y=60 at x=60
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    
    # Top boundary wall: horizontal line from x=-10 to x=60 at y=60
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    
    # Left boundary wall: vertical line from y=-10 to y=60 at x=-10
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    
    # Interior vertical wall #1: at x=20, from y=-10 to y=40
    # This forces robot to navigate around it
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    
    # Interior vertical wall #2: at x=40, from y=60 down to y=20
    # Creates a maze-like environment with the first wall
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    # Setup visualization if animation is enabled
    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")  # Plot obstacles as black dots
        plt.plot(sx, sy, "og")  # Plot start as green circle
        plt.plot(gx, gy, "xb")  # Plot goal as blue X
        plt.grid(True)          # Show grid lines
        plt.axis("equal")       # Equal aspect ratio (square cells)

    # Create A* planner instance (builds obstacle map)
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    
    # Run A* algorithm to find path from start to goal
    rx, ry = a_star.planning(sx, sy, gx, gy)

    # Plot the resulting path if animation is enabled
    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")  # Plot path as red line
        plt.pause(0.1)          # Brief pause to render
        plt.show()              # Display final plot

# Standard Python idiom: only run main() if script is executed directly
# (not when imported as a module)
if __name__ == '__main__':
    main()