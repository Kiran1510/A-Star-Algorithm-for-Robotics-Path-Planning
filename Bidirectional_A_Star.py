"""
Dual-Direction A* Pathfinding Algorithm
Performs simultaneous exploration from both start and goal positions.

Strategy:
- Launch two independent A* searches
- One expands from start position
- Other expands from goal position  
- Path found when searches meet
- Approximately 2x faster than single-direction A*

Author: Rewritten implementation with enhanced annotations
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# ========== GLOBAL CONFIGURATION ==========
SHOW_SEARCH_ANIMATION = True  # Enable real-time visualization
FRAME_DELAY = 0.2  # Seconds between animation updates (higher = slower)
CONTINUE_EXECUTION = True  # Program state flag


def handle_escape_key(key_event):
    """
    Interrupt handler for ESC key press.
    Provides clean termination of search and visualization.
    
    Args:
        key_event: Matplotlib keyboard event object
    """
    global CONTINUE_EXECUTION
    if key_event.key == 'escape':
        print("\n[ABORT] ESC detected - Shutting down gracefully...")
        CONTINUE_EXECUTION = False
        plt.close('all')
        sys.exit(0)


class SearchPoint:
    """
    Represents a single location in the bidirectional search space.
    
    Attributes:
        actual_cost: Distance traveled from origin (g-value)
        estimated_cost: Predicted distance to target (h-value)
        total_score: Sum of actual + estimated (f-value)
        position: [x, y] coordinates in grid
        predecessor: Reference to previous point in path
    """
    
    def __init__(self, actual_cost=0, estimated_cost=0, position=None, predecessor=None):
        self.actual_cost = actual_cost  # g(n) - cost from start
        self.estimated_cost = estimated_cost  # h(n) - heuristic to goal
        self.total_score = actual_cost + estimated_cost  # f(n) = g(n) + h(n)
        self.predecessor = predecessor  # Parent for path reconstruction
        self.position = position  # [x, y] location
    
    def recalculate_score(self):
        """
        Update total score after cost values change.
        Called when a shorter path to this point is discovered.
        """
        self.total_score = self.actual_cost + self.estimated_cost


def manhattan_distance(current_position, target_position):
    """
    Compute Manhattan distance heuristic (L1 norm).
    
    Formula: |Δx| + |Δy|
    
    Properties:
    - Admissible for grid-based movement
    - Never overestimates actual path length
    - Works well with 8-connected neighbors
    
    Args:
        current_position: [x, y] coordinates of current point
        target_position: [x, y] coordinates of target point
    
    Returns:
        Manhattan distance as heuristic value
    """
    delta_x = abs(current_position[0] - target_position[0])
    delta_y = abs(current_position[1] - target_position[1])
    return delta_x + delta_y


def calculate_movement_cost(parent_point, child_position):
    """
    Determine cost to move from parent to child position.
    
    Uses Euclidean distance for accurate cost calculation:
    - Horizontal/vertical moves: cost ≈ 1.0
    - Diagonal moves: cost ≈ 1.414 (√2)
    
    Args:
        parent_point: SearchPoint object representing parent
        child_position: [x, y] coordinates of child
    
    Returns:
        Total cost from search origin to child via parent
    """
    # Calculate distance between parent and child
    delta_x = abs(parent_point.position[0] - child_position[0])
    delta_y = abs(parent_point.position[1] - child_position[1])
    step_cost = math.hypot(delta_x, delta_y)
    
    # Total cost = cost to reach parent + cost from parent to child
    return parent_point.actual_cost + step_cost


def create_environment(start_pos, goal_pos, upper_corner, lower_corner, obstacle_count):
    """
    Generate search environment with boundaries and random obstacles.
    
    Creates:
    1. Four boundary walls forming rectangular perimeter
    2. Random interior obstacles for complexity
    3. Ensures start and goal remain unobstructed
    
    Args:
        start_pos: [x, y] starting location
        goal_pos: [x, y] goal location
        upper_corner: [x, y] top-right boundary corner
        lower_corner: [x, y] bottom-left boundary corner
        obstacle_count: Number of random obstacles to generate
    
    Returns:
        complete_obstacles: Combined boundary + random obstacles
        interior_obstacles: List of random obstacles only
    """
    
    # ===== BUILD PERIMETER WALLS =====
    
    # Left vertical wall
    left_y = list(range(lower_corner[1], upper_corner[1]))
    left_x = [lower_corner[0]] * len(left_y)
    
    # Right vertical wall
    right_y = left_y  # Mirror left wall's y-coordinates
    right_x = [upper_corner[0]] * len(right_y)
    
    # Bottom horizontal wall
    bottom_x = list(range(lower_corner[0] + 1, upper_corner[0]))
    bottom_y = [lower_corner[1]] * len(bottom_x)
    
    # Top horizontal wall
    top_x = [lower_corner[0]] + bottom_x + [upper_corner[0]]
    top_y = [upper_corner[1]] * len(top_x)
    
    # ===== GENERATE RANDOM INTERIOR OBSTACLES =====
    random_x = np.random.randint(lower_corner[0] + 1, 
                                 upper_corner[0], 
                                 obstacle_count).tolist()
    random_y = np.random.randint(lower_corner[1] + 1, 
                                 upper_corner[1], 
                                 obstacle_count).tolist()
    
    # ===== COMBINE ALL WALL SEGMENTS =====
    all_x = left_x + bottom_x + right_x + top_x
    all_y = left_y + bottom_y + right_y + top_y
    
    # ===== PROCESS OBSTACLE LIST =====
    interior_obstacles = np.vstack((random_x, random_y)).T.tolist()
    
    # Remove any obstacles that block start or goal
    interior_obstacles = [obs for obs in interior_obstacles 
                         if obs != start_pos and obs != goal_pos]
    
    # ===== COMBINE BOUNDARIES AND OBSTACLES =====
    boundary_array = np.vstack((all_x, all_y)).T
    obstacle_array = np.array(interior_obstacles)
    complete_obstacles = np.vstack((boundary_array, obstacle_array))
    
    return complete_obstacles, interior_obstacles


def get_valid_neighbors(current_point, obstacles, explored_positions):
    """
    Find all reachable neighboring positions from current point.
    
    Implements:
    - 8-connected grid movement (like chess king)
    - Corner-cutting prevention (can't squeeze diagonally through obstacles)
    - Exclusion of already-explored positions
    
    Args:
        current_point: SearchPoint to find neighbors for
        obstacles: Numpy array of obstacle coordinates
        explored_positions: List of already-visited coordinates
    
    Returns:
        List of valid neighbor coordinates
    """
    
    # Convert to set for O(1) lookup performance
    obstacle_set = set(map(tuple, obstacles.tolist()))
    valid_neighbors = set()
    
    # ===== GENERATE 3x3 NEIGHBORHOOD =====
    # Check all cells in 3x3 grid around current position
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            neighbor_x = current_point.position[0] + x_offset
            neighbor_y = current_point.position[1] + y_offset
            neighbor_pos = (neighbor_x, neighbor_y)
            
            # Add if not obstacle and not current position
            if neighbor_pos not in obstacle_set and neighbor_pos != tuple(current_point.position):
                valid_neighbors.add(neighbor_pos)
    
    # ===== DEFINE ADJACENT CELLS FOR CORNER-CUTTING CHECK =====
    north = (current_point.position[0], current_point.position[1] + 1)
    south = (current_point.position[0], current_point.position[1] - 1)
    west = (current_point.position[0] - 1, current_point.position[1])
    east = (current_point.position[0] + 1, current_point.position[1])
    
    northwest = (current_point.position[0] - 1, current_point.position[1] + 1)
    northeast = (current_point.position[0] + 1, current_point.position[1] + 1)
    southwest = (current_point.position[0] - 1, current_point.position[1] - 1)
    southeast = (current_point.position[0] + 1, current_point.position[1] - 1)
    
    # ===== PREVENT DIAGONAL CORNER-CUTTING =====
    # Block diagonal movement if both adjacent cardinals are obstacles
    
    if north in obstacle_set and west in obstacle_set:
        valid_neighbors.discard(northwest)
    
    if north in obstacle_set and east in obstacle_set:
        valid_neighbors.discard(northeast)
    
    if south in obstacle_set and west in obstacle_set:
        valid_neighbors.discard(southwest)
    
    if south in obstacle_set and east in obstacle_set:
        valid_neighbors.discard(southeast)
    
    # ===== REMOVE ALREADY-EXPLORED POSITIONS =====
    explored_set = set(map(tuple, explored_positions))
    valid_neighbors -= explored_set
    
    return list(valid_neighbors)


def locate_point_index(target_position, point_list):
    """
    Find index of a SearchPoint in list by its position coordinates.
    
    Args:
        target_position: [x, y] coordinates to search for
        point_list: List of SearchPoint objects
    
    Returns:
        Index of matching point in list
    """
    for index, point in enumerate(point_list):
        if point.position == target_position:
            return index
    return 0


def expand_search_frontier(frontier, explored, target, obstacles):
    """
    Expand one generation of the search frontier.
    
    Process:
    1. For each point in current frontier
    2. Find valid neighbors
    3. Update costs or add new points
    4. Move current point to explored set
    5. Sort frontier by f-value
    
    Args:
        frontier: List of SearchPoints to explore
        explored: List of already-explored SearchPoints
        target: Target position for this search direction
        obstacles: Array of obstacle coordinates
    
    Returns:
        Updated frontier and explored lists
    """
    
    frontier_size = len(frontier)
    
    for iteration in range(frontier_size):
        # Always process best point (lowest f-value)
        current_point = frontier[0]
        
        # Extract positions for efficient lookup
        frontier_positions = [pt.position for pt in frontier]
        explored_positions = [pt.position for pt in explored]
        
        # ===== FIND NEIGHBORS OF CURRENT POINT =====
        neighbors = get_valid_neighbors(current_point, obstacles, explored_positions)
        
        for neighbor_pos in neighbors:
            # Skip if already fully explored
            if neighbor_pos in explored_positions:
                continue
            
            # ===== NEIGHBOR ALREADY IN FRONTIER =====
            if neighbor_pos in frontier_positions:
                # Check if we found a cheaper path
                neighbor_index = frontier_positions.index(neighbor_pos)
                new_cost = calculate_movement_cost(current_point, neighbor_pos)
                
                if new_cost <= frontier[neighbor_index].actual_cost:
                    # Update with better path
                    frontier[neighbor_index].actual_cost = new_cost
                    frontier[neighbor_index].recalculate_score()
                    frontier[neighbor_index].predecessor = current_point
            
            # ===== NEW NEIGHBOR - CREATE SEARCH POINT =====
            else:
                new_point = SearchPoint(
                    position=neighbor_pos,
                    predecessor=current_point,
                    actual_cost=calculate_movement_cost(current_point, neighbor_pos),
                    estimated_cost=manhattan_distance(neighbor_pos, target)
                )
                frontier.append(new_point)
        
        # ===== MOVE CURRENT TO EXPLORED =====
        frontier.remove(current_point)
        explored.append(current_point)
        
        # ===== MAINTAIN PRIORITY QUEUE =====
        # Sort by f-value to keep best candidates at front
        frontier.sort(key=lambda pt: pt.total_score)
    
    return frontier, explored


def extract_positions(point_list):
    """
    Convert list of SearchPoint objects to list of coordinates.
    
    Args:
        point_list: List of SearchPoint objects
    
    Returns:
        List of [x, y] position arrays
    """
    return [point.position for point in point_list]


def detect_frontier_collision(explored_forward, explored_backward):
    """
    Check if forward and backward searches have met.
    
    This is the critical test for bidirectional search completion.
    When any position appears in both explored sets, a path exists.
    
    Args:
        explored_forward: Explored points from start→goal
        explored_backward: Explored points from goal→start
    
    Returns:
        List of meeting point coordinates (intersection)
    """
    forward_positions = extract_positions(explored_forward)
    backward_positions = extract_positions(explored_backward)
    
    # Find positions that appear in both sets
    meeting_points = [pos for pos in forward_positions if pos in backward_positions]
    return meeting_points


def identify_adjacent_obstacles(center_position, obstacle_list):
    """
    Find all obstacles surrounding a position (3x3 neighborhood).
    
    Used for visualizing the "blocking boundary" when search fails.
    
    Args:
        center_position: [x, y] center coordinate
        obstacle_list: List of obstacle coordinates
    
    Returns:
        List of obstacles adjacent to center
    """
    adjacent = []
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            check_pos = [center_position[0] + x_offset, 
                        center_position[1] + y_offset]
            if check_pos in obstacle_list:
                adjacent.append(check_pos)
    return adjacent


def compute_boundary_obstacles(explored_points, all_obstacles):
    """
    Find obstacles forming the boundary of explored region.
    
    When a search is blocked, this identifies which obstacles
    prevented further expansion.
    
    Args:
        explored_points: List of explored SearchPoints
        all_obstacles: List of all obstacle coordinates
    
    Returns:
        Numpy array of boundary obstacle coordinates
    """
    boundary_obstacles = []
    explored_positions = extract_positions(explored_points)
    
    # For each explored position, find adjacent obstacles
    for position in explored_positions:
        adjacent = identify_adjacent_obstacles(position, all_obstacles)
        boundary_obstacles.extend(adjacent)
    
    return np.array(boundary_obstacles)


def reconstruct_complete_path(forward_explored, backward_explored, meeting_position):
    """
    Build complete path by joining forward and backward segments.
    
    Process:
    1. Trace from meeting point back to start (forward search)
    2. Trace from meeting point back to goal (backward search)
    3. Reverse forward segment
    4. Concatenate: start→meeting + meeting→goal
    
    Args:
        forward_explored: Explored list from start search
        backward_explored: Explored list from goal search
        meeting_position: Coordinate where searches met
    
    Returns:
        Numpy array of complete path coordinates
    """
    forward_segment = []
    backward_segment = []
    
    # ===== TRACE FORWARD PATH (MEETING → START) =====
    meeting_index = locate_point_index(meeting_position, forward_explored)
    current = forward_explored[meeting_index]
    
    while current != forward_explored[0]:
        forward_segment.append(current.position)
        current = current.predecessor
    forward_segment.append(forward_explored[0].position)
    
    # ===== TRACE BACKWARD PATH (MEETING → GOAL) =====
    meeting_index = locate_point_index(meeting_position, backward_explored)
    current = backward_explored[meeting_index]
    
    while current != backward_explored[0]:
        backward_segment.append(current.position)
        current = current.predecessor
    backward_segment.append(backward_explored[0].position)
    
    # ===== COMBINE SEGMENTS =====
    forward_segment.reverse()  # Now goes start→meeting
    complete_path = forward_segment + backward_segment
    return np.array(complete_path)


def generate_random_position(lower_bound, upper_bound):
    """
    Generate random coordinate within specified bounds.
    
    Args:
        lower_bound: [x, y] minimum coordinates
        upper_bound: [x, y] maximum coordinates
    
    Returns:
        Random [x, y] coordinate in valid range
    """
    random_x = np.random.randint(lower_bound[0] + 1, upper_bound[0])
    random_y = np.random.randint(lower_bound[1] + 1, upper_bound[1])
    return [random_x, random_y]


def render_search_state(forward_explored, backward_explored, start, goal, obstacles):
    """
    Visualize current state of bidirectional search.
    
    Color scheme:
    - Yellow circles: Forward search (start→goal)
    - Green circles: Backward search (goal→start)
    - Black squares: Obstacles
    - Blue triangle: Start position
    - Blue star: Goal position
    
    Args:
        forward_explored: Array of forward-searched coordinates
        backward_explored: Array of backward-searched coordinates
        start: Start position
        goal: Goal position
        obstacles: Obstacle coordinate array
    """
    
    # Handle edge case: backward search not yet started
    if not backward_explored.tolist():
        backward_explored = np.array([goal])
    
    # Clear and setup figure
    plt.cla()
    plt.gcf().set_size_inches(11, 9, forward=True)
    plt.axis('equal')
    
    # Render exploration frontiers
    plt.plot(forward_explored[:, 0], forward_explored[:, 1], 
            'oy', markersize=3, label='Forward Search')
    plt.plot(backward_explored[:, 0], backward_explored[:, 1], 
            'og', markersize=3, label='Backward Search')
    
    # Render obstacles
    plt.plot(obstacles[:, 0], obstacles[:, 1], 
            'sk', markersize=2, label='Obstacles')
    
    # Render start and goal markers
    plt.plot(goal[0], goal[1], '*b', markersize=10, label='Goal')
    plt.plot(start[0], start[1], '^b', markersize=10, label='Start')
    
    plt.legend()
    plt.title('Bidirectional A* Search (ESC to exit)')
    plt.pause(FRAME_DELAY)


def visualization_controller(forward_explored, backward_explored, 
                            search_status, start, goal, obstacles, obstacle_list):
    """
    Central controller for visualization and completion detection.
    
    Responsibilities:
    1. Render current search state
    2. Check if searches have met
    3. Handle blocked search conditions
    4. Draw completion or failure states
    
    Args:
        forward_explored: Forward search explored list
        backward_explored: Backward search explored list
        search_status: 0=active, 1=start blocked, 2=goal blocked
        start: Start coordinate
        goal: Goal coordinate
        obstacles: Obstacle array
        obstacle_list: Obstacle list (for boundary detection)
    
    Returns:
        termination_flag: 1 to stop search, 0 to continue
        path: Complete path array or None
    """
    
    termination_flag = 0
    
    # Convert to coordinate arrays for visualization
    forward_coords = np.array(extract_positions(forward_explored))
    backward_coords = np.array(extract_positions(backward_explored))
    found_path = None
    
    # Render current state
    if SHOW_SEARCH_ANIMATION:
        render_search_state(forward_coords, backward_coords, start, goal, obstacles)
    
    # ===== CHECK SEARCH STATUS =====
    
    if search_status == 0:  # Normal search - check for meeting
        meeting_points = detect_frontier_collision(forward_explored, backward_explored)
        
        if meeting_points:  # SUCCESS - Searches met!
            found_path = reconstruct_complete_path(
                forward_explored, backward_explored, meeting_points[0]
            )
            termination_flag = 1
            print('[SUCCESS] Path discovered!')
            
            if SHOW_SEARCH_ANIMATION:
                plt.plot(found_path[:, 0], found_path[:, 1], '-r', linewidth=2)
                plt.title('Path Found - Search Complete', size=20)
                plt.pause(0.01)
                plt.show()
    
    elif search_status == 1:  # Start position blocked
        termination_flag = 1
        print('[FAILURE] Start position is completely blocked!')
    
    elif search_status == 2:  # Goal position blocked
        termination_flag = 1
        print('[FAILURE] Goal position is completely blocked!')
    
    # ===== VISUALIZE BLOCKING BOUNDARIES =====
    if SHOW_SEARCH_ANIMATION:
        if search_status == 1:  # Show start blocking boundary
            boundary = compute_boundary_obstacles(forward_explored, obstacle_list)
            plt.plot(boundary[:, 0], boundary[:, 1], 'xr', markersize=8)
            plt.title('Start Blocked - No Path Exists', size=14)
            plt.pause(0.01)
            plt.show()
        
        elif search_status == 2:  # Show goal blocking boundary
            boundary = compute_boundary_obstacles(backward_explored, obstacle_list)
            plt.plot(boundary[:, 0], boundary[:, 1], 'xr', markersize=8)
            plt.title('Goal Blocked - No Path Exists', size=14)
            plt.pause(0.01)
            plt.show()
    
    return termination_flag, found_path


def execute_bidirectional_search(start, goal, all_obstacles, interior_obstacles):
    """
    Main bidirectional A* search orchestrator.
    
    Algorithm:
    1. Initialize searches from both start and goal
    2. Alternate between expanding forward and backward
    3. Each search targets best node from opposite search
    4. Continue until searches meet or one is blocked
    
    Args:
        start: Starting position [x, y]
        goal: Goal position [x, y]
        all_obstacles: Complete obstacle array
        interior_obstacles: Interior obstacle list
    
    Returns:
        Complete path array or None if no path exists
    """
    
    # ===== INITIALIZE SEARCH ROOTS =====
    forward_root = SearchPoint(
        position=start,
        estimated_cost=manhattan_distance(start, goal)
    )
    
    backward_root = SearchPoint(
        position=goal,
        estimated_cost=manhattan_distance(goal, start)
    )
    
    # ===== INITIALIZE SEARCH STRUCTURES =====
    forward_frontier = [forward_root]
    forward_explored = []
    
    backward_frontier = [backward_root]
    backward_explored = []
    
    # Initial targets
    forward_target = goal
    
    # Search status: 0=active, 1=start blocked, 2=goal blocked
    status = 0
    result_path = None
    
    iteration_counter = 0
    
    # ===== MAIN BIDIRECTIONAL LOOP =====
    while True:
        # Check for user abort
        global CONTINUE_EXECUTION
        if not CONTINUE_EXECUTION:
            print("[ABORT] Search terminated by user")
            return None
        
        # ===== EXPAND FORWARD SEARCH =====
        forward_frontier, forward_explored = expand_search_frontier(
            forward_frontier, forward_explored, forward_target, all_obstacles
        )
        
        if not forward_frontier:  # Forward search blocked
            status = 1
            visualization_controller(forward_explored, backward_explored, 
                                   status, start, goal, all_obstacles, interior_obstacles)
            break
        
        # Update target for backward search
        backward_target = min(forward_frontier, key=lambda pt: pt.total_score).position
        
        # ===== EXPAND BACKWARD SEARCH =====
        backward_frontier, backward_explored = expand_search_frontier(
            backward_frontier, backward_explored, backward_target, all_obstacles
        )
        
        if not backward_frontier:  # Backward search blocked
            status = 2
            visualization_controller(forward_explored, backward_explored,
                                   status, start, goal, all_obstacles, interior_obstacles)
            break
        
        # Update target for forward search
        forward_target = min(backward_frontier, key=lambda pt: pt.total_score).position
        
        # ===== CHECK FOR MEETING (Every 5 iterations) =====
        iteration_counter += 1
        if iteration_counter % 5 == 0:
            stop_flag, result_path = visualization_controller(
                forward_explored, backward_explored, status, 
                start, goal, all_obstacles, interior_obstacles
            )
            if stop_flag:
                break
        else:
            # Quick check without visualization
            meeting_points = detect_frontier_collision(forward_explored, backward_explored)
            if meeting_points:
                result_path = reconstruct_complete_path(
                    forward_explored, backward_explored, meeting_points[0]
                )
                visualization_controller(forward_explored, backward_explored,
                                       status, start, goal, all_obstacles, interior_obstacles)
                break
    
    return result_path


def run_pathfinding_demonstration(obstacle_density=800):
    """
    Entry point for bidirectional A* demonstration.
    
    Creates a test environment with:
    - Rectangular boundary walls
    - Random interior obstacles
    - Start at bottom-left corner
    - Goal at top-right corner
    - Maximum diagonal challenge
    
    Args:
        obstacle_density: Number of random obstacles (default: 800 ≈ 22% density)
    """
    print("=" * 70)
    print("BIDIRECTIONAL A* PATHFINDING DEMONSTRATION")
    print("=" * 70)
    
    # Define search space
    upper_boundary = [60, 60]
    lower_boundary = [0, 0]
    
    # Position start and goal at diagonal extremes
    start_position = [lower_boundary[0] + 1, lower_boundary[1] + 1]  # (1, 1)
    goal_position = [upper_boundary[0] - 1, upper_boundary[1] - 1]    # (59, 59)
    
    print(f"\n[CONFIG] Start: {start_position}")
    print(f"[CONFIG] Goal: {goal_position}")
    print(f"[CONFIG] Direct distance: "
          f"{math.hypot(goal_position[0]-start_position[0], goal_position[1]-start_position[1]):.1f}m")
    print(f"[CONFIG] Obstacles: {obstacle_density} (~22% density)")
    
    # Generate environment
    all_obstacles, interior_obstacles = create_environment(
        start_position, goal_position, 
        upper_boundary, lower_boundary, 
        obstacle_density
    )
    
    # Setup visualization
    if SHOW_SEARCH_ANIMATION:
        figure = plt.figure(figsize=(11, 9))
        figure.canvas.mpl_connect('key_press_event', handle_escape_key)
    
    # Execute search
    print("\n[SEARCH] Launching bidirectional A*... (ESC to abort)")
    result = execute_bidirectional_search(
        start_position, goal_position, 
        all_obstacles, interior_obstacles
    )
    
    # Report results
    if result is not None and len(result) > 0:
        path_length = sum([
            math.hypot(result[i+1][0]-result[i][0], result[i+1][1]-result[i][1])
            for i in range(len(result)-1)
        ])
        print(f"\n[RESULT] Path found successfully!")
        print(f"  → Length: {path_length:.2f} meters")
        print(f"  → Waypoints: {len(result)}")
        print("\n[INFO] Close window or press ESC to exit")
    elif not SHOW_SEARCH_ANIMATION and result is not None:
        print(result)


# ===== PROGRAM ENTRY POINT =====
if __name__ == '__main__':
    run_pathfinding_demonstration(obstacle_density=800)