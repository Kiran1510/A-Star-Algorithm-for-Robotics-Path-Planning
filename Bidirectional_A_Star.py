"""
Bidirectional A* Algorithm
Author: Weicent

This implementation searches for a path by exploring from both the start 
and goal simultaneously, meeting in the middle. This can be roughly 2x 
faster than standard A* in open environments.

Features:
- Random obstacle generation for testing
- Dual-direction search (start→goal and goal→start)
- Detects when start or goal is completely blocked
- Visual animation of search process
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import sys

# Flag to enable/disable real-time visualization of the search process
show_animation = True

# Animation speed control (seconds between updates)
# Increase for slower, more visible animation
# Decrease for faster execution
animation_delay = 0.2  # 50ms delay = ~20 updates per second

# Global flag for clean exit
should_exit = False


def on_key_press(event):
    """
    Handle keyboard events for clean program termination.
    Press ESC to exit the program gracefully.
    """
    global should_exit
    if event.key == 'escape':
        print("\nESC pressed - Exiting program cleanly...")
        should_exit = True
        plt.close('all')
        sys.exit(0)


class Node:
    """
    Node represents a single cell/position in the search grid.
    
    Properties:
    - G: Cost from start to this node (g(n) in A*)
    - H: Heuristic estimated cost from this node to goal (h(n) in A*)
    - F: Total cost (F = G + H, the evaluation function f(n) in A*)
    - coordinate: [x, y] position in the grid
    - parent: Reference to the parent node (for path reconstruction)
    """

    def __init__(self, G=0, H=0, coordinate=None, parent=None):
        self.G = G  # Actual cost from start to this node
        self.H = H  # Heuristic estimated cost to goal
        self.F = G + H  # Total estimated cost f(n) = g(n) + h(n)
        self.parent = parent  # Parent node for backtracking the path
        self.coordinate = coordinate  # [x, y] position

    def reset_f(self):
        """
        Recalculate F value after G or H is updated.
        Called when a cheaper path to this node is found.
        """
        self.F = self.G + self.H


def hcost(node_coordinate, goal):
    """
    Calculate heuristic cost h(n) from node to goal.
    
    Uses Manhattan distance (L1 norm): |dx| + |dy|
    This is admissible for grid-based 8-connected movement.
    
    Args:
        node_coordinate: [x, y] position of current node
        goal: [x, y] position of goal
    
    Returns:
        Manhattan distance between node and goal
    """
    dx = abs(node_coordinate[0] - goal[0])
    dy = abs(node_coordinate[1] - goal[1])
    hcost = dx + dy
    return hcost


def gcost(fixed_node, update_node_coordinate):
    """
    Calculate actual cost g(n) to reach update_node from start.
    
    Args:
        fixed_node: The parent node (already in closed list)
        update_node_coordinate: [x, y] coordinate of the node to update
    
    Returns:
        Total cost from start to update_node via fixed_node
        
    Note: Uses Euclidean distance for movement cost between adjacent cells
          (1.0 for cardinal, ~1.414 for diagonal moves)
    """
    # Distance from fixed_node to update_node
    dx = abs(fixed_node.coordinate[0] - update_node_coordinate[0])
    dy = abs(fixed_node.coordinate[1] - update_node_coordinate[1])
    gc = math.hypot(dx, dy)  # Euclidean distance: sqrt(dx^2 + dy^2)
    
    # Total cost = cost to reach fixed_node + cost from fixed_node to update_node
    gcost = fixed_node.G + gc
    return gcost


def boundary_and_obstacles(start, goal, top_vertex, bottom_vertex, obs_number):
    """
    Generate the environment: rectangular boundary walls and random obstacles.
    
    Args:
        start: [x, y] starting position
        goal: [x, y] goal position
        top_vertex: [x, y] top-right corner of boundary
        bottom_vertex: [x, y] bottom-left corner of boundary
        obs_number: Number of random obstacles to generate
    
    Returns:
        bound_obs: numpy array of all obstacles (boundary + random obstacles)
        obstacle: list of random obstacle coordinates (excluding start/goal)
    """
    
    # ===== CONSTRUCT BOUNDARY WALLS =====
    # Create four walls forming a rectangle
    
    # Left wall: vertical line at x = bottom_vertex[0]
    ay = list(range(bottom_vertex[1], top_vertex[1]))
    ax = [bottom_vertex[0]] * len(ay)
    
    # Right wall: vertical line at x = top_vertex[0]
    cy = ay  # Same y-coordinates as left wall
    cx = [top_vertex[0]] * len(cy)
    
    # Bottom wall: horizontal line at y = bottom_vertex[1]
    bx = list(range(bottom_vertex[0] + 1, top_vertex[0]))
    by = [bottom_vertex[1]] * len(bx)
    
    # Top wall: horizontal line at y = top_vertex[1]
    dx = [bottom_vertex[0]] + bx + [top_vertex[0]]
    dy = [top_vertex[1]] * len(dx)

    # ===== GENERATE RANDOM OBSTACLES =====
    # Create random interior obstacles within the boundary
    ob_x = np.random.randint(bottom_vertex[0] + 1,
                             top_vertex[0], obs_number).tolist()
    ob_y = np.random.randint(bottom_vertex[1] + 1,
                             top_vertex[1], obs_number).tolist()
    
    # Combine x,y coordinates for boundary walls in order
    x = ax + bx + cx + dx
    y = ay + by + cy + dy
    
    # Create obstacle list from random coordinates
    obstacle = np.vstack((ob_x, ob_y)).T.tolist()
    
    # Remove start and goal from obstacles (ensure they're not blocked)
    obstacle = [coor for coor in obstacle if coor != start and coor != goal]
    
    # Combine boundary and obstacles into single array
    obs_array = np.array(obstacle)
    bound = np.vstack((x, y)).T
    bound_obs = np.vstack((bound, obs_array))
    
    return bound_obs, obstacle


def find_neighbor(node, ob, closed):
    """
    Find all valid neighboring cells around a node.
    
    Implements 8-connected movement (up, down, left, right, and 4 diagonals)
    with corner-cutting prevention (can't move diagonally through obstacles).
    
    Args:
        node: Current node to find neighbors for
        ob: numpy array of obstacle coordinates
        closed: list of already-explored coordinates
    
    Returns:
        List of valid neighbor coordinates
    """
    
    # Convert obstacles to set for O(1) lookup
    ob_set = set(map(tuple, ob.tolist()))  
    neighbor_set = set()

    # ===== GENERATE ALL 8 NEIGHBORS =====
    # Check 3x3 grid around current node (excluding node itself)
    for x in range(node.coordinate[0] - 1, node.coordinate[0] + 2):
        for y in range(node.coordinate[1] - 1, node.coordinate[1] + 2):
            coord = (x, y)
            # Add if not an obstacle and not the current node
            if coord not in ob_set and coord != tuple(node.coordinate):
                neighbor_set.add(coord)

    # ===== DEFINE CARDINAL AND DIAGONAL NEIGHBORS =====
    # Cardinal directions (up, down, left, right)
    top_nei = (node.coordinate[0], node.coordinate[1] + 1)
    bottom_nei = (node.coordinate[0], node.coordinate[1] - 1)
    left_nei = (node.coordinate[0] - 1, node.coordinate[1])
    right_nei = (node.coordinate[0] + 1, node.coordinate[1])
    
    # Diagonal directions
    lt_nei = (node.coordinate[0] - 1, node.coordinate[1] + 1)  # left-top
    rt_nei = (node.coordinate[0] + 1, node.coordinate[1] + 1)  # right-top
    lb_nei = (node.coordinate[0] - 1, node.coordinate[1] - 1)  # left-bottom
    rb_nei = (node.coordinate[0] + 1, node.coordinate[1] - 1)  # right-bottom

    # ===== PREVENT CORNER CUTTING =====
    # Can't move diagonally if both adjacent cardinal cells are blocked
    # This prevents "squeezing" through diagonal gaps
    
    # Can't go top-left if both top and left are blocked
    if top_nei in ob_set and left_nei in ob_set:
        neighbor_set.discard(lt_nei)
    
    # Can't go top-right if both top and right are blocked
    if top_nei in ob_set and right_nei in ob_set:
        neighbor_set.discard(rt_nei)
    
    # Can't go bottom-left if both bottom and left are blocked
    if bottom_nei in ob_set and left_nei in ob_set:
        neighbor_set.discard(lb_nei)
    
    # Can't go bottom-right if both bottom and right are blocked
    if bottom_nei in ob_set and right_nei in ob_set:
        neighbor_set.discard(rb_nei)

    # ===== REMOVE ALREADY-EXPLORED NEIGHBORS =====
    # Don't revisit nodes already in closed list
    closed_set = set(map(tuple, closed))
    neighbor_set -= closed_set

    return list(neighbor_set)


def find_node_index(coordinate, node_list):
    """
    Find the index of a node in a list by its coordinate.
    
    Args:
        coordinate: [x, y] coordinate to search for
        node_list: List of Node objects
    
    Returns:
        Index of the node with matching coordinate
    """
    ind = 0
    for node in node_list:
        if node.coordinate == coordinate:
            target_node = node
            ind = node_list.index(target_node)
            break
    return ind


def find_path(open_list, closed_list, goal, obstacle):
    """
    Core A* search step: expand all nodes in current open list.
    
    This processes one "generation" of nodes - all nodes currently in the
    open list get expanded, their neighbors evaluated, and costs updated.
    
    Args:
        open_list: List of nodes to be explored (frontier)
        closed_list: List of already-explored nodes
        goal: Target coordinate for this search direction
        obstacle: Array of obstacle coordinates
    
    Returns:
        Updated open_list and closed_list after expansion
    """
    
    # Process all nodes currently in open list
    flag = len(open_list)
    for i in range(flag):
        # Always process the best node (lowest F value) first
        node = open_list[0]
        
        # Extract coordinates for faster lookup
        open_coordinate_list = [node.coordinate for node in open_list]
        closed_coordinate_list = [node.coordinate for node in closed_list]
        
        # ===== FIND AND PROCESS NEIGHBORS =====
        temp = find_neighbor(node, obstacle, closed_coordinate_list)
        
        for element in temp:
            # Skip if already fully explored
            if element in closed_list:
                continue
            
            # ===== NEIGHBOR ALREADY IN OPEN LIST =====
            elif element in open_coordinate_list:
                # Node discovered before - check if we found a cheaper path
                ind = open_coordinate_list.index(element)
                new_g = gcost(node, element)
                
                # If this path is cheaper, update the node
                if new_g <= open_list[ind].G:
                    open_list[ind].G = new_g
                    open_list[ind].reset_f()  # Recalculate F = G + H
                    open_list[ind].parent = node  # Update parent for path reconstruction
            
            # ===== NEW NEIGHBOR - CREATE NODE =====
            else:
                # First time seeing this coordinate, create new node
                ele_node = Node(coordinate=element, 
                              parent=node,
                              G=gcost(node, element), 
                              H=hcost(element, goal))
                open_list.append(ele_node)
        
        # ===== MOVE CURRENT NODE TO CLOSED LIST =====
        # Done exploring this node
        open_list.remove(node)
        closed_list.append(node)
        
        # ===== SORT OPEN LIST BY F VALUE =====
        # Keep best candidates (lowest F) at front for next iteration
        open_list.sort(key=lambda x: x.F)
    
    return open_list, closed_list


def node_to_coordinate(node_list):
    """
    Extract just the coordinates from a list of Node objects.
    
    Args:
        node_list: List of Node objects
    
    Returns:
        List of [x, y] coordinates
    """
    coordinate_list = [node.coordinate for node in node_list]
    return coordinate_list


def check_node_coincide(close_ls1, closed_ls2):
    """
    Check if the two search frontiers have met.
    
    This is the key to bidirectional search - when a node appears in both
    closed lists, the searches have met and a path exists.
    
    Args:
        close_ls1: Closed list from start→goal search
        closed_ls2: Closed list from goal→start search
    
    Returns:
        List of coordinates where the searches intersect (meeting points)
    """
    # Convert node lists to coordinate lists
    cl1 = node_to_coordinate(close_ls1)
    cl2 = node_to_coordinate(closed_ls2)
    
    # Find coordinates that appear in both lists
    intersect_ls = [node for node in cl1 if node in cl2]
    return intersect_ls


def find_surrounding(coordinate, obstacle):
    """
    Find all obstacles in the 3x3 grid around a coordinate.
    
    Used to draw the "border line" when a search gets completely blocked,
    showing which obstacles are confining the start or goal.
    
    Args:
        coordinate: [x, y] center position
        obstacle: List of obstacle coordinates
    
    Returns:
        List of obstacle coordinates adjacent to the input coordinate
    """
    boundary: list = []
    # Check 3x3 grid around coordinate
    for x in range(coordinate[0] - 1, coordinate[0] + 2):
        for y in range(coordinate[1] - 1, coordinate[1] + 2):
            if [x, y] in obstacle:
                boundary.append([x, y])
    return boundary


def get_border_line(node_closed_ls, obstacle):
    """
    Find all obstacles on the boundary of explored region.
    
    When a search is completely blocked, this identifies the "wall" of
    obstacles that prevented further expansion.
    
    Args:
        node_closed_ls: List of explored nodes
        obstacle: List of all obstacles
    
    Returns:
        numpy array of obstacle coordinates forming the boundary
    """
    border: list = []
    coordinate_closed_ls = node_to_coordinate(node_closed_ls)
    
    # For each explored node, find surrounding obstacles
    for coordinate in coordinate_closed_ls:
        temp = find_surrounding(coordinate, obstacle)
        border = border + temp
    
    border_ary = np.array(border)
    return border_ary


def get_path(org_list, goal_list, coordinate):
    """
    Reconstruct the complete path from start to goal.
    
    Traces back from the meeting point to both start and goal,
    then combines them into the full path.
    
    Args:
        org_list: Closed list from start→goal search
        goal_list: Closed list from goal→start search
        coordinate: Meeting point where both searches intersected
    
    Returns:
        numpy array of coordinates forming the complete path
    """
    path_org: list = []
    path_goal: list = []
    
    # ===== TRACE PATH FROM MEETING POINT TO START =====
    # Find the meeting node in origin search
    ind = find_node_index(coordinate, org_list)
    node = org_list[ind]
    
    # Follow parent pointers back to start
    while node != org_list[0]:
        path_org.append(node.coordinate)
        node = node.parent
    path_org.append(org_list[0].coordinate)  # Add start coordinate
    
    # ===== TRACE PATH FROM MEETING POINT TO GOAL =====
    # Find the meeting node in goal search
    ind = find_node_index(coordinate, goal_list)
    node = goal_list[ind]
    
    # Follow parent pointers back to goal
    while node != goal_list[0]:
        path_goal.append(node.coordinate)
        node = node.parent
    path_goal.append(goal_list[0].coordinate)  # Add goal coordinate
    
    # ===== COMBINE PATHS =====
    # Reverse origin path so it goes start→meeting
    path_org.reverse()
    # Concatenate: start→meeting + meeting→goal
    path = path_org + path_goal
    path = np.array(path)
    return path


def random_coordinate(bottom_vertex, top_vertex):
    """
    Generate a random coordinate within the boundary.
    
    Args:
        bottom_vertex: [x, y] bottom-left corner
        top_vertex: [x, y] top-right corner
    
    Returns:
        Random [x, y] coordinate inside the boundary
    """
    coordinate = [np.random.randint(bottom_vertex[0] + 1, top_vertex[0]),
                  np.random.randint(bottom_vertex[1] + 1, top_vertex[1])]
    return coordinate


def draw(close_origin, close_goal, start, end, bound):
    """
    Visualize the current search state.
    
    Shows:
    - Yellow circles: Nodes explored from start
    - Green circles: Nodes explored from goal
    - Black squares: Obstacles
    - Blue markers: Start (^) and Goal (*)
    
    Args:
        close_origin: Array of coordinates explored from start
        close_goal: Array of coordinates explored from goal
        start: Starting coordinate
        end: Goal coordinate
        bound: Array of obstacle coordinates
    """
    
    # Handle edge case where goal search hasn't started yet
    if not close_goal.tolist():
        # If origin is blocked immediately, goal search never runs
        # Add goal coordinate to array for plotting purposes
        close_goal = np.array([end])
    
    # Clear previous plot and set figure size
    plt.cla()
    plt.gcf().set_size_inches(11, 9, forward=True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Plot explored nodes
    plt.plot(close_origin[:, 0], close_origin[:, 1], 'oy', markersize=3)  # Yellow: from start
    plt.plot(close_goal[:, 0], close_goal[:, 1], 'og', markersize=3)      # Green: from goal
    
    # Plot obstacles and boundary
    plt.plot(bound[:, 0], bound[:, 1], 'sk', markersize=2)  # Black squares
    
    # Plot start and goal markers
    plt.plot(end[0], end[1], '*b', markersize=10, label='Goal')      # Blue star
    plt.plot(start[0], start[1], '^b', markersize=10, label='Origin')  # Blue triangle
    
    plt.legend()
    plt.title('Bidirectional A* Search (Press ESC to exit)')
    plt.pause(animation_delay)  # Pause based on animation_delay setting


def draw_control(org_closed, goal_closed, flag, start, end, bound, obstacle):
    """
    Control the visualization and evaluate search completion.
    
    This is the main visualization controller that:
    1. Draws the current search state
    2. Checks if searches have met (path found)
    3. Handles blocked conditions and draws border lines
    
    Args:
        org_closed: Closed list from start search
        goal_closed: Closed list from goal search
        flag: Status flag (0=searching, 1=start blocked, 2=goal blocked)
        start: Start coordinate
        end: Goal coordinate
        bound: Obstacle array
        obstacle: Obstacle list
    
    Returns:
        stop_loop: 1 if search should stop, 0 to continue
        path: numpy array of path coordinates (None if no path)
    """
    
    stop_loop = 0  # Flag to stop the search loop
    
    # Convert node lists to coordinate arrays for plotting
    org_closed_ls = node_to_coordinate(org_closed)
    org_array = np.array(org_closed_ls)
    goal_closed_ls = node_to_coordinate(goal_closed)
    goal_array = np.array(goal_closed_ls)
    path = None
    
    # Draw current search state
    if show_animation:
        draw(org_array, goal_array, start, end, bound)
    
    # ===== CHECK SEARCH STATUS =====
    
    if flag == 0:  # Normal searching - check if paths have met
        node_intersect = check_node_coincide(org_closed, goal_closed)
        
        if node_intersect:  # Searches have met - path found!
            path = get_path(org_closed, goal_closed, node_intersect[0])
            stop_loop = 1
            print('Path found!')
            
            if show_animation:  # Draw the complete path
                plt.plot(path[:, 0], path[:, 1], '-r')  # Red line
                plt.title('Robot Arrived', size=20, loc='center')
                plt.pause(0.01)
                plt.show()
    
    elif flag == 1:  # Start point is completely blocked
        stop_loop = 1
        print('There is no path to the goal! Start point is blocked!')
    
    elif flag == 2:  # Goal point is completely blocked
        stop_loop = 1
        print('There is no path to the goal! End point is blocked!')
    
    # ===== DRAW BORDER LINES FOR BLOCKED CASES =====
    if show_animation:
        info = 'There is no path to the goal!' \
               ' Robot&Goal are split by border' \
               ' shown in red \'x\'!'
        
        if flag == 1:  # Show obstacles blocking start
            border = get_border_line(org_closed, obstacle)
            plt.plot(border[:, 0], border[:, 1], 'xr')  # Red X markers
            plt.title(info, size=14, loc='center')
            plt.pause(0.01)
            plt.show()
        
        elif flag == 2:  # Show obstacles blocking goal
            border = get_border_line(goal_closed, obstacle)
            plt.plot(border[:, 0], border[:, 1], 'xr')  # Red X markers
            plt.title(info, size=14, loc='center')
            plt.pause(0.01)
            plt.show()
    
    return stop_loop, path


def searching_control(start, end, bound, obstacle):
    """
    Main bidirectional search controller.
    
    Manages the alternating search process:
    1. Search from start toward goal's best node
    2. Search from goal toward start's best node
    3. Check if searches have met
    4. Repeat until path found or one side is blocked
    
    Args:
        start: Starting coordinate [x, y]
        end: Goal coordinate [x, y]
        bound: Array of all obstacles (boundary + random)
        obstacle: List of random obstacles only
    
    Returns:
        path: numpy array of path coordinates (None if no path exists)
    """
    
    # ===== INITIALIZE NODES =====
    # Create starting nodes for both directions
    origin = Node(coordinate=start, H=hcost(start, end))
    goal = Node(coordinate=end, H=hcost(end, start))
    
    # ===== INITIALIZE SEARCH LISTS =====
    # Lists for searching from origin to goal
    origin_open: list = [origin]  # Nodes to explore
    origin_close: list = []       # Nodes already explored
    
    # Lists for searching from goal to origin
    goal_open = [goal]           # Nodes to explore
    goal_close: list = []        # Nodes already explored
    
    # ===== INITIALIZE SEARCH TARGETS =====
    target_goal = end  # Initial target for origin search
    
    # Status flag: 0=searching, 1=start blocked, 2=goal blocked
    flag = 0
    path = None
    
    # Counter for batch visualization updates
    iteration_count = 0
    
    # ===== MAIN BIDIRECTIONAL SEARCH LOOP =====
    while True:
        # Check for clean exit request
        global should_exit
        if should_exit:
            print("Search interrupted by user")
            return None
        
        # ===== SEARCH FROM START TOWARD GOAL =====
        origin_open, origin_close = \
            find_path(origin_open, origin_close, target_goal, bound)
        
        # Check if start search is blocked (open list empty = nowhere to expand)
        if not origin_open:
            flag = 1  # Origin node is blocked
            draw_control(origin_close, goal_close, flag, start, end, bound,
                         obstacle)
            break
        
        # ===== UPDATE TARGET FOR GOAL SEARCH =====
        # Goal search should aim for the best node found by origin search
        target_origin = min(origin_open, key=lambda x: x.F).coordinate

        # ===== SEARCH FROM GOAL TOWARD START =====
        goal_open, goal_close = \
            find_path(goal_open, goal_close, target_origin, bound)
        
        # Check if goal search is blocked
        if not goal_open:
            flag = 2  # Goal is blocked
            draw_control(origin_close, goal_close, flag, start, end, bound,
                         obstacle)
            break
        
        # ===== UPDATE TARGET FOR ORIGIN SEARCH =====
        # Origin search should aim for the best node found by goal search
        target_goal = min(goal_open, key=lambda x: x.F).coordinate

        # ===== CHECK IF SEARCHES HAVE MET =====
        # Update visualization every 5 iterations instead of every 10
        # (More frequent updates = slower, more visible animation)
        iteration_count += 1
        if iteration_count % 5 == 0:
            stop_sign, path = draw_control(origin_close, goal_close, flag, start,
                                           end, bound, obstacle)
            if stop_sign:
                break
        else:
            # Check for path without drawing (faster)
            node_intersect = check_node_coincide(origin_close, goal_close)
            if node_intersect:
                path = get_path(origin_close, goal_close, node_intersect[0])
                # Draw final result
                draw_control(origin_close, goal_close, flag, start, end, bound, obstacle)
                break
    
    return path


def main(obstacle_number=800):
    """
    Main entry point for bidirectional A* demonstration.
    
    Creates a random environment with obstacles and searches for a path
    from a random start to a random goal using bidirectional A*.
    
    Args:
        obstacle_number: Number of random obstacles to generate (default: 800)
                        In a 60x60 grid, 800 obstacles = ~22% density
    """
    print(__file__ + ' start!')

    # Define boundary of the search space
    top_vertex = [60, 60]    # Top-right corner
    bottom_vertex = [0, 0]   # Bottom-left corner

    # ===== SET START AND GOAL AT DIAGONAL EXTREMES =====
    # Place at opposite corners for maximum path length challenge
    start = [bottom_vertex[0] + 1, bottom_vertex[1] + 1]  # Bottom-left: (1, 1)
    end = [top_vertex[0] - 1, top_vertex[1] - 1]          # Top-right: (59, 59)
    
    print(f"Start position: {start}")
    print(f"Goal position: {end}")
    print(f"Diagonal distance: ~{math.hypot(end[0]-start[0], end[1]-start[1]):.1f} units")

    # ===== GENERATE ENVIRONMENT =====
    # Create boundary walls and random obstacles
    bound, obstacle = boundary_and_obstacles(start, end, top_vertex,
                                             bottom_vertex,
                                             obstacle_number)

    # Setup figure and connect keyboard handler
    if show_animation:
        fig = plt.figure(figsize=(11, 9))
        fig.canvas.mpl_connect('key_press_event', on_key_press)

    # ===== RUN BIDIRECTIONAL A* SEARCH =====
    print("\nStarting bidirectional A* search... (Press ESC to exit)")
    path = searching_control(start, end, bound, obstacle)
    
    # If animation is off, print the path array
    if not show_animation and path is not None:
        print(path)
    
    # Calculate and display path statistics
    if path is not None and len(path) > 0:
        path_length = sum([math.hypot(path[i+1][0]-path[i][0], 
                                     path[i+1][1]-path[i][1]) 
                          for i in range(len(path)-1)])
        print(f"\nPath found! Total path length: {path_length:.2f} units")
        print("Press ESC to close window, or close manually.")


# Run the program
if __name__ == '__main__':
    # Run with fewer obstacles for better path success rate
    # 800 obstacles = ~22% density (much more navigable than 40%)
    main(obstacle_number=800)