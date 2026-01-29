# A* Algorithm for Robotics Path Planning

Implementations of A* pathfinding algorithms for robotics applications, featuring standard A* with varying complexity levels and bidirectional A* for enhanced performance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Algorithm Descriptions](#algorithm-descriptions)
- [Visualizations](#visualizations)
- [Performance Considerations](#performance-considerations)
- [Advanced Topics](#advanced-topics)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides three distinct implementations of the A* pathfinding algorithm, designed for robotics path planning applications. Each implementation demonstrates different aspects of the algorithm, from basic concepts to advanced optimization strategies.

### What is A*?

A* (A-star) is a graph traversal and pathfinding algorithm widely used in robotics, video games, and AI applications. It finds the shortest path between two points while avoiding obstacles by using a heuristic function to guide its search.

**Core Concept:**
```
f(n) = g(n) + h(n)
```
- **g(n)**: Actual cost from start to node n
- **h(n)**: Estimated cost from node n to goal (heuristic)
- **f(n)**: Total estimated cost through node n

## Features

- **Three Progressive Implementations**: Simple maze, complex maze, and bidirectional search
- **Real-time Visualization**: Animated search process showing frontier expansion
- **Optimized Performance**: Batch rendering for smooth animation without slowdown
- **Interactive Controls**: ESC key for clean program exit
- **Comprehensive Documentation**: Extensively commented code explaining every step
- **Flexible Configuration**: Adjustable grid resolution, animation speed, and obstacle density
- **Path Metrics**: Automatic calculation of path length and waypoint count

## Requirements

```bash
Python 3.7+
numpy>=1.19.0
matplotlib>=3.3.0
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kiran1510/A-Star-Algorithm-for-Robotics-Path-Planning.git
   cd A-Star-Algorithm-for-Robotics-Path-Planning
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib
   ```

3. **Run an implementation:**
   ```bash
   python A_Star.py
   # or
   python A_Star_Complex.py
   # or
   python Bidirectional_A_Star.py
   ```

## Repository Structure

```
A-Star-Algorithm-for-Robotics-Path-Planning/
├── A_Star.py                  # Basic A* with simple maze
├── A_Star_Complex.py          # A* with dense, challenging maze
├── Bidirectional_A_Star.py    # Dual-direction search implementation
├── LICENSE                    # MIT License
└── README.md                  # This file
```

### File Descriptions

#### `A_Star.py` - Simple Implementation
- **Purpose**: Introduction to A* pathfinding concepts
- **Environment**: Simple maze with two interior walls
- **Grid**: 2.0m cell resolution
- **Start**: (10, 10)
- **Goal**: (50, 50)
- **Best for**: Learning the basics, understanding core algorithm

#### `A_Star_Complex.py` - Complex Implementation
- **Purpose**: Demonstrates robustness in constrained environments
- **Environment**: Dense maze with 30+ obstacle elements
- **Grid**: 1.0m cell resolution for precise navigation
- **Start**: (10, 10)
- **Goal**: (140, 80)
- **Features**: Multiple wall segments, scattered pillars, narrow corridors
- **Best for**: Testing algorithm performance, seeing complex path planning

#### `Bidirectional_A_Star.py` - Dual-Direction Search
- **Purpose**: Enhanced performance through simultaneous exploration
- **Environment**: Random obstacle generation (configurable density)
- **Grid**: 60x60 with random obstacles
- **Start**: (1, 1) - bottom-left corner
- **Goal**: (59, 59) - top-right corner
- **Optimization**: ~2x faster than standard A* in open spaces
- **Best for**: Understanding advanced search strategies, performance optimization

## Usage

### Basic Usage

Simply run any implementation:

```bash
python A_Star.py
```

The visualization will open showing:
- Initial environment with obstacles
- Real-time search exploration
- Final path highlighted in red

### Customization Options

#### Adjusting Grid Resolution
```python
grid_size = 2.0  # Larger = faster but less precise
grid_size = 1.0  # Smaller = slower but more accurate
```

#### Changing Start/Goal Positions
```python
sx = 10.0  # Start X coordinate [meters]
sy = 10.0  # Start Y coordinate [meters]
gx = 50.0  # Goal X coordinate [meters]
gy = 50.0  # Goal Y coordinate [meters]
```

#### Modifying Robot Safety Radius
```python
robot_radius = 1.0  # Obstacle inflation radius [meters]
```

#### Animation Speed Control (Bidirectional only)
```python
animation_delay = 0.05  # Seconds between visualization updates
# 0.2 = Slow, educational viewing
# 0.05 = Smooth, balanced (default)
# 0.01 = Fast execution
```

#### Obstacle Density (Bidirectional only)
```python
main(obstacle_number=800)   # ~22% density (default, good success rate)
main(obstacle_number=1200)  # ~33% density (challenging)
main(obstacle_number=1500)  # ~40% density (very difficult)
```

### Interactive Controls

While the program is running:
- **ESC**: Clean exit and close all windows
- **Window Close Button**: Standard close

## Algorithm Descriptions

### Standard A* (`A_Star.py`, `A_Star_Complex.py`)

**Algorithm Steps:**
1. Initialize open set with start node (f = g + h)
2. Select node with minimum f-value from open set
3. If selected node is goal, reconstruct path and exit
4. For each neighbor of selected node:
   - Calculate tentative g-value
   - If neighbor not in open set, add it
   - If neighbor in open set but new path is cheaper, update it
5. Move selected node to closed set
6. Repeat from step 2

**Heuristic Function:** Euclidean distance
```python
h(n) = √((x₂-x₁)² + (y₂-y₁)²)
```
- Admissible (never overestimates)
- Optimal for 8-connected grid movement

**Movement Model:** 8-connected grid (like chess king)
- Cardinal moves (↑↓←→): cost = 1.0
- Diagonal moves (↗↘↖↙): cost = √2 ≈ 1.414

**Key Optimizations:**
- Batch rendering: Updates display every 50 nodes instead of every node
- Dictionary-based sets: O(1) lookup for open/closed membership
- Obstacle inflation: Single collision check per node

### Bidirectional A* (`Bidirectional_A_Star.py`)

**Algorithm Steps:**
1. Initialize two searches: forward (from start) and backward (from goal)
2. Expand forward search toward backward search's best node
3. Expand backward search toward forward search's best node
4. Check if searches have met (any node in both closed sets)
5. If met, reconstruct complete path by joining both segments
6. Repeat from step 2

**Heuristic Function:** Manhattan distance
```python
h(n) = |x₂-x₁| + |y₂-y₁|
```
- Faster computation than Euclidean
- Still admissible for grid-based search

**Performance Advantage:**
- Each search explores roughly half the space
- Meeting point typically near the midpoint
- ~2x speedup in open or symmetric environments

**Visualization:** Updates every 5 iterations for smooth animation

## Visualizations

### Legend

**Standard A* (A_Star.py, A_Star_Complex.py)**

- 🟢 Green Circle: Start position
- 🔵 Blue X: Goal position  
- 🟦 Cyan X: Explored nodes (search frontier)
- ⬛ Black Dot: Obstacle
- 🔴 Red Line: Final optimal path

**Bidirectional A* (Bidirectional_A_Star.py)**

- 🔺 Blue Triangle: Start position
- ⭐ Blue Star: Goal position
- 🟡 Yellow Circle: Forward search exploration (start→goal)
- 🟢 Green Circle: Backward search exploration (goal→start)
- ⬛ Black Square: Obstacle
- 🔴 Red Line: Complete path
- ❌ Red X: Blocking boundary (when no path exists)

### Example Results

**Simple Maze (A_Star.py):**
```
Path found! Total distance: 67.48 meters
Waypoints: 35
Exploration pattern: Focused beam toward goal
```

**Complex Maze (A_Star_Complex.py):**
```
Path found! Total distance: 165.82 meters
Waypoints: 112
Exploration pattern: Dense exploration through narrow corridors
```

**Bidirectional (Bidirectional_A_Star.py):**
```
Start: [1, 1] → Goal: [59, 59]
Diagonal distance: ~82.0 units
Meeting point: ~[30, 30] (approximate midpoint)
Path length: Varies with random obstacle distribution
```

## Performance Considerations

### Computational Complexity

**Time Complexity:** O(b^d)
- b = branching factor (8 for 8-connected grid)
- d = depth of optimal solution

**Space Complexity:** O(b^d)
- Storage for open and closed sets

### Optimization Techniques Used

1. **Batch Rendering**
   - Standard A*: Updates every 50 nodes
   - Bidirectional A*: Updates every 5 iterations
   - Result: ~100x reduction in rendering overhead

2. **Dictionary-Based Sets**
   - O(1) lookup time for node membership
   - Much faster than list-based searching

3. **Grid Resolution Trade-offs**
   - Coarse (2.0m): Faster, suitable for initial planning
   - Fine (1.0m): Slower but more accurate paths

### Performance Tips

**For faster execution:**
```python
ENABLE_VISUALIZATION = False  # Disable animation
```

**For educational viewing:**
```python
animation_delay = 0.2  # Slow down to watch behavior
```

**For large environments:**
```python
# Update visualization less frequently
if len(closed_set) % 100 == 0:
    plt.plot(explored_x, explored_y, "xc", markersize=2)
```


## Advanced Topics

### Possible Extensions

1. **Weighted A* (ε-A*)**
   ```python
   heuristic_weight = 1.5  # > 1.0 for faster but suboptimal paths
   h = heuristic_weight * euclidean_distance(current, goal)
   ```

2. **Jump Point Search (JPS)**
   - Skip intermediate nodes on straight paths
   - 10-40x speedup for uniform grids

3. **Theta***
   - Any-angle paths (not restricted to grid)
   - Smoother, more natural trajectories

4. **Dynamic Replanning**
   - Integrate D* or D* Lite
   - Handle moving obstacles

### Real-World Applications

- **Mobile Robotics**: Warehouse navigation, autonomous delivery
- **Autonomous Vehicles**: Urban path planning with traffic
- **Drones**: Extend to 3D for aerial navigation
- **Mars Rovers**: NASA uses Field D* (based on A*)
- **Video Games**: NPC pathfinding, RTS unit movement

## Contributing

Contributions are welcome! Potential improvements:

- Additional heuristic functions (Chebyshev, Octile)
- 3D pathfinding extension
- Dynamic obstacle handling
- Path smoothing post-processing
- Integration with ROS
- Performance benchmarking suite
- Additional maze configurations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Kiran**
- Master's in Robotics - Northeastern University
- GitHub: [@Kiran1510](https://github.com/Kiran1510)

## Acknowledgments

- Original A* algorithm: Hart, Nilsson & Raphael (1968)
- Bidirectional search: Russell & Norvig, "AI: A Modern Approach"
- Visualization inspired by PythonRobotics repository

## References

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
2. Russell, S., & Norvig, P. "Artificial Intelligence: A Modern Approach" (4th Edition)
3. LaValle, S. M. "Planning Algorithms" - Cambridge University Press
4. Koenig, S., & Likhachev, M. "D* Lite" - AAAI 2002

## Known Issues

- Very high obstacle density (>40%) may result in no path found
- Large grids (>200x200) may slow visualization
- Matplotlib window requires manual close on some systems

---

*Last Updated: January 2026*