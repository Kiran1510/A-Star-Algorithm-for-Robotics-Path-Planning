# A* Algorithm for Robotics Path Planning

A comprehensive collection of A* pathfinding implementations for robotics applications, featuring standard A*, bidirectional A*, and extensively annotated educational variants.

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
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides multiple implementations of the A* pathfinding algorithm, specifically tailored for robotics applications. It includes both educational and optimized versions, demonstrating various pathfinding strategies from basic A* to advanced bidirectional search.

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

- **Multiple Implementations**: Standard, simple A*, Bidirectional A*, and more a complex variant
- **Real-time Visualization**: Watch the algorithm explore the search space
- **Optimized Performance**: Batch rendering for smooth visualization
- **Interactive Controls**: ESC key for clean exit, adjustable animation speed
- **Comprehensive Documentation**: Detailed inline comments explaining every step
- **Obstacle Handling**: Support for both static mazes and random obstacle generation
- **Path Metrics**: Distance calculations and waypoint counts
- **Educational Focus**: Extensively annotated code for learning

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
   python astar_simple.py
   # or
   python bidirectional_astar.py
   ```

## Repository Structure

```
A-Star-Algorithm-for-Robotics-Path-Planning/
├── astar_simple.py                    # Basic A* with simple maze
├── astar_complex.py                   # A* with complex maze
├── astar_rewritten.py                 # Refactored A* with enhanced documentation
├── bidirectional_astar.py             # Dual-direction search implementation
├── bidirectional_astar_rewritten.py   # Refactored bidirectional variant
├── README.md                          # This file
└── examples/                          # Example outputs and visualizations
```

### File Descriptions

#### Standard A* Implementations

**`astar_simple.py`**
- Clean implementation with a basic maze
- Two interior walls creating a simple navigation challenge
- Perfect for understanding core A* concepts
- Start: (10, 10), Goal: (50, 50)
- Grid resolution: 2.0m cells

**`astar_complex.py`**
- Dense maze with multiple obstacles
- 30+ distinct obstacle elements
- Demonstrates algorithm robustness in constrained spaces
- 10+ major wall segments and scattered pillars
- Grid resolution: 1.0m cells for precise navigation

**`astar_rewritten.py`**
- Completely refactored implementation
- Professional code structure with descriptive naming
- Enhanced documentation and annotations
- Same functionality, different perspective
- Ideal for code review and learning

#### Bidirectional A* Implementations

**`bidirectional_astar.py`**
- Searches from both start and goal simultaneously
- ~2x performance improvement in open spaces
- Random obstacle generation (configurable density)
- Diagonal corner placement for maximum challenge
- Detects and visualizes blocked scenarios

**`bidirectional_astar_rewritten.py`**
- Refactored bidirectional implementation
- Improved variable naming and structure
- Comprehensive inline documentation
- Professional code organization

## Usage

### Basic Usage

Run any implementation directly:

```bash
python astar_simple.py
```

### Customization

#### Adjusting Grid Resolution
```python
grid_size = 2.0  # Cell size in meters (smaller = finer resolution)
```

#### Changing Start/Goal Positions
```python
sx = 10.0  # Start X coordinate
sy = 10.0  # Start Y coordinate
gx = 50.0  # Goal X coordinate
gy = 50.0  # Goal Y coordinate
```

#### Modifying Obstacle Density (Bidirectional)
```python
main(obstacle_number=800)  # ~22% density (default)
main(obstacle_number=1500) # ~40% density (very challenging)
```

#### Animation Speed Control
```python
animation_delay = 0.05  # Seconds between updates
# 0.2 = Slow, educational
# 0.05 = Smooth (default)
# 0.001 = Fast
```

### Interactive Controls

While running:
- **ESC**: Clean exit and close visualization
- **Close Window**: Standard window close

## Algorithm Descriptions

### Standard A* (`astar_simple.py`, `astar_complex.py`)

**Algorithm Flow:**
1. Initialize open set with start node
2. Select node with minimum f(n) from open set
3. If goal reached, reconstruct path
4. Expand neighbors, update costs
5. Move current to closed set
6. Repeat from step 2

**Heuristic:** Euclidean distance (straight-line distance to goal)

**Movement Model:** 8-connected grid (like chess king)
- Cardinal moves (↑↓←→): cost = 1.0
- Diagonal moves (↗↘↖↙): cost = √2 ≈ 1.414

**Optimizations:**
- Batch rendering (update every 50-100 nodes)
- Dictionary-based open/closed sets for O(1) lookup
- Obstacle inflation for robot safety radius

### Bidirectional A* (`bidirectional_astar.py`)

**Algorithm Flow:**
1. Initialize searches from both start and goal
2. Expand forward search toward goal's best node
3. Expand backward search toward start's best node
4. Check if searches have met
5. If met, reconstruct path from both directions
6. Repeat from step 2

**Advantages:**
- ~2x faster in open environments
- Each search explores roughly half the space
- Better for symmetric scenarios

**Heuristic:** Manhattan distance (|Δx| + |Δy|)

**Meeting Condition:** When any node appears in both closed sets

**Performance:** Updates visualization every 5 iterations for smooth animation

## Visualizations

### Color Scheme

**Standard A***
- 🟢 **Green Circle**: Start position
- 🔵 **Blue X**: Goal position
- 🟦 **Cyan X's**: Explored nodes (search frontier)
- ⬛ **Black Dots**: Obstacles
- 🔴 **Red Line**: Final optimal path

**Bidirectional A***
- 🔺 **Blue Triangle**: Start position
- ⭐ **Blue Star**: Goal position
- 🟡 **Yellow Circles**: Forward search exploration (start→goal)
- 🟢 **Green Circles**: Backward search exploration (goal→start)
- ⬛ **Black Squares**: Obstacles
- 🔴 **Red Line**: Complete path
- ❌ **Red X's**: Blocking boundary (if no path exists)

### Example Outputs

**Simple Maze:**
```
Start: (10, 10) → Goal: (50, 50)
Path length: ~67.5 meters
Waypoints: 35
```

**Complex Maze:**
```
Start: (10, 10) → Goal: (140, 80)
Path length: ~165.8 meters
Waypoints: 112
Multiple corridor navigation required
```

**Bidirectional (Diagonal):**
```
Start: (1, 1) → Goal: (59, 59)
Direct distance: ~82 units
Meeting point: ~(30, 30)
Path length varies with obstacle distribution
```

## Performance Considerations

### Computational Complexity

**Time Complexity:** O(b^d) where:
- b = branching factor (8 for 8-connected grid)
- d = depth of optimal solution

**Space Complexity:** O(b^d) for storing open and closed sets

### Optimization Techniques

1. **Batch Rendering**
   - Plot nodes in groups rather than individually
   - Reduces matplotlib overhead by ~100x
   - Default: update every 50 nodes

2. **Dictionary-Based Sets**
   - O(1) lookup for open/closed membership
   - Faster than list-based searches

3. **Grid Resolution Trade-offs**
   - Finer grid (1.0m): More accurate, slower
   - Coarser grid (2.0m): Faster, less precise

4. **Heuristic Selection**
   - Euclidean: More accurate for 8-connected movement
   - Manhattan: Faster computation, admissible for grid-based search

### Performance Tips

**For Large Environments:**
```python
# Reduce visualization frequency
if len(closed_set) % 100 == 0:  # Update every 100 nodes instead of 50
    plt.plot(explored_x, explored_y, "xc", markersize=2)
```

**For Real-Time Applications:**
```python
show_animation = False  # Disable visualization for maximum speed
```

**For Educational Purposes:**
```python
animation_delay = 0.2  # Slow down to watch algorithm behavior
```


### Learning Path

1. Start with `astar_simple.py` - understand basic concepts
2. Study `astar_complex.py` - see robustness in constrained spaces
3. Explore `bidirectional_astar.py` - learn optimization strategies
4. Review rewritten versions - see professional code organization

## Advanced Topics

### Variations to Explore

1. **Weighted A*** (ε-A*)
   ```python
   w = 1.5  # Weight > 1.0
   d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
   ```
   - Faster but suboptimal
   - Guarantees solution within (1+ε) of optimal

2. **Jump Point Search (JPS)**
   - 10-40x faster for uniform grids
   - Skips intermediate nodes in straight paths

3. **Theta***
   - Any-angle paths (not restricted to grid edges)
   - Smoother, more natural paths

4. **Dynamic Obstacles**
   - Integrate D* or D* Lite for replanning
   - Handle moving obstacles

### Real-World Applications

- **Mobile Robotics**: Warehouse navigation, delivery robots
- **Autonomous Vehicles**: Urban path planning
- **Drones**: 3D pathfinding (extend to 3D grid)
- **Video Games**: NPC movement, strategy games
- **Mars Rovers**: Used Field D* based on A*

## Contributing

Contributions are welcome! Areas for improvement:

- Additional heuristic functions
- 3D pathfinding extension
- Dynamic obstacle handling
- Performance benchmarking suite
- Additional maze configurations
- Path smoothing post-processing
- Integration with ROS (Robot Operating System)

## License

This project is open source and available under the MIT License.

## Author

**Kiran**
- Master's in Robotics - Northeastern University
- GitHub: [@Kiran1510](https://github.com/Kiran1510)

## Acknowledgments

- Original A* algorithm by Peter Hart, Nils Nilsson, and Bertram Raphael (1968)
- Bidirectional search concepts from Russell & Norvig's "Artificial Intelligence: A Modern Approach"
- Visualization techniques inspired by PythonRobotics repository

## References

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
2. Russell, S., & Norvig, P. "Artificial Intelligence: A Modern Approach" (4th Edition)
3. LaValle, S. M. "Planning Algorithms" - Cambridge University Press
4. Koenig, S., & Likhachev, M. "D* Lite" - AAAI Conference on Artificial Intelligence

## Known Issues

- Very high obstacle density (>40%) may result in no path
- Large grids (>200x200) may have slow visualization
- matplotlib window must be closed manually on some systems



*Last Updated: January 2026*