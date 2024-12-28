import networkx as nx
from collections import deque
import matplotlib.pyplot as plt

# Set of nodes that have been visited during traversal
visited = set()

def display_road_network(cities, roads, search_strategy=None, start_city=None, goal_city=None):
    """
    Renders a visualization of the road network and optionally highlights the path found 
    using a specified search method.

    Parameters:
    - cities: A list of city names.
    - roads: A dictionary where keys are cities and values are lists of tuples 
      representing neighboring cities and the distance to them.
    - search_strategy: The search method to use ('bfs', 'dfs', or 'weighted_bfs').
    - start_city: The city where the journey starts (used for path visualization).
    - goal_city: The target city to reach (used for path visualization).
    """
    # Initialize an empty graph to represent the road network
    graph = nx.Graph()
    
    # Add edges to the graph with weights representing distances
    for city, neighbors in roads.items():
        for neighbor, distance in neighbors:
            graph.add_edge(city, neighbor, weight=distance)
    
    # Set up the layout for the graph visualization
    layout = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))
    
    # Assign colors to nodes based on their status (start, goal, or regular city)
    node_colors = []
    for node in graph.nodes():
        if node == start_city:
            node_colors.append('lightgreen')  # Start city highlighted in green
        elif node == goal_city:
            node_colors.append('lightcoral')  # Goal city highlighted in red
        else:
            node_colors.append('lightblue')  # Other cities in blue
    
    # Draw nodes, labels, edges, and edge labels
    nx.draw_networkx_nodes(graph, layout, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(graph, layout)
    nx.draw_networkx_edges(graph, layout)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels)
    
    # If a search strategy and start/goal cities are provided, display the path
    if search_strategy and start_city and goal_city:
        path, total_cost = find_path(cities, roads, start_city, goal_city, search_strategy)
        if path:
            path_edges = list(zip(path[:-1], path[1:]))  # Prepare edges for visualization
            nx.draw_networkx_edges(graph, layout, edgelist=path_edges, edge_color='red', width=2)
            plt.title(f"{search_strategy.upper()} Path - Total Distance: {total_cost} km")
        else:
            plt.title(f"No path found using {search_strategy.upper()}")
    else:
        plt.title("Road Network Overview")
    
    # Hide axes for better presentation
    plt.axis('off')
    plt.show()

def find_path(cities, roads, start_city, goal_city, search_strategy):
    """
    Finds a path between the start and goal cities using the specified search strategy.

    Parameters:
    - cities: List of city names.
    - roads: Dictionary with city connections and distances.
    - start_city: The city from where the search begins.
    - goal_city: The city to reach.
    - search_strategy: The search algorithm to apply ('bfs', 'dfs', 'weighted_bfs').
    
    Returns:
    - path: List of cities representing the path from start to goal.
    - total_cost: The total distance or cost of the path.
    """
    visited.clear()  # Reset visited cities for a fresh search
    
    # Call the appropriate search algorithm based on the selected strategy
    if search_strategy == 'bfs':
        return breadth_first_search(cities, roads, start_city, goal_city)
    elif search_strategy == 'dfs':
        return depth_first_search(cities, roads, start_city, goal_city)
    elif search_strategy == 'weighted_bfs':
        return weighted_bfs(cities, roads, start_city, goal_city)
    else:
        raise ValueError("Invalid search strategy. Must be one of 'bfs', 'dfs', or 'weighted_bfs'.")

def explore_all_paths(cities, roads, start_city, search_strategy):
    """
    Finds and prints paths from the start city to all other cities using the chosen search method.
    
    Parameters:
    - cities: List of all cities.
    - roads: Dictionary of city connections.
    - start_city: The city from which the exploration begins.
    - search_strategy: The strategy used to find paths ('bfs', 'dfs', or 'weighted_bfs').
    
    Returns:
    - paths_with_costs: List of tuples containing the path and cost to each goal city.
    """
    paths_with_costs = []
    
    # Iterate through all cities, finding paths to each one
    for goal_city in cities:
        if goal_city != start_city:
            path, cost = find_path(cities, roads, start_city, goal_city, search_strategy)
            if path:
                paths_with_costs.append((path, cost))
                print(f"{search_strategy.upper()} Path from {start_city} to {goal_city}:")
                print(f"Path: {' -> '.join(path)}")
                print(f"Total Distance: {cost} km\n")
            else:
                print(f"No path found from {start_city} to {goal_city}\n")
                
    return paths_with_costs

def breadth_first_search(cities, roads, start_city, goal_city=None):
    """
    Implements the Breadth-First Search (BFS) algorithm to find the shortest path.

    Parameters:
    - cities: List of cities.
    - roads: Dictionary of city connections with distances.
    - start_city: The starting city for the search.
    - goal_city: The target city to reach (optional).
    
    Returns:
    - path: A list of cities representing the found path.
    - total_cost: The total distance or cost of the path.
    """
    if start_city == goal_city:
        return ([start_city], 0)  # Return immediately if start equals goal
    
    # Initialize BFS structures
    queue = deque([start_city])  # Queue for BFS traversal
    paths = {start_city: (None, 0)}  # Dictionary to track paths and their costs
    
    # Perform BFS traversal
    while queue:
        current = queue.popleft()
        visited.add(current)
        
        if current == goal_city:
            break  # Goal found, terminate early
            
        for neighbor, distance in roads[current]:
            if neighbor not in visited and neighbor not in paths:
                queue.append(neighbor)
                paths[neighbor] = (current, distance)
    
    # If goal city is not reached, return no path
    if goal_city not in paths:
        return (None, 0)
        
    # Reconstruct the path from start to goal
    path = []
    total_cost = 0
    current = goal_city
    
    while current:
        path.append(current)
        prev, cost = paths[current]
        total_cost += cost
        current = prev
        
    return (path[::-1], total_cost)

def depth_first_search(cities, roads, start_city, goal_city=None):
    """
    Performs Depth-First Search (DFS) to explore a path from start to goal.

    Parameters:
    - cities: List of cities.
    - roads: Dictionary of city connections.
    - start_city: The city from where the DFS starts.
    - goal_city: The city to reach (optional).
    
    Returns:
    - path: List of cities from start to goal.
    - total_cost: Total cost (distance) of the path.
    """
    visited.add(start_city)
    path = [start_city]
    total_cost = 0

    if start_city == goal_city:
        return (path, total_cost)

    for neighbor in roads[start_city]:
        if neighbor[0] not in visited:
            new_path, new_cost = depth_first_search(cities, roads, neighbor[0], goal_city)
            if new_path:
                path.extend(new_path)
                total_cost += (neighbor[1] + new_cost)
                return (path, total_cost)
    
    return (None, 0)

def weighted_bfs(cities, roads, start_city, goal_city=None):
    """
    Performs a Weighted BFS considering the road distances (weights).

    Parameters:
    - cities: List of city names.
    - roads: Dictionary of city connections and their respective distances.
    - start_city: The starting point for BFS.
    - goal_city: The target city to reach (optional).
    
    Returns:
    - path: The shortest path found from start to goal.
    - total_cost: The total cost (distance) of the path.
    """
    if start_city == goal_city:
        return ([start_city], 0)  # Return immediately if start equals goal

    paths = {start_city: (None, 0)}  # Store the paths and their respective costs
    queue = deque([start_city])

    while queue:
        current = queue.popleft()
        visited.add(current)
        cost_to_current = paths[current][1]

        for neighbor, step_cost in roads[current]:
            total_cost = cost_to_current + step_cost
            
            # Update path if a shorter one is found
            if neighbor not in paths or total_cost < paths[neighbor][1]:
                paths[neighbor] = (current, total_cost)
                if neighbor
