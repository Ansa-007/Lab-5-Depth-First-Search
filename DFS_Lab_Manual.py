"""
Professional Depth-First Search (DFS) Implementation
Industry-Grade Graph Traversal Algorithm

This module provides a comprehensive DFS implementation with:
- Recursive and iterative approaches
- Performance optimizations
- Error handling and validation
- Industry-standard coding practices
"""

from typing import Dict, List, Set, Any, Optional, Union
from collections import deque
import sys
import time


class GraphTraversalError(Exception):
    """Custom exception for graph traversal errors."""
    pass


def validate_graph(graph: Dict[Any, List[Any]]) -> None:
    """
    Validate graph structure for DFS traversal.
    
    Args:
        graph: Dictionary representing adjacency list
        
    Raises:
        GraphTraversalError: If graph structure is invalid
    """
    if not isinstance(graph, dict):
        raise GraphTraversalError("Graph must be a dictionary")
    
    for node, neighbors in graph.items():
        if not isinstance(neighbors, list):
            raise GraphTraversalError(f"Neighbors of node {node} must be a list")
        
        for neighbor in neighbors:
            if neighbor not in graph:
                raise GraphTraversalError(f"Neighbor {neighbor} not found in graph keys")


def dfs_recursive(
    graph: Dict[Any, List[Any]], 
    start: Any, 
    visited: Optional[Set[Any]] = None,
    path: Optional[List[Any]] = None
) -> List[Any]:
    """
    Depth-First Search using recursive approach.
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node for traversal
        visited: Set of visited nodes (for internal recursion)
        path: Current traversal path (for internal recursion)
        
    Returns:
        List of nodes in DFS traversal order
        
    Raises:
        GraphTraversalError: If start node not found or graph is invalid
    """
    if visited is None:
        visited = set()
    if path is None:
        path = []
    
    # Input validation
    if start not in graph:
        raise GraphTraversalError(f"Start node {start} not found in graph")
    
    visited.add(start)
    path.append(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, path)
    
    return path


def dfs_iterative(graph: Dict[Any, List[Any]], start: Any) -> List[Any]:
    """
    Depth-First Search using iterative approach with explicit stack.
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node for traversal
        
    Returns:
        List of nodes in DFS traversal order
        
    Raises:
        GraphTraversalError: If start node not found or graph is invalid
    """
    if start not in graph:
        raise GraphTraversalError(f"Start node {start} not found in graph")
    
    visited = set()
    stack = [start]
    path = []
    
    while stack:
        node = stack.pop()
        
        if node not in visited:
            visited.add(node)
            path.append(node)
            
            # Add neighbors to stack in reverse order for consistent traversal
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return path


def dfs_with_path_tracking(
    graph: Dict[Any, List[Any]], 
    start: Any,
    target: Optional[Any] = None
) -> Dict[str, Union[List[Any], bool]]:
    """
    DFS with comprehensive path tracking and target search.
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node for traversal
        target: Target node to search for (optional)
        
    Returns:
        Dictionary containing:
        - 'path': Full traversal path
        - 'found': Boolean indicating if target was found
        - 'target_path': Path to target if found, empty otherwise
    """
    visited = set()
    stack = [(start, [start])]
    full_path = []
    target_found = False
    target_path = []
    
    while stack:
        node, current_path = stack.pop()
        
        if node not in visited:
            visited.add(node)
            full_path.append(node)
            
            # Check if target found
            if target is not None and node == target:
                target_found = True
                target_path = current_path
                break
            
            # Add neighbors to stack
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append((neighbor, current_path + [neighbor]))
    
    return {
        'path': full_path,
        'found': target_found,
        'target_path': target_path
    }


def dfs_with_timestamps(graph: Dict[Any, List[Any]], start: Any) -> Dict[str, Dict[Any, List[int]]]:
    """
    DFS with entry and exit timestamps for each node.
    Useful for advanced graph algorithms like topological sort.
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node for traversal
        
    Returns:
        Dictionary with 'entry' and 'exit' timestamps for each node
    """
    visited = set()
    entry_time = {}
    exit_time = {}
    timestamp = [0]  # Use list for mutable integer
    
    def dfs_util(node: Any):
        timestamp[0] += 1
        entry_time[node] = timestamp[0]
        visited.add(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs_util(neighbor)
        
        timestamp[0] += 1
        exit_time[node] = timestamp[0]
    
    dfs_util(start)
    
    return {
        'entry': entry_time,
        'exit': exit_time
    }


def performance_comparison(graph: Dict[Any, List[Any]], start: Any) -> Dict[str, Dict[str, float]]:
    """
    Compare performance between recursive and iterative DFS implementations.
    
    Args:
        graph: Dictionary representing adjacency list
        start: Starting node for traversal
        
    Returns:
        Dictionary with timing results for each implementation
    """
    results = {}
    
    # Test recursive DFS
    start_time = time.perf_counter()
    recursive_path = dfs_recursive(graph, start)
    end_time = time.perf_counter()
    results['recursive'] = {
        'time': end_time - start_time,
        'path_length': len(recursive_path)
    }
    
    # Test iterative DFS
    start_time = time.perf_counter()
    iterative_path = dfs_iterative(graph, start)
    end_time = time.perf_counter()
    results['iterative'] = {
        'time': end_time - start_time,
        'path_length': len(iterative_path)
    }
    
    return results


def main():
    """
    Main demonstration function with industry-grade examples.
    """
    # Industry example: Network topology
    network_graph = {
        'Router_A': ['Switch_B', 'Switch_C'],
        'Switch_B': ['Server_D', 'Server_E'],
        'Switch_C': ['Firewall_F'],
        'Server_D': [],
        'Server_E': [],
        'Firewall_F': []
    }
    
    # Software dependency graph example
    dependency_graph = {
        'main_app': ['auth_module', 'database', 'ui'],
        'auth_module': ['crypto_lib', 'user_service'],
        'database': ['connection_pool', 'query_builder'],
        'ui': ['components', 'styles'],
        'crypto_lib': [],
        'user_service': ['database'],
        'connection_pool': [],
        'query_builder': [],
        'components': [],
        'styles': []
    }
    
    print("=== Professional DFS Implementation Demo ===\n")
    
    try:
        # Validate graphs
        validate_graph(network_graph)
        validate_graph(dependency_graph)
        
        # Network topology traversal
        print("1. Network Topology Traversal:")
        print("   Graph:", network_graph)
        
        network_path_recursive = dfs_recursive(network_graph, 'Router_A')
        network_path_iterative = dfs_iterative(network_graph, 'Router_A')
        
        print(f"   Recursive DFS path: {network_path_recursive}")
        print(f"   Iterative DFS path: {network_path_iterative}")
        
        # Target search in network
        target_result = dfs_with_path_tracking(network_graph, 'Router_A', 'Server_E')
        print(f"   Path to Server_E: {target_result['target_path']}")
        print(f"   Target found: {target_result['found']}")
        
        # Software dependency analysis
        print("\n2. Software Dependency Analysis:")
        print("   Graph:", dependency_graph)
        
        dep_path = dfs_recursive(dependency_graph, 'main_app')
        print(f"   Dependency resolution order: {dep_path}")
        
        # Timestamp analysis for topological sorting
        timestamps = dfs_with_timestamps(dependency_graph, 'main_app')
        print(f"   Entry timestamps: {timestamps['entry']}")
        print(f"   Exit timestamps: {timestamps['exit']}")
        
        # Performance comparison
        print("\n3. Performance Comparison:")
        perf_results = performance_comparison(dependency_graph, 'main_app')
        for method, stats in perf_results.items():
            print(f"   {method.capitalize()} DFS: {stats['time']:.6f}s, "
                  f"Path length: {stats['path_length']}")
        
        # Original example from lab manual
        print("\n4. Original Lab Example:")
        original_graph = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': [], 'E': [], 'F': []
        }
        
        original_path = dfs_recursive(original_graph, 'A')
        print(f"   Original DFS path: {original_path}")
        print(f"   Expected: ['A', 'B', 'D', 'E', 'C', 'F']")
        print(f"   Match: {original_path == ['A', 'B', 'D', 'E', 'C', 'F']}")
        
    except GraphTraversalError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Set recursion limit for deep graphs
    sys.setrecursionlimit(10000)
    main()
