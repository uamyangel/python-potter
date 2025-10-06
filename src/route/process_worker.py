"""
Multi-processing worker for parallel routing.

This module provides process-safe worker functions that bypass Python's GIL,
enabling true parallel execution across multiple CPU cores.

FIRST PRINCIPLES:
- Avoid pickle circular references (RouteNode.children ↔ parents)
- Use shared memory for large NumPy arrays (zero-copy across processes)
- Only serialize small metadata (IDs, shapes, connection info)
- Rebuild minimal data structures in worker processes
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from ..db.route_node import Connection, RouteNode, Net
from ..db.route_node_graph import RouteNodeGraph
from ..db.database import Database
from .astar_route import AStarRouter, NodeScratch
from .shared_memory_manager import attach_shared_array
from ..utils.log import log
from ..global_defs import NodeType, IntentCode


# Global state initialized once per worker process
_worker_graph: RouteNodeGraph = None
_worker_nets: List[Net] = None
_worker_connections: Dict[Tuple[int, int], Connection] = None  # (net_id, conn_id) → Connection
_worker_router: AStarRouter = None
_worker_node_scratch: NodeScratch = None
_worker_id: int = -1
_worker_shm_refs: List = []  # Keep SharedMemory objects alive to prevent segfault


def init_worker(
    database_state: Dict,
    worker_id: int,
    present_cong_factor: float,
    historical_cong_factor: float
):
    """
    Initialize worker process with lightweight database state.

    FIRST PRINCIPLES: Rebuild minimal data structures from shared memory + metadata.
    - Large NumPy arrays: attached from shared memory (zero-copy)
    - Small metadata: passed via pickle (connections, nets, CSR indices)
    - No RouteNode.children/parents lists (use CSR for adjacency)
    - No circular references (safe for pickle)
    - Memory footprint: ~O(1) per worker (not O(n))

    This is called once when the worker process starts via ProcessPoolExecutor's
    initializer parameter, avoiding repeated serialization overhead.

    Args:
        database_state: Serialized database state (metadata + shared memory refs)
        worker_id: Unique worker ID
        present_cong_factor: Present congestion factor
        historical_cong_factor: Historical congestion factor
    """
    global _worker_graph, _worker_nets, _worker_connections
    global _worker_router, _worker_node_scratch, _worker_id, _worker_shm_refs

    _worker_id = worker_id
    _worker_shm_refs = []  # Clear shared memory references

    # Rebuild routing graph from shared memory + metadata
    _worker_graph = RouteNodeGraph()
    _worker_graph.num_nodes = database_state['num_nodes']
    _worker_graph.num_edges = database_state['num_edges']

    # Restore CSR adjacency from shared memory (ZERO-COPY!)
    shm_metadata = database_state['shared_arrays']
    _worker_graph.csr_indptr, shm = attach_shared_array(
        shm_metadata['csr_indptr']['shm_name'],
        shm_metadata['csr_indptr']['shape'],
        shm_metadata['csr_indptr']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.csr_indices, shm = attach_shared_array(
        shm_metadata['csr_indices']['shm_name'],
        shm_metadata['csr_indices']['shape'],
        shm_metadata['csr_indices']['dtype']
    )
    _worker_shm_refs.append(shm)
    _worker_graph._has_csr = True

    # Attach to shared NumPy arrays (ZERO-COPY!)
    # CRITICAL: Keep SharedMemory references alive to prevent segfault!
    
    _worker_graph.tile_x_arr, shm = attach_shared_array(
        shm_metadata['tile_x']['shm_name'],
        shm_metadata['tile_x']['shape'],
        shm_metadata['tile_x']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.tile_y_arr, shm = attach_shared_array(
        shm_metadata['tile_y']['shm_name'],
        shm_metadata['tile_y']['shape'],
        shm_metadata['tile_y']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.base_cost_arr, shm = attach_shared_array(
        shm_metadata['base_cost']['shm_name'],
        shm_metadata['base_cost']['shape'],
        shm_metadata['base_cost']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.length_arr, shm = attach_shared_array(
        shm_metadata['length']['shm_name'],
        shm_metadata['length']['shape'],
        shm_metadata['length']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.node_type_arr, shm = attach_shared_array(
        shm_metadata['node_type']['shm_name'],
        shm_metadata['node_type']['shape'],
        shm_metadata['node_type']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.intent_code_arr, shm = attach_shared_array(
        shm_metadata['intent_code']['shm_name'],
        shm_metadata['intent_code']['shape'],
        shm_metadata['intent_code']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph.capacity_arr, shm = attach_shared_array(
        shm_metadata['capacity']['shm_name'],
        shm_metadata['capacity']['shape'],
        shm_metadata['capacity']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    # Congestion cost arrays: need MUTABLE copies (each worker modifies independently)
    pres_shared, shm = attach_shared_array(
        shm_metadata['pres_cong_cost']['shm_name'],
        shm_metadata['pres_cong_cost']['shape'],
        shm_metadata['pres_cong_cost']['dtype']
    )
    _worker_shm_refs.append(shm)
    _worker_graph.pres_cong_cost_arr = pres_shared.copy()  # Mutable copy
    
    hist_shared, shm = attach_shared_array(
        shm_metadata['hist_cong_cost']['shm_name'],
        shm_metadata['hist_cong_cost']['shape'],
        shm_metadata['hist_cong_cost']['dtype']
    )
    _worker_shm_refs.append(shm)
    _worker_graph.hist_cong_cost_arr = hist_shared.copy()  # Mutable copy
    
    _worker_graph.is_accessible_arr, shm = attach_shared_array(
        shm_metadata['is_accessible']['shm_name'],
        shm_metadata['is_accessible']['shape'],
        shm_metadata['is_accessible']['dtype']
    )
    _worker_shm_refs.append(shm)
    
    _worker_graph._numpy_arrays_built = True

    # Build minimal RouteNode objects (NO children/parents to avoid circular refs)
    _worker_graph.nodes = []
    _worker_graph.node_map = {}

    for node_id in range(_worker_graph.num_nodes):
        # Create RouteNode with attributes from NumPy arrays
        rnode = RouteNode(
            id=node_id,
            tile_id=0,  # Not needed for routing
            tile_x=int(_worker_graph.tile_x_arr[node_id]),
            tile_y=int(_worker_graph.tile_y_arr[node_id]),
            wire_id=0,  # Not needed
            node_type=NodeType(int(_worker_graph.node_type_arr[node_id])),
            intent_code=IntentCode(int(_worker_graph.intent_code_arr[node_id])),
            base_cost=float(_worker_graph.base_cost_arr[node_id]),
            length=int(_worker_graph.length_arr[node_id]),
            is_accessible_wire=bool(_worker_graph.is_accessible_arr[node_id]),
            children=[],  # EMPTY - use CSR instead!
            parents=[],   # EMPTY - use CSR instead!
            capacity=int(_worker_graph.capacity_arr[node_id]),
            present_congestion_cost=float(_worker_graph.pres_cong_cost_arr[node_id]),
            historical_congestion_cost=float(_worker_graph.hist_cong_cost_arr[node_id]),
        )
        _worker_graph.nodes.append(rnode)
        _worker_graph.node_map[node_id] = rnode

    # Rebuild Net objects from metadata
    _worker_nets = []
    for net_meta in database_state['nets_meta']:
        net = Net(
            id=net_meta['id'],
            name=net_meta['name'],
            x_min=net_meta['x_min'],
            y_min=net_meta['y_min'],
            x_max=net_meta['x_max'],
            y_max=net_meta['y_max'],
            center_x=net_meta['center_x'],
            center_y=net_meta['center_y'],
            hpwl=net_meta['hpwl'],
            connections=[],  # Will be populated on-demand
        )
        _worker_nets.append(net)

    # Rebuild Connection objects from metadata
    _worker_connections = {}
    for conn_meta in database_state['connections_meta']:
        net_id = conn_meta['net_id']
        conn_id = conn_meta['conn_id']

        # Get source/sink RouteNodes by ID
        source_node = _worker_graph.node_map[conn_meta['source_id']]
        sink_node = _worker_graph.node_map[conn_meta['sink_id']]

        connection = Connection(
            id=conn_id,
            net_id=net_id,
            source_node=source_node,
            sink_node=sink_node,
            x_min=conn_meta['x_min'],
            y_min=conn_meta['y_min'],
            x_max=conn_meta['x_max'],
            y_max=conn_meta['y_max'],
            center_x=conn_meta['center_x'],
            center_y=conn_meta['center_y'],
            hpwl=conn_meta['hpwl'],
            is_indirect=conn_meta['is_indirect'],
            # Restore pre-expanded bbox
            x_min_bb=conn_meta['x_min_bb'],
            y_min_bb=conn_meta['y_min_bb'],
            x_max_bb=conn_meta['x_max_bb'],
            y_max_bb=conn_meta['y_max_bb'],
        )

        _worker_connections[(net_id, conn_id)] = connection

        # Add to net's connections list
        net = _worker_nets[net_id]
        net.connections.append(connection)

    # Create minimal Database wrapper for AStarRouter
    # (AStarRouter needs database.routing_graph and database.nets)
    worker_database = Database()
    worker_database.routing_graph = _worker_graph
    worker_database.nets = _worker_nets

    # Create per-worker router and scratch space
    _worker_router = AStarRouter(
        worker_database,
        present_cong_factor,
        historical_cong_factor
    )

    _worker_node_scratch = NodeScratch(_worker_graph.num_nodes)


def route_partition_worker(
    partition_connections: List[Tuple[int, int]],  # (net_id, conn_id) pairs
    iteration: int,
    connection_id_base: int,
    needs_ripup: bool
) -> Dict:
    """
    Route all connections in a partition (worker function).

    This runs in a separate process, bypassing GIL.

    Args:
        partition_connections: List of (net_id, conn_id) tuples to route
        iteration: Current routing iteration
        connection_id_base: Base ID for unique connection IDs
        needs_ripup: Whether to ripup existing routes

    Returns:
        Dictionary containing routing results and statistics
    """
    global _worker_router, _worker_node_scratch, _worker_connections, _worker_nets, _worker_id

    if _worker_router is None:
        raise RuntimeError("Worker not initialized! Call init_worker first.")

    routed_count = 0
    failed_count = 0
    rerouted_count = 0

    # Results to return
    routed_paths: List[Tuple[int, int, List[int]]] = []  # (net_id, conn_id, [node_ids])
    node_usage_updates: Dict[int, Dict[int, int]] = {}  # node_id -> {net_id -> count}

    for net_id, conn_id in partition_connections:
        # Get connection from worker's connection map
        connection = _worker_connections.get((net_id, conn_id))
        if connection is None:
            continue

        net = _worker_nets[net_id]

        # Decide if routing needed
        needs_route = _worker_router.should_route(connection)
        if not needs_route:
            continue

        # Ripup on subsequent iterations
        if needs_ripup and iteration > 0:
            # Clear route (don't update global state yet - just local)
            connection.route_nodes.clear()
            connection.is_routed = False

        # Generate unique connection ID
        conn_unique_id = connection.id + connection_id_base

        # Route connection
        success = _worker_router.route_one_connection(
            connection,
            _worker_node_scratch,
            conn_unique_id,
            tid=_worker_id,
            sync=False  # Async mode - don't update global costs
        )

        if success:
            routed_count += 1
            rerouted_count += 1

            # Record routed path (as node IDs for serialization)
            node_ids = [rn.id for rn in connection.route_nodes]
            routed_paths.append((net_id, conn_id, node_ids))

            # Record node usage updates (for later merging)
            for rnode in connection.route_nodes:
                if rnode.id not in node_usage_updates:
                    node_usage_updates[rnode.id] = {}
                if net_id not in node_usage_updates[rnode.id]:
                    node_usage_updates[rnode.id][net_id] = 0
                node_usage_updates[rnode.id][net_id] += 1
        else:
            failed_count += 1

    return {
        'worker_id': _worker_id,
        'routed_count': routed_count,
        'failed_count': failed_count,
        'rerouted_count': rerouted_count,
        'routed_paths': routed_paths,
        'node_usage_updates': node_usage_updates
    }


def prepare_database_state(database: Database, shm_manager) -> Dict:
    """
    Extract minimal database state for worker initialization.

    FIRST PRINCIPLES: Use shared memory for ALL large arrays (>10MB).
    - Large NumPy arrays: shared memory (zero-copy)
    - Large CSR indices: shared memory (zero-copy) ← CRITICAL FIX!
    - Small metadata (<1MB): pickled (acceptable overhead)
    - No object references (RouteNode.children/parents avoided)
    
    Memory scaling:
    - Old approach: O(n × num_workers) - each worker gets full copy
    - New approach: O(n) - all large data shared, only small metadata copied

    Args:
        database: Full database object
        shm_manager: SharedMemoryManager for creating shared arrays

    Returns:
        Serializable state dictionary with shared memory references
    """
    from ..utils.log import log
    
    graph = database.routing_graph

    state = {
        'num_nodes': graph.num_nodes,
        'num_edges': graph.num_edges,
    }

    # Create shared memory for large NumPy arrays
    if not graph._numpy_arrays_built:
        raise RuntimeError("NumPy arrays not built! Call routing_graph.build_numpy_arrays() before multiprocessing")
    
    log("  Creating shared memory for NumPy arrays...")
    
    # Create shared arrays for node attributes (ZERO-COPY for all workers!)
    array_names = [
        'tile_x', 'tile_y', 'base_cost', 'length',
        'node_type', 'intent_code', 'capacity',
        'pres_cong_cost', 'hist_cong_cost', 'is_accessible'
    ]
    
    total_bytes = 0
    for name in array_names:
        arr = getattr(graph, f"{name}_arr")
        shm_manager.create_shared_array(name, arr)
        total_bytes += arr.nbytes
    
    # CRITICAL FIX: Also share CSR indices/indptr (can be 500+ MB!)
    if graph._has_csr:
        # Convert to numpy arrays if they're lists
        csr_indptr_arr = np.array(graph.csr_indptr, dtype=np.int32) if isinstance(graph.csr_indptr, list) else graph.csr_indptr
        csr_indices_arr = np.array(graph.csr_indices, dtype=np.int32) if isinstance(graph.csr_indices, list) else graph.csr_indices
        
        shm_manager.create_shared_array('csr_indptr', csr_indptr_arr)
        shm_manager.create_shared_array('csr_indices', csr_indices_arr)
        
        total_bytes += csr_indptr_arr.nbytes + csr_indices_arr.nbytes
        log(f"  CSR shared: indptr {csr_indptr_arr.nbytes/1024/1024:.1f} MB, indices {csr_indices_arr.nbytes/1024/1024:.1f} MB")
    else:
        # Must have CSR for multiprocessing (no object references)
        raise RuntimeError("CSR not built! Call routing_graph.build_csr() before multiprocessing")
    
    log(f"  Total shared memory: {total_bytes / 1024 / 1024:.1f} MB")
    
    # Store shared array metadata (only names + shapes + dtypes - small!)
    state['shared_arrays'] = shm_manager.get_shared_arrays_metadata()

    # Extract Net metadata (basic types only - no Connection object references)
    nets_meta = []
    for net in database.nets:
        nets_meta.append({
            'id': net.id,
            'name': net.name,
            'x_min': net.x_min,
            'y_min': net.y_min,
            'x_max': net.x_max,
            'y_max': net.y_max,
            'center_x': net.center_x,
            'center_y': net.center_y,
            'hpwl': net.hpwl,
            'num_connections': net.num_connections,
        })
    state['nets_meta'] = nets_meta

    # Extract Connection metadata (IDs only - no RouteNode object references)
    connections_meta = []
    for net in database.nets:
        for conn in net.connections:
            connections_meta.append({
                'conn_id': conn.id,
                'net_id': conn.net_id,
                'source_id': conn.source_node.id,  # ID not object!
                'sink_id': conn.sink_node.id,      # ID not object!
                'x_min': conn.x_min,
                'y_min': conn.y_min,
                'x_max': conn.x_max,
                'y_max': conn.y_max,
                'center_x': conn.center_x,
                'center_y': conn.center_y,
                'hpwl': conn.hpwl,
                'is_indirect': conn.is_indirect,
                # Pre-expanded bbox (if set)
                'x_min_bb': conn.x_min_bb,
                'y_min_bb': conn.y_min_bb,
                'x_max_bb': conn.x_max_bb,
                'y_max_bb': conn.y_max_bb,
            })
    state['connections_meta'] = connections_meta

    return state


def extract_partition_connections(
    partition,
    conns_by_leaf: Dict
) -> List[Tuple[int, int]]:
    """
    Extract connection identifiers from a partition.

    Returns lightweight (net_id, conn_id) tuples instead of full Connection objects.
    This avoids pickling circular references.

    Args:
        partition: Partition tree node
        conns_by_leaf: Mapping of leaf_id -> [Connection]

    Returns:
        List of (net_id, conn_id) tuples
    """
    leaf_id = id(partition)
    leaf_conns = conns_by_leaf.get(leaf_id, [])

    # Extract (net_id, conn_id) tuples
    conn_tuples = []
    for conn in leaf_conns:
        conn_tuples.append((conn.net_id, conn.id))

    return conn_tuples
