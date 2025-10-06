"""Main routing coordinator with parallel support."""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from ..db.database import Database
from ..db.route_node import Connection, RouteResult, Net
from .astar_route import AStarRouter, NodeInfo, NodeScratch
from .partition_tree import PartitionTree, PartitionBBox
from .process_worker import (
    init_worker,
    route_partition_worker,
    prepare_database_state,
    extract_partition_connections
)
from .shared_memory_manager import SharedMemoryManager
from ..utils.log import log
from ..utils.memory_monitor import log_memory_usage


class Router:
    """Main router coordinating parallel routing."""

    def __init__(self, database: Database, is_runtime_first: bool = False, use_multiprocessing: bool = True):
        self.database = database
        self.is_runtime_first = is_runtime_first
        self.num_thread = database.get_num_thread()
        self.use_multiprocessing = use_multiprocessing  # New: toggle for process vs thread

        # Routing parameters
        self.max_iter = 500
        # Dynamic present congestion factor (pres_fac) schedule like VPR: increase each iteration
        self.present_cong_factor = 0.5
        self.historical_cong_factor = 1.0
        self.present_cong_multiplier = 1.8
        self.max_present_cong_factor = 1e6

        # Statistics
        self.routed_connections = 0
        self.failed_connections = 0
        self.total_wirelength = 0
        self.connection_id_base = 0

        # Results
        self.node_routing_results: List[RouteResult] = []

        # Per-thread data (array-backed scratch for speed)
        # Only needed for thread-based routing
        num_nodes = self.database.routing_graph.num_nodes
        if not use_multiprocessing:
            self.node_info_per_thread: List[NodeScratch] = [
                NodeScratch(num_nodes) for _ in range(self.num_thread)
            ]
            # Router instances per thread
            self.routers_per_thread: List[AStarRouter] = [
                AStarRouter(
                    database,
                    self.present_cong_factor,
                    self.historical_cong_factor
                )
                for _ in range(self.num_thread)
            ]
        else:
            self.node_info_per_thread = []
            self.routers_per_thread = []
            # Create shared memory manager for zero-copy multiprocessing
            self._shm_manager: Optional[SharedMemoryManager] = SharedMemoryManager()
            # Prepare database state for worker processes
            log("Preparing database state for multiprocessing (with shared memory)...")
            self._database_state = prepare_database_state(database, self._shm_manager)
            log(f"  Database state prepared: {self._database_state['num_nodes']:,} nodes")
            log(f"  Memory model: Shared (zero-copy across {self.num_thread} processes)")

        # Per-partition cap to avoid stalls (process all by default)
        self.max_conns_per_partition = None  # route all connections in a partition per iteration

        # Overuse tracking (incremental) and iteration stats
        self._overused_nodes: set[int] = set()
        self._rerouted_this_iter: int = 0

        # Convergence tracking for early stop
        self._prev_overused_count = float('inf')
        self._stall_iterations = 0
        self._convergence_threshold = 0.01  # 1% improvement threshold

    def route(self) -> None:
        """Main routing entry point."""
        log("=" * 60)
        mode = "runtime-first" if self.is_runtime_first else "stability-first"
        parallelism = "multiprocessing" if self.use_multiprocessing else "threading"
        log(f"Starting routing ({mode} mode, {parallelism})")
        log(f"Threads/Processes: {self.num_thread}")
        log("=" * 60)
        
        # Log initial memory usage
        log_memory_usage("Initial memory: ")

        start_time = time.time()

        try:
            # Ensure connection bounding boxes are prepared (if not precomputed via CLI)
            # Only compute when missing to respect CLI-provided margins
            if not self._has_prepared_bboxes():
                self._prepare_connection_bboxes(x_margin=3, y_margin=15)

            if self.is_runtime_first:
                self._runtime_first_routing()
            else:
                self._stable_first_routing()

            elapsed = time.time() - start_time

            log("=" * 60)
            log("Routing completed:")
            log(f"  Time: {elapsed:.2f}s")
            log(f"  Routed connections: {self.routed_connections}")
            log(f"  Failed connections: {self.failed_connections}")
            log(f"  Total wirelength: {self.total_wirelength}")
            log("=" * 60)
            
            # Log final memory usage
            log_memory_usage("Final memory: ")
            
        finally:
            # Clean up shared memory
            self._cleanup_shared_memory()

    def _stable_first_routing(self) -> None:
        """Stability-first (deterministic) parallel routing."""
        log("Using stability-first parallel routing strategy")

        # Build partition tree
        device_bbox = PartitionBBox(
            x_min=self.database.layout.x_min,
            y_min=self.database.layout.y_min,
            x_max=self.database.layout.x_max,
            y_max=self.database.layout.y_max
        )
        partition_tree = PartitionTree(self.database.nets, device_bbox)
        partition_tree.build(max_nets_per_partition=100)

        log(f"Built partition tree: depth={partition_tree.max_depth}, leaves={len(partition_tree.leaves)}")

        # OPTIMIZATION: Precompute connection-to-leaf mapping using spatial hash
        # Instead of calling find_leaf_for_point() for each connection (O(n*log(d))),
        # do single-pass assignment using connection center (O(n))
        log("Assigning connections to leaf partitions...")
        self._conns_by_leaf = {}

        for net in self.database.nets:
            for conn in net.connections:
                # Find leaf containing connection center
                cx = int(conn.center_x)
                cy = int(conn.center_y)
                leaf = partition_tree.find_leaf_for_point(cx, cy)
                if leaf is None:
                    continue
                leaf_id = id(leaf)
                if leaf_id not in self._conns_by_leaf:
                    self._conns_by_leaf[leaf_id] = []
                self._conns_by_leaf[leaf_id].append(conn)

        # Map leaf object id to leaf object for lookup
        self._leaf_by_id = {id(leaf): leaf for leaf in partition_tree.leaves}
        log(f"Assigned {sum(len(v) for v in self._conns_by_leaf.values())} connections to {len(self._conns_by_leaf)} partitions")

        # Use leaf partitions to maximize parallelism and avoid re-routing same nets across levels
        leaf_partitions = partition_tree.leaves
        log(f"Routing with {len(leaf_partitions)} leaf partitions in parallel")

        # Route in iterations with congestion resolution
        for iteration in range(self.max_iter):
            iter_start = time.time()
            log(f"Iteration {iteration + 1}/{self.max_iter}")
            self._rerouted_this_iter = 0

            # Route all leaf partitions in parallel
            self._route_partitions_parallel(leaf_partitions, iteration)

            # Check congestion (note: incremental tracking via set)
            num_overused = self._count_overused_nodes()
            iter_time = time.time() - iter_start
            log(f"  Overused nodes: {num_overused}")
            log(f"  Rerouted: {self._rerouted_this_iter} connections")
            log(f"  Iteration time: {iter_time:.2f}s")
            
            # Log memory usage every 10 iterations
            if (iteration + 1) % 10 == 0:
                log_memory_usage("  Memory: ")

            if num_overused == 0:
                log("Routing converged - no overused nodes")
                break

            # OPTIMIZATION: Early stop if no progress
            if self._rerouted_this_iter == 0:
                log("  No connections re-routed this iteration; early stop")
                break

            # OPTIMIZATION: Convergence detection - stop if improvement is too slow
            improvement_rate = (self._prev_overused_count - num_overused) / max(1, self._prev_overused_count)
            if improvement_rate < self._convergence_threshold and iteration > 10:
                self._stall_iterations += 1
                log(f"  Convergence stalled (improvement: {improvement_rate:.1%})")
                if self._stall_iterations >= 3:
                    log("  Convergence stalled for 3 iterations - early stop")
                    break
            else:
                self._stall_iterations = 0

            self._prev_overused_count = num_overused

            # Update congestion schedule (with adaptive adjustment)
            self._update_congestion_costs_adaptive(improvement_rate)

        self._save_all_routing_solutions()

    def _runtime_first_routing(self) -> None:
        """Runtime-first (non-deterministic) parallel routing."""
        log("Using runtime-first parallel routing strategy")

        # Route all connections in parallel without strict ordering
        # This allows maximum parallelism but may be non-deterministic

        for iteration in range(self.max_iter):
            iter_start = time.time()
            log(f"Iteration {iteration + 1}/{self.max_iter}")
            self._rerouted_this_iter = 0

            # Sort connections by criticality
            sorted_conns = self._sort_connections()

            # Route in parallel batches
            batch_size = max(1, len(sorted_conns) // self.num_thread)
            batches = [sorted_conns[i:i + batch_size]
                      for i in range(0, len(sorted_conns), batch_size)]

            with ThreadPoolExecutor(max_workers=self.num_thread) as executor:
                futures = []
                for i, batch in enumerate(batches):
                    tid = i % self.num_thread
                    futures.append(executor.submit(self._route_batch, batch, tid))
                for future in as_completed(futures):
                    future.result()

            # Check convergence (note: incremental tracking)
            num_overused = self._count_overused_nodes()
            iter_time = time.time() - iter_start
            log(f"  Overused nodes: {num_overused}")
            log(f"  Rerouted: {self._rerouted_this_iter} connections")
            log(f"  Iteration time: {iter_time:.2f}s")

            if num_overused == 0:
                log("Routing converged")
                break

            self._update_congestion_costs()

        self._save_all_routing_solutions()

    def _route_partitions_parallel(
        self,
        partitions: List,
        iteration: int
    ) -> None:
        """Route partitions in parallel with exclusive per-worker state.

        Uses ProcessPoolExecutor for true parallelism (bypassing GIL) when
        use_multiprocessing=True, otherwise falls back to ThreadPoolExecutor.
        """
        if self.use_multiprocessing:
            self._route_partitions_multiprocess(partitions, iteration)
        else:
            self._route_partitions_threaded(partitions, iteration)

    def _route_partitions_threaded(
        self,
        partitions: List,
        iteration: int
    ) -> None:
        """Thread-based parallel routing (original implementation)."""
        from queue import Queue, Empty

        q: Queue = Queue()
        for p in partitions:
            q.put(p)

        def worker(tid: int):
            while True:
                try:
                    part = q.get_nowait()
                except Empty:
                    break
                try:
                    self._route_partition(part, iteration, tid)
                finally:
                    q.task_done()

        with ThreadPoolExecutor(max_workers=self.num_thread) as executor:
            futures = [executor.submit(worker, tid) for tid in range(self.num_thread)]
            for future in as_completed(futures):
                future.result()

    def _route_partitions_multiprocess(
        self,
        partitions: List,
        iteration: int
    ) -> None:
        """
        Process-based parallel routing (GIL-free).

        Each partition is routed in a separate process. Results are collected
        and merged back into the main database.
        """
        # Extract lightweight connection identifiers from partitions
        partition_tasks = []
        for partition in partitions:
            conn_tuples = extract_partition_connections(partition, self._conns_by_leaf)
            if conn_tuples:
                partition_tasks.append(conn_tuples)

        if not partition_tasks:
            return

        # Create worker initializer function with current congestion factors
        from functools import partial
        worker_init = partial(
            init_worker,
            self._database_state,
            present_cong_factor=self.present_cong_factor,
            historical_cong_factor=self.historical_cong_factor
        )

        # Route partitions in parallel using processes
        results = []
        with ProcessPoolExecutor(
            max_workers=self.num_thread,
            initializer=worker_init,
            initargs=(0,)  # worker_id will be set by executor
        ) as executor:
            futures = []
            for worker_id, task in enumerate(partition_tasks):
                # Re-initialize worker with correct ID
                future = executor.submit(
                    route_partition_worker,
                    task,
                    iteration,
                    self.connection_id_base,
                    needs_ripup=(iteration > 0)
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Merge results back into main database
        self._merge_worker_results(results)

    def _merge_worker_results(self, results: List[Dict]) -> None:
        """
        Merge routing results from worker processes back into main database.

        Args:
            results: List of result dictionaries from workers
        """
        for result in results:
            worker_id = result['worker_id']
            routed_count = result['routed_count']
            failed_count = result['failed_count']
            rerouted_count = result['rerouted_count']

            # Update statistics
            self.routed_connections += routed_count
            self.failed_connections += failed_count
            self._rerouted_this_iter += rerouted_count

            # Restore routed paths to Connection objects
            for net_id, conn_id, node_ids in result['routed_paths']:
                net = self.database.nets[net_id]
                connection = net.connections[conn_id]

                # Reconstruct route_nodes from node IDs
                connection.route_nodes = [
                    self.database.routing_graph.node_map[nid] for nid in node_ids
                ]
                connection.is_routed = True

                # Update overuse tracking
                for rnode in connection.route_nodes:
                    self._update_overuse_for_node(rnode)

            # Merge node usage updates
            for node_id, net_usage in result['node_usage_updates'].items():
                node = self.database.routing_graph.node_map.get(node_id)
                if node is None:
                    continue

                # Update node's user connection counts
                for net_id, count in net_usage.items():
                    if net_id not in node.users_connection_counts:
                        node.users_connection_counts[net_id] = 0
                    node.users_connection_counts[net_id] += count

                    # Update net's user tracking
                    net = self.database.nets[net_id]
                    if node.id not in net.users_connection_counts:
                        net.users_connection_counts[node.id] = 0
                    net.users_connection_counts[node.id] += count

                # Update present congestion cost
                node.update_present_congestion_cost(self.present_cong_factor)


    def _route_partition(self, partition, iteration: int, tid: int) -> None:
        """Route all connections in a partition."""
        router = self.routers_per_thread[tid]
        node_infos = self.node_info_per_thread[tid]

        # Route only connections assigned to this leaf
        leaf_conns = self._conns_by_leaf.get(id(partition), [])
        if not leaf_conns:
            return

        # Route connections in this partition
        routed_in_partition = 0
        for conn in leaf_conns:
            # Decide if this connection needs ripup/reroute
            needs_route = router.should_route(conn)
            if not needs_route:
                continue

            # Ripup on subsequent iterations
            if iteration > 0:
                self._ripup_connection(conn)

            # Generate unique connection ID
            conn_unique_id = conn.id + self.connection_id_base

            # Route connection
            success = router.route_one_connection(
                conn, node_infos, conn_unique_id, tid, sync=False
            )

            if success:
                self.routed_connections += 1
                # Update overuse tracking for nodes touched
                for rn in conn.route_nodes:
                    self._update_overuse_for_node(rn)
                self._rerouted_this_iter += 1
            else:
                self.failed_connections += 1
            routed_in_partition += 1

            # Periodic progress logging per partition (reduced frequency for performance)
            if routed_in_partition % 2000 == 0:
                log(f"[T{tid}] Routed {routed_in_partition} conns in partition (iter {iteration+1})")

            # Cap per-partition work per iteration to keep system responsive
            if self.max_conns_per_partition is not None and routed_in_partition >= self.max_conns_per_partition:
                return

    def _route_batch(self, connections: List[Connection], tid: int) -> None:
        """Route a batch of connections."""
        router = self.routers_per_thread[tid]
        node_infos = self.node_info_per_thread[tid]

        for conn in connections:
            if not router.should_route(conn):
                continue

            self._ripup_connection(conn)
            conn_unique_id = conn.id + self.connection_id_base
            ok = router.route_one_connection(
                conn, node_infos, conn_unique_id, tid, sync=False
            )
            if ok:
                for rn in conn.route_nodes:
                    self._update_overuse_for_node(rn)
                self._rerouted_this_iter += 1

    def _ripup_connection(self, connection: Connection) -> None:
        """Rip up existing routing for a connection."""
        net = self.database.nets[connection.net_id]

        for node in connection.route_nodes:
            node.decrement_user(connection.net_id)
            net.decrement_user(node)
            self._update_overuse_for_node(node)

        connection.route_nodes.clear()
        connection.is_routed = False

    def _sort_connections(self) -> List[Connection]:
        """Sort connections by routing criticality."""
        conns = self.database.indirect_connections[:]
        # Sort by bounding box size (smaller first)
        conns.sort(key=lambda c: c.width * c.height)
        return conns

    def _count_overused_nodes(self) -> int:
        """Count number of overused routing nodes (using incremental tracking)."""
        # Use incrementally-maintained set (avoids full graph scan)
        return len(self._overused_nodes)

    def _update_congestion_costs_adaptive(self, improvement_rate: float) -> None:
        """
        Escalate pres_fac adaptively based on convergence rate.

        OPTIMIZATION: Adaptive scheduling
        - Fast convergence: reduce multiplier (avoid over-penalizing)
        - Slow convergence: increase multiplier (accelerate resolution)
        """
        # Adaptive multiplier adjustment
        if improvement_rate > 0.15:  # Fast convergence (>15% improvement)
            # Reduce aggressiveness
            actual_multiplier = self.present_cong_multiplier * 0.9
        elif improvement_rate < 0.03:  # Very slow (<3% improvement)
            # Increase aggressiveness
            actual_multiplier = self.present_cong_multiplier * 1.1
        else:
            actual_multiplier = self.present_cong_multiplier

        # Increase global present congestion factor for next iteration
        self.present_cong_factor = min(
            self.present_cong_factor * actual_multiplier,
            self.max_present_cong_factor,
        )

        # Propagate updated pres_fac to thread-local A* routers (if using threading)
        if not self.use_multiprocessing:
            for r in self.routers_per_thread:
                r.present_cong_factor = self.present_cong_factor

        # OPTIMIZATION: Use Numba batch update for congestion costs
        self._update_congestion_costs_batch_numba()

    def _update_congestion_costs_batch_numba(self) -> None:
        """
        Batch update congestion costs using Numba (OPTIMIZATION).

        Only updates nodes in the overuse set (dirty nodes), avoiding full graph scan.
        """
        if len(self._overused_nodes) == 0:
            return

        pres = self.present_cong_factor
        hist_fac = self.historical_cong_factor

        # Check if Numba batch update is available
        graph = self.database.routing_graph
        if graph._numpy_arrays_built:
            try:
                from .numba_kernels import update_congestion_costs_batch

                # Collect node IDs to update
                node_ids_arr = np.array(list(self._overused_nodes), dtype=np.int32)

                # Build occupancy array (need to sync from RouteNode objects)
                occupancy_arr = np.zeros(graph.num_nodes, dtype=np.int32)
                for node in graph.nodes:
                    occupancy_arr[node.id] = node.occupancy

                # Batch update using Numba
                update_congestion_costs_batch(
                    node_ids_arr,
                    len(node_ids_arr),
                    occupancy_arr,
                    graph.capacity_arr,
                    pres,
                    hist_fac,
                    graph.pres_cong_cost_arr,  # Modified in-place
                    graph.hist_cong_cost_arr   # Modified in-place
                )

                # Sync back to RouteNode objects
                for node_id in node_ids_arr:
                    node = graph.node_map.get(node_id)
                    if node:
                        node.present_congestion_cost = graph.pres_cong_cost_arr[node_id]
                        node.historical_congestion_cost = graph.hist_cong_cost_arr[node_id]

                return  # Success - Numba path taken
            except Exception as e:
                log(f"Warning: Numba batch update failed: {e}, falling back to Python")

        # Fallback: Python loop (original implementation)
        for node_id in self._overused_nodes:
            node = self.database.routing_graph.node_map.get(node_id)
            if node is None:
                continue
            overuse = node.occupancy - node.capacity
            if overuse <= 0:
                node.present_congestion_cost = 1.0 + pres
            else:
                node.present_congestion_cost = 1.0 + (overuse + 1) * pres
                node.historical_congestion_cost += overuse * hist_fac

    def _update_congestion_costs(self) -> None:
        """Legacy congestion cost update (non-adaptive)."""
        # Increase global present congestion factor for next iteration
        self.present_cong_factor = min(
            self.present_cong_factor * self.present_cong_multiplier,
            self.max_present_cong_factor,
        )

        # Propagate updated pres_fac to thread-local A* routers
        for r in self.routers_per_thread:
            r.present_cong_factor = self.present_cong_factor

        # OPTIMIZATION: Only update nodes that have been touched (in overuse set or near it)
        # Instead of full graph scan, maintain a "dirty" set of nodes that need cost update
        pres = self.present_cong_factor
        hist_fac = self.historical_cong_factor

        # Track which nodes to update: all currently/previously overused + their neighbors
        nodes_to_update = set(self._overused_nodes)

        # Update only nodes that are or were overused
        for node_id in nodes_to_update:
            node = self.database.routing_graph.node_map.get(node_id)
            if node is None:
                continue
            overuse = node.occupancy - node.capacity
            if overuse <= 0:
                node.present_congestion_cost = 1.0 + pres
            else:
                node.present_congestion_cost = 1.0 + (overuse + 1) * pres
                node.historical_congestion_cost += overuse * hist_fac

    def _save_all_routing_solutions(self) -> None:
        """Save all routing solutions."""
        for net in self.database.nets:
            for conn in net.connections:
                if conn.is_routed:
                    result = RouteResult(
                        connection_id=conn.id,
                        net_id=conn.net_id,
                        nodes=conn.route_nodes,
                        wirelength=len(conn.route_nodes),
                        success=True
                    )
                    self.node_routing_results.append(result)
                    self.total_wirelength += len(conn.route_nodes)

    def _update_overuse_for_node(self, node) -> None:
        """Update the tracked overuse set based on node occupancy vs capacity."""
        if node.occupancy > node.capacity:
            self._overused_nodes.add(node.id)
        else:
            # Remove when not overused
            if node.id in self._overused_nodes:
                self._overused_nodes.discard(node.id)

    def _prepare_connection_bboxes(self, x_margin: int = 3, y_margin: int = 15) -> None:
        """Compute and store per-connection routing bbox with fixed margins (like C++)."""
        for net in self.database.nets:
            for conn in net.connections:
                conn.x_min_bb = conn.x_min - x_margin
                conn.x_max_bb = conn.x_max + x_margin
                conn.y_min_bb = conn.y_min - y_margin
                conn.y_max_bb = conn.y_max + y_margin

    def _has_prepared_bboxes(self) -> bool:
        """Check if bboxes were prepared for at least one connection."""
        for net in self.database.nets:
            for conn in net.connections:
                return conn.x_min_bb is not None
        return False
    
    def _cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources."""
        if self.use_multiprocessing and hasattr(self, '_shm_manager') and self._shm_manager is not None:
            log("Cleaning up shared memory...")
            self._shm_manager.cleanup()
            self._shm_manager = None
            log("  Shared memory cleaned up successfully")
