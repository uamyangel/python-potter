"""FPGA device representation and parsing."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional
from .route_node_graph import RouteNodeGraph
from .route_node import RouteNode
from ..utils.log import log


class Device:
    """Represents FPGA device with tiles and routing resources."""

    def __init__(self, routing_graph: RouteNodeGraph):
        self.routing_graph = routing_graph
        self.name: str = ""
        self.tiles: List[Dict] = []
        self.tile_types: List[Dict] = []
        self.tile_map: Dict[str, int] = {}
        self.num_tiles: int = 0

        # Extended attributes for full device support
        self.strings: List[str] = []
        self.sites: List[Dict] = []
        self.site_types: List[Dict] = []
        self.tile_name_to_idx: Dict[str, int] = {}
        self.site_name_to_idx: Dict[str, int] = {}

        # Site pin mapping structures
        self.site_type_pin_name_to_idx: List[Dict[str, int]] = []
        self.tile_type_site_pin_to_wire_idx: List[List[List[int]]] = []
        self.tile_wire_to_node: List[List[int]] = []
        # Tile-type wire name indices and PIPs
        self.tile_type_wire_idx_to_str: List[List[int]] = []
        self.tile_type_pips: List[List[tuple]] = []

    def read(self, device_file: str, cache_mode: str = "off") -> None:
        """Read device file and populate routing graph."""
        log(f"Reading device file: {device_file}")

        device_path = Path(device_file)
        if not device_path.exists():
            raise FileNotFoundError(f"Device file not found: {device_file}")

        # Check for cached parsed data
        dump_dir = Path("./dump")
        dump_dir.mkdir(exist_ok=True)
        cached_file = dump_dir / f"{device_path.stem}.pkl"
        light_npz = dump_dir / f"{device_path.stem}.light.npz"
        light_meta = dump_dir / f"{device_path.stem}.light.pkl"

        # Prefer light cache when requested
        if cache_mode == "light" and light_npz.exists() and light_meta.exists():
            try:
                log(f"Loading light device cache from {light_npz.name}")
                self._load_light_cache(light_npz, light_meta)
                log(f"Device loaded from light cache: {self.num_tiles} tiles, {self.routing_graph.num_nodes} nodes")
                return
            except Exception as e:
                log(f"Light cache load failed: {e}. Falling back to parse...")

        if cached_file.exists():
            try:
                log(f"Loading cached device data from {cached_file}")
                self._load_cached(cached_file)
            except Exception as e:
                log(f"Cache load failed: {e}. Rebuilding cache...")
                try:
                    cached_file.unlink(missing_ok=True)
                except Exception:
                    pass
                log(f"Parsing device file (rebuild)...")
                self._parse_device_file(device_path)
                log(f"Caching parsed data to {cached_file}")
                self._save_cached(cached_file)
        else:
            log(f"Parsing device file (first run)...")
            self._parse_device_file(device_path)
            log(f"Caching parsed data to {cached_file}")
            self._save_cached(cached_file)

        # After parse, optionally create light cache for future runs
        if cache_mode == "light":
            try:
                self._save_light_cache(light_npz, light_meta)
                log(f"Saved light cache: {light_npz.name}")
            except Exception as e:
                log(f"Saving light cache failed: {e}")

        log(f"Device loaded: {self.num_tiles} tiles, {self.routing_graph.num_nodes} nodes")

    def get_site_pin_node(self, site_name: str, pin_name: str) -> Optional[int]:
        """
        Get node index for a site pin.

        Args:
            site_name: Name of the site
            pin_name: Name of the pin

        Returns:
            Node index, or None if not found
        """
        # Find site by name
        if site_name not in self.site_name_to_idx:
            return None

        site_idx = self.site_name_to_idx[site_name]
        if site_idx >= len(self.sites):
            return None

        site = self.sites[site_idx]
        tile_idx = site['tile_idx']
        tile_type_idx = site['tile_type_idx']
        in_tile_site_idx = site['in_tile_site_idx']
        site_type_idx = site['site_type_idx']

        # Find pin index in site type
        if site_type_idx < 0 or site_type_idx >= len(self.site_type_pin_name_to_idx):
            return None

        pin_name_to_idx = self.site_type_pin_name_to_idx[site_type_idx]
        if pin_name not in pin_name_to_idx:
            return None

        pin_idx = pin_name_to_idx[pin_name]

        # Get wire index from site pin
        if tile_type_idx >= len(self.tile_type_site_pin_to_wire_idx):
            return None

        site_pin_to_wire = self.tile_type_site_pin_to_wire_idx[tile_type_idx]
        if in_tile_site_idx >= len(site_pin_to_wire):
            return None

        pin_to_wire = site_pin_to_wire[in_tile_site_idx]
        if pin_idx >= len(pin_to_wire):
            return None

        tile_type_wire_idx = pin_to_wire[pin_idx]
        if tile_type_wire_idx < 0:
            return None

        # Get node from tile wire
        if tile_idx >= len(self.tile_wire_to_node):
            return None

        tile_wires_to_nodes = self.tile_wire_to_node[tile_idx]
        if tile_type_wire_idx >= len(tile_wires_to_nodes):
            return None

        node_idx = tile_wires_to_nodes[tile_type_wire_idx]
        if node_idx < 0:
            return None

        return node_idx

    def _parse_device_file(self, device_path: Path) -> None:
        """Parse device file using DeviceParser."""
        try:
            from .device_parser import DeviceParser

            parser = DeviceParser()
            parsed_device = parser.parse(str(device_path))

            # Copy data from parsed device
            self.name = parsed_device.name
            self.tiles = parsed_device.tiles
            self.tile_types = parsed_device.tile_types
            self.num_tiles = parsed_device.num_tiles
            self.strings = parsed_device.strings
            self.tile_name_to_idx = parsed_device.tile_name_to_idx
            # Extended site/pin and mapping structures for pin-to-wire resolution
            self.sites = parsed_device.sites
            self.site_types = parsed_device.site_types
            self.site_name_to_idx = parsed_device.site_name_to_idx
            self.site_type_pin_name_to_idx = parsed_device.site_type_pin_name_to_idx
            self.tile_type_site_pin_to_wire_idx = parsed_device.tile_type_site_pin_to_wire_idx
            # Optional: direct mapping tile -> tile_type-local wire -> node id
            self.tile_wire_to_node = parsed_device.tile_wire_to_node
            # Tile-type mappings and PIPs
            self.tile_type_wire_idx_to_str = getattr(parsed_device, 'tile_type_wire_idx_to_str', [])
            self.tile_type_pips = getattr(parsed_device, 'tile_type_pips', [])

            # Copy routing graph
            self.routing_graph.nodes = parsed_device.routing_graph.nodes
            self.routing_graph.node_map = {n.id: n for n in self.routing_graph.nodes}
            self.routing_graph.num_nodes = len(self.routing_graph.nodes)
            self.routing_graph.num_edges = parsed_device.routing_graph.num_edges

        except ImportError as e:
            log(f"DeviceParser not available: {e}")
            log("Device parsing not yet implemented - requires FPGA interchange schema")
        except Exception as e:
            log(f"Error parsing device: {e}")
            log("Device parsing not yet implemented - requires FPGA interchange schema")

    def _load_cached(self, cache_path: Path) -> None:
        """Load cached device data."""
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
            self.name = cached.get('name', '')
            self.tiles = cached.get('tiles', [])
            self.tile_map = cached.get('tile_map', {})
            self.num_tiles = cached.get('num_tiles', 0)
            # Nodes are not cached for large devices; force parse if missing
            cached_nodes = cached.get('nodes', [])
            if cached_nodes:
                self.routing_graph.nodes = cached_nodes
                self.routing_graph.node_map = {n.id: n for n in self.routing_graph.nodes}
                self.routing_graph.num_nodes = len(self.routing_graph.nodes)
                self.routing_graph.num_edges = cached.get('num_edges', 0)
            else:
                raise ValueError("Cache does not contain nodes; requires re-parse")
            # Optional cached mappings
            self.sites = cached.get('sites', [])
            self.site_types = cached.get('site_types', [])
            self.site_name_to_idx = cached.get('site_name_to_idx', {})
            self.site_type_pin_name_to_idx = cached.get('site_type_pin_name_to_idx', [])
            self.tile_type_site_pin_to_wire_idx = cached.get('tile_type_site_pin_to_wire_idx', [])
            self.tile_wire_to_node = cached.get('tile_wire_to_node', [])
            self.tile_type_wire_idx_to_str = cached.get('tile_type_wire_idx_to_str', [])
            self.tile_type_pips = cached.get('tile_type_pips', [])
            self.tile_type_wire_idx_to_str = cached.get('tile_type_wire_idx_to_str', [])
            self.tile_type_pips = cached.get('tile_type_pips', [])

    def _save_cached(self, cache_path: Path) -> None:
        """Save parsed device data to cache."""
        # Guard against excessive size / recursion in pickling (due to graph edges)
        MAX_CACHE_NODES = 5_000_000
        if self.routing_graph.num_nodes > MAX_CACHE_NODES:
            log(f"Device too large to cache efficiently (nodes={self.routing_graph.num_nodes:,}); skipping cache")
            return
        try:
            cached = {
                'name': self.name,
                'tiles': self.tiles,
                'tile_map': self.tile_map,
                'num_tiles': self.num_tiles,
                # Cache nodes without parent/children links to reduce recursion; store only ids
                # For simplicity, skip nodes entirely for now; caller will rebuild from device file if needed
                'nodes': [],
                'num_edges': 0,
                'sites': getattr(self, 'sites', []),
                'site_types': getattr(self, 'site_types', []),
                'site_name_to_idx': getattr(self, 'site_name_to_idx', {}),
                'site_type_pin_name_to_idx': getattr(self, 'site_type_pin_name_to_idx', []),
                'tile_type_site_pin_to_wire_idx': getattr(self, 'tile_type_site_pin_to_wire_idx', []),
                'tile_wire_to_node': getattr(self, 'tile_wire_to_node', []),
                'tile_type_wire_idx_to_str': getattr(self, 'tile_type_wire_idx_to_str', []),
                'tile_type_pips': getattr(self, 'tile_type_pips', []),
            }
            # Atomic write to avoid partial/corrupted cache files
            tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
            with open(tmp_path, 'wb') as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
            tmp_path.replace(cache_path)
        except (RecursionError, MemoryError) as e:
            log(f"Skipping cache due to serialization limits: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def check_memory_peak(self, device_id: int = -1) -> None:
        """Check memory usage."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        log(f"Memory usage: {mem_mb:.2f} MB")

    # -------- Light cache (numpy + pickle) ---------
    def _save_light_cache(self, npz_path: Path, meta_path: Path) -> None:
        """Save lightweight cache with arrays + metadata lists."""
        import numpy as np
        # Extract node attributes into arrays
        nodes = self.routing_graph.nodes
        n = len(nodes)
        tile_id = np.empty(n, dtype=np.int32)
        tile_x = np.empty(n, dtype=np.int32)
        tile_y = np.empty(n, dtype=np.int32)
        wire_id = np.empty(n, dtype=np.int32)
        node_type = np.empty(n, dtype=np.int16)
        intent = np.empty(n, dtype=np.int16)
        base_cost = np.empty(n, dtype=np.float32)
        length = np.empty(n, dtype=np.int16)
        for i, rn in enumerate(nodes):
            tile_id[i] = rn.tile_id
            tile_x[i] = rn.tile_x
            tile_y[i] = rn.tile_y
            wire_id[i] = rn.wire_id
            node_type[i] = int(rn.node_type)
            intent[i] = int(rn.intent_code)
            base_cost[i] = float(rn.base_cost)
            length[i] = int(rn.length)

        # Save arrays
        np.savez_compressed(npz_path, tile_id=tile_id, tile_x=tile_x, tile_y=tile_y,
                            wire_id=wire_id, node_type=node_type, intent=intent,
                            base_cost=base_cost, length=length)

        # Save metadata lists via pickle (no RouteNode objects)
        meta = {
            'name': self.name,
            'tiles': [{
                'name_idx': t.get('name_idx', 0),
                'type': t.get('type', 0),
                'row': t.get('row', 0),
                'col': t.get('col', 0),
            } for t in self.tiles],
            'tile_types': [{
                'name': tt.get('name', ''),
                'num_wires': tt.get('num_wires', 0),
            } for tt in self.tile_types],
            'sites': self.sites,
            'site_types': self.site_types,
            'site_name_to_idx': self.site_name_to_idx,
            'site_type_pin_name_to_idx': self.site_type_pin_name_to_idx,
            'tile_type_site_pin_to_wire_idx': self.tile_type_site_pin_to_wire_idx,
            'tile_wire_to_node': self.tile_wire_to_node,
            'tile_type_wire_idx_to_str': getattr(self, 'tile_type_wire_idx_to_str', []),
            'tile_type_pips': getattr(self, 'tile_type_pips', []),
        }
        import pickle
        with open(meta_path.with_suffix(meta_path.suffix + '.tmp'), 'wb') as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
        meta_path.with_suffix(meta_path.suffix + '.tmp').replace(meta_path)

    def _load_light_cache(self, npz_path: Path, meta_path: Path) -> None:
        """Load lightweight cache and rebuild device+graph quickly."""
        import numpy as np
        import pickle
        data = np.load(npz_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        # Restore basic metadata
        self.name = meta.get('name', '')
        self.tiles = meta.get('tiles', [])
        self.tile_types = meta.get('tile_types', [])
        self.num_tiles = len(self.tiles)
        self.sites = meta.get('sites', [])
        self.site_types = meta.get('site_types', [])
        self.site_name_to_idx = meta.get('site_name_to_idx', {})
        self.site_type_pin_name_to_idx = meta.get('site_type_pin_name_to_idx', [])
        self.tile_type_site_pin_to_wire_idx = meta.get('tile_type_site_pin_to_wire_idx', [])
        self.tile_wire_to_node = meta.get('tile_wire_to_node', [])
        self.tile_type_wire_idx_to_str = meta.get('tile_type_wire_idx_to_str', [])
        self.tile_type_pips = meta.get('tile_type_pips', [])

        # Recreate RouteNode objects from arrays
        tile_id = data['tile_id']
        tile_x = data['tile_x']
        tile_y = data['tile_y']
        wire_id = data['wire_id']
        node_type = data['node_type']
        intent = data['intent']
        base_cost = data['base_cost']
        length = data['length']
        n = tile_id.shape[0]

        from .route_node import RouteNode
        nodes: List[RouteNode] = [None] * n
        for i in range(n):
            rn = RouteNode(
                id=i,
                tile_id=int(tile_id[i]),
                tile_x=int(tile_x[i]),
                tile_y=int(tile_y[i]),
                wire_id=int(wire_id[i]),
                node_type=int(node_type[i]),
                intent_code=int(intent[i]),
                base_cost=float(base_cost[i]),
                length=int(length[i]),
            )
            nodes[i] = rn

        # Populate graph
        self.routing_graph.nodes = nodes
        self.routing_graph.node_map = {i: nodes[i] for i in range(n)}
        self.routing_graph.num_nodes = n
        self.routing_graph.num_edges = 0

        # Rebuild edges from cached PIPs and tile_wire_to_node
        self._build_edges_from_cache()
        # Build CSR adjacency for faster neighbor iteration
        try:
            self.routing_graph.build_csr()
        except Exception:
            pass

    def _build_edges_from_cache(self) -> None:
        """Build parent/child edges using cached tile_wire_to_node and tile_type_pips."""
        edge_count = 0
        for tile_idx, tile in enumerate(self.tiles):
            tile_type_idx = tile['type']
            pips = self.tile_type_pips[tile_type_idx] if tile_type_idx < len(self.tile_type_pips) else []
            wires_to_nodes = self.tile_wire_to_node[tile_idx] if tile_idx < len(self.tile_wire_to_node) else []
            for (w0, w1, directional) in pips:
                n0 = wires_to_nodes[w0] if w0 < len(wires_to_nodes) else -1
                n1 = wires_to_nodes[w1] if w1 < len(wires_to_nodes) else -1
                if n0 >= 0 and n1 >= 0:
                    self.routing_graph.add_edge(self.routing_graph.nodes[n0], self.routing_graph.nodes[n1])
                    edge_count += 1
                    if not directional:
                        self.routing_graph.add_edge(self.routing_graph.nodes[n1], self.routing_graph.nodes[n0])
                        edge_count += 1
        self.routing_graph.num_edges = edge_count
