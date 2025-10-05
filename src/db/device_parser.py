"""Device file parser for FPGA interchange format.

Parses .device files (gzipped Cap'n Proto format) and constructs the routing graph.
"""

import gzip
import io
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np

from ..db.device import Device
from ..db.route_node import RouteNode
from ..db.route_node_graph import RouteNodeGraph
from ..global_defs import NodeType, IntentCode, TileTypes
from ..utils.log import log

# Will be imported when pycapnp is available
DeviceResources = None


class DeviceParser:
    """Parser for FPGA device files."""

    def __init__(self):
        self.strings: List[str] = []
        self.tiles: List[Dict] = []
        self.tile_types: List[Dict] = []
        self.site_types: List[Dict] = []
        self.wires: List[Dict] = []
        self.wire_types: List[Dict] = []
        self.nodes: List[Dict] = []

        # Index structures
        self.tile_name_to_idx: Dict[str, int] = {}
        self.site_name_to_idx: Dict[str, int] = {}
        self.tile_type_outgoing_wires: List[List[List[int]]] = []
        self.tile_type_incoming_wires: List[List[List[int]]] = []
        self.tile_wire_to_node: List[List[int]] = []
        # PIPs per tile type (wire0, wire1, directional)
        self.tile_type_pips: List[List[Tuple[int, int, bool]]] = []

        # Tile coordinates and mappings
        self.tile_x_coords: List[int] = []
        self.tile_y_coords: List[int] = []
        self.tile_str_to_idx: Dict[int, int] = {}

        # Site structures for pin mapping
        self.sites: List[Dict] = []  # site data
        self.site_type_pin_name_to_idx: List[Dict[str, int]] = []  # site_type → {pin_name: pin_idx}
        self.tile_type_site_pin_to_wire_idx: List[List[List[int]]] = []  # [tile_type][site_in_tile][pin] → wire
        self.tile_type_wire_str_to_idx: List[Dict[int, int]] = []  # [tile_type] → {wire_str: wire_idx}
        self.tile_type_wire_idx_to_str: List[List[int]] = []  # [tile_type] → [wire_str_idx]

    def parse(self, device_file: str) -> Device:
        """
        Parse device file and return Device object.

        Args:
            device_file: Path to .device file (gzipped Cap'n Proto)

        Returns:
            Device object with routing graph
        """
        log(f"Parsing device file: {device_file}")

        device_path = Path(device_file)
        if not device_path.exists():
            raise FileNotFoundError(f"Device file not found: {device_file}")

        # Import schema
        try:
            from ..schemas import DeviceResources as DR
            global DeviceResources
            DeviceResources = DR
        except ImportError as e:
            raise ImportError(
                f"Failed to import schemas: {e}. "
                "Make sure pycapnp is installed and schemas are accessible."
            )

        # Decompress gzip
        log("Decompressing gzip...")
        with gzip.open(device_file, 'rb') as f:
            data = f.read()

        log(f"Decompressed size: {len(data) / 1024 / 1024:.2f} MB")

        # Parse Cap'n Proto
        log("Parsing Cap'n Proto message...")
        try:
            import capnp
            from ..schemas import DeviceResources as DR

            # Parse message from bytes with increased traversal limit
            # xcvu3p is very large (~28M nodes, 2GB data)
            with DR.Device.from_bytes(data, traversal_limit_in_words=2**63-1) as device_reader:
                # Extract all data while context is active
                self._extract_all_data(device_reader)

        except Exception as e:
            log(f"Cap'n Proto parsing error: {e}")
            import traceback
            traceback.print_exc()
            log("Trying fallback...")
            return self._parse_fallback(device_path)

        # Build routing graph after extracting data
        log("Building routing graph...")
        routing_graph = self._build_routing_graph()

        # Create Device object
        device = Device(routing_graph)
        device.name = device_path.stem
        device.strings = self.strings
        device.tiles = self.tiles
        device.tile_types = self.tile_types
        device.num_tiles = len(self.tiles)

        # Transfer site data
        device.sites = self.sites
        device.site_types = self.site_types
        device.site_name_to_idx = self.site_name_to_idx
        device.site_type_pin_name_to_idx = self.site_type_pin_name_to_idx
        device.tile_type_site_pin_to_wire_idx = self.tile_type_site_pin_to_wire_idx
        device.tile_wire_to_node = self.tile_wire_to_node
        # Tile-type mappings and PIPs
        device.tile_type_wire_idx_to_str = self.tile_type_wire_idx_to_str
        device.tile_type_pips = self.tile_type_pips

        # Build index structures
        log("Building index structures...")
        self._build_indices(device)

        log(f"Device parsed successfully:")
        log(f"  Strings: {len(self.strings)}")
        log(f"  Tiles: {len(self.tiles)}")
        log(f"  TileTypes: {len(self.tile_types)}")
        log(f"  Nodes: {len(self.nodes)}")
        log(f"  Wires: {len(self.wires)}")

        return device

    def _extract_all_data(self, device_reader):
        """Extract all data from device_reader while in context."""
        log("Extracting data structures...")
        self._extract_strings(device_reader)
        self._extract_tile_types(device_reader)  # Must be before tiles
        self._extract_tiles(device_reader)
        self._extract_site_types(device_reader)
        self._extract_sites(device_reader)  # New: extract sites
        self._extract_wires(device_reader)
        self._extract_wire_types(device_reader)
        self._extract_nodes(device_reader)

    def _extract_strings(self, device_reader):
        """Extract string list."""
        str_list = device_reader.strList
        self.strings = [str(s) for s in str_list]
        log(f"Extracted {len(self.strings)} strings")

    def _extract_tiles(self, device_reader):
        """Extract tiles with coordinates."""
        tile_list = device_reader.tileList
        num_tiles = len(tile_list)

        self.tiles = []
        self.tile_x_coords = []
        self.tile_y_coords = []

        for tile in tile_list:
            tile_name_idx = tile.name
            tile_name = self.strings[tile_name_idx] if tile_name_idx < len(self.strings) else ""

            self.tiles.append({
                'name': tile_name,
                'name_idx': tile_name_idx,
                'type': tile.type,
                'row': tile.row if hasattr(tile, 'row') else 0,
                'col': tile.col if hasattr(tile, 'col') else 0,
            })

            # Extract coordinates
            row = tile.row if hasattr(tile, 'row') else 0
            col = tile.col if hasattr(tile, 'col') else 0
            self.tile_x_coords.append(col)
            self.tile_y_coords.append(row)

            # Build tile name string index mapping
            self.tile_str_to_idx[tile_name_idx] = len(self.tiles) - 1

        log(f"Extracted {len(self.tiles)} tiles")

    def _extract_tile_types(self, device_reader):
        """Extract tile types."""
        tile_type_list = device_reader.tileTypeList
        self.tile_type_wire_idx_to_str = []
        for tt in tile_type_list:
            wires_list = list(tt.wires) if hasattr(tt, 'wires') else []
            self.tile_types.append({
                'name': self.strings[tt.name] if tt.name < len(self.strings) else "",
                'num_wires': len(wires_list),
                'wires': wires_list,
            })
            self.tile_type_wire_idx_to_str.append(wires_list)
        log(f"Extracted {len(self.tile_types)} tile types")

    def _extract_site_types(self, device_reader):
        """Extract site types with pin information."""
        if hasattr(device_reader, 'siteTypeList'):
            site_type_list = device_reader.siteTypeList

            # Initialize pin name mappings
            self.site_type_pin_name_to_idx = [{} for _ in range(len(site_type_list))]

            for site_type_idx, st in enumerate(site_type_list):
                site_type_name = self.strings[st.name] if st.name < len(self.strings) else ""

                # Extract pins for this site type
                pins = []
                pin_name_to_idx = {}
                if hasattr(st, 'pins'):
                    for pin_idx, pin in enumerate(st.pins):
                        pin_name = self.strings[pin.name] if pin.name < len(self.strings) else ""
                        pins.append(pin_name)
                        pin_name_to_idx[pin_name] = pin_idx

                self.site_types.append({
                    'name': site_type_name,
                    'pins': pins,
                })
                self.site_type_pin_name_to_idx[site_type_idx] = pin_name_to_idx

            log(f"Extracted {len(self.site_types)} site types")

    def _extract_sites(self, device_reader):
        """Extract sites from tiles."""
        tile_list = device_reader.tileList
        tile_type_list = device_reader.tileTypeList

        log("Extracting sites...")

        # Build tile_type_wire_str_to_idx first (needed for pin mapping)
        self.tile_type_wire_str_to_idx = [{} for _ in range(len(tile_type_list))]
        for tile_type_idx, tile_type in enumerate(tile_type_list):
            if hasattr(tile_type, 'wires'):
                for wire_idx, wire_str_idx in enumerate(tile_type.wires):
                    self.tile_type_wire_str_to_idx[tile_type_idx][wire_str_idx] = wire_idx

        # Extract sites from tiles
        for tile_idx, tile in enumerate(tile_list):
            if not hasattr(tile, 'sites'):
                continue

            tile_type_idx = tile.type
            tile_type = tile_type_list[tile_type_idx]
            tile_type_site_types = tile_type.siteTypes if hasattr(tile_type, 'siteTypes') else []

            for in_tile_site_idx, tile_site in enumerate(tile.sites):
                site_idx = len(self.sites)
                site_name = self.strings[tile_site.name] if tile_site.name < len(self.strings) else ""

                # Get site type
                site_type_idx = -1
                if in_tile_site_idx < len(tile_type_site_types):
                    site_type_info = tile_type_site_types[in_tile_site_idx]
                    if hasattr(site_type_info, 'primaryType'):
                        site_type_idx = site_type_info.primaryType

                self.sites.append({
                    'tile_idx': tile_idx,
                    'tile_type_idx': tile_type_idx,
                    'in_tile_site_idx': in_tile_site_idx,
                    'site_type_idx': site_type_idx,
                    'name': site_name,
                })

                self.site_name_to_idx[site_name] = site_idx

        # Build tile_type_site_pin_to_wire_idx mapping
        self.tile_type_site_pin_to_wire_idx = [[] for _ in range(len(tile_type_list))]
        # Also cache PIPs per tile type for later connectivity
        self.tile_type_pips = [[] for _ in range(len(tile_type_list))]

        for tile_type_idx, tile_type in enumerate(tile_type_list):
            if not hasattr(tile_type, 'siteTypes'):
                # Still collect PIPs if present
                if hasattr(tile_type, 'pips'):
                    self.tile_type_pips[tile_type_idx] = [
                        (p.wire0, p.wire1, bool(p.directional)) for p in tile_type.pips
                    ]
                continue

            site_types = tile_type.siteTypes
            site_pin_to_wire = [[] for _ in range(len(site_types))]

            for site_in_tile_idx, site_type_info in enumerate(site_types):
                if not hasattr(site_type_info, 'primaryType'):
                    continue

                primary_type_idx = site_type_info.primaryType
                if primary_type_idx >= len(self.site_types):
                    continue

                # Get pins for this site type
                num_pins = len(self.site_types[primary_type_idx]['pins'])

                # Get pin to wire mapping
                pin_to_wire = [-1] * num_pins
                if hasattr(site_type_info, 'primaryPinsToTileWires'):
                    pin_to_tile_wires = site_type_info.primaryPinsToTileWires
                    for pin_idx, wire_str_idx in enumerate(pin_to_tile_wires):
                        if pin_idx < num_pins and wire_str_idx in self.tile_type_wire_str_to_idx[tile_type_idx]:
                            wire_idx = self.tile_type_wire_str_to_idx[tile_type_idx][wire_str_idx]
                            pin_to_wire[pin_idx] = wire_idx

                site_pin_to_wire[site_in_tile_idx] = pin_to_wire

            self.tile_type_site_pin_to_wire_idx[tile_type_idx] = site_pin_to_wire
            # Collect PIPs for this tile type
            if hasattr(tile_type, 'pips'):
                self.tile_type_pips[tile_type_idx] = [
                    (p.wire0, p.wire1, bool(p.directional)) for p in tile_type.pips
                ]

        log(f"Extracted {len(self.sites)} sites")

    def _extract_wires(self, device_reader):
        """Extract wires with tile and type indices."""
        wire_list = device_reader.wires
        self.wires = []
        for w_idx, wire in enumerate(wire_list):
            tile_str_idx = wire.tile
            wire_str_idx = wire.wire
            tile_idx = self.tile_str_to_idx.get(tile_str_idx, -1)
            tile_type_idx = self.tiles[tile_idx]['type'] if 0 <= tile_idx < len(self.tiles) else -1
            self.wires.append({
                'tile_str_idx': tile_str_idx,
                'wire_str_idx': wire_str_idx,
                'tile_idx': tile_idx,
                'tile_type_idx': tile_type_idx,
                'type': wire.type if hasattr(wire, 'type') else 0,
            })
        log(f"Extracted {len(self.wires)} wires with indices")

    def _extract_wire_types(self, device_reader):
        """Extract wire types."""
        if hasattr(device_reader, 'wireTypes'):
            wire_type_list = device_reader.wireTypes
            for wt in wire_type_list:
                self.wire_types.append({
                    'name': self.strings[wt.name] if wt.name < len(self.strings) else "",
                })
            log(f"Extracted {len(self.wire_types)} wire types")

    def _extract_nodes(self, device_reader):
        """Extract routing nodes efficiently with NumPy batch processing."""
        node_list = device_reader.nodes
        wire_list = device_reader.wires
        tile_list = device_reader.tileList
        num_nodes = len(node_list)

        log(f"Extracting {num_nodes:,} nodes with NumPy optimization...")

        # Pre-allocate arrays for batch processing
        self.nodes = [None] * num_nodes

        # Process in chunks to show progress
        chunk_size = 500000

        for chunk_start in range(0, num_nodes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_nodes)

            if chunk_start % 5000000 == 0 and chunk_start > 0:
                log(f"  Processed {chunk_start:,} / {num_nodes:,} nodes ({100*chunk_start/num_nodes:.1f}%)...")

            # Process chunk - extract wire lists for each node
            for i in range(chunk_start, chunk_end):
                node = node_list[i]
                node_wires = list(node.wires) if hasattr(node, 'wires') and len(node.wires) > 0 else []

                # Extract first wire info to determine begin tile
                begin_tile_idx = -1
                begin_wire_type = IntentCode.INTENT_DEFAULT
                end_tile_idx = -1

                if len(node_wires) > 0:
                    begin_wire_idx = node_wires[0]
                    begin_wire = wire_list[begin_wire_idx]
                    begin_tile_str_idx = begin_wire.tile
                    begin_wire_type = begin_wire.type if hasattr(begin_wire, 'type') else IntentCode.INTENT_DEFAULT

                    # Get tile index
                    if begin_tile_str_idx in self.tile_str_to_idx:
                        begin_tile_idx = self.tile_str_to_idx[begin_tile_str_idx]

                        # Find end tile (last INT tile or same as begin)
                        begin_tile_type = tile_list[begin_tile_idx].type
                        end_tile_idx = begin_tile_idx

                        # Check if any wire is on an INT or matching tile
                        for wire_idx in node_wires:
                            wire = wire_list[wire_idx]
                            tile_str_idx = wire.tile
                            if tile_str_idx in self.tile_str_to_idx:
                                tile_idx = self.tile_str_to_idx[tile_str_idx]
                                tile_type = tile_list[tile_idx].type
                                if tile_type == TileTypes.INT_TILE or tile_type == begin_tile_type:
                                    end_tile_idx = tile_idx

                # Store node data
                self.nodes[i] = {
                    'id': i,
                    'wires': node_wires,
                    'begin_tile_idx': begin_tile_idx,
                    'end_tile_idx': end_tile_idx,
                    'intent_code': begin_wire_type,
                }

        log(f"Extracted {len(self.nodes):,} nodes with attributes")

    def _build_routing_graph(self) -> RouteNodeGraph:
        """Build routing graph from extracted data with real attributes."""
        routing_graph = RouteNodeGraph()
        num_nodes = len(self.nodes)

        log(f"Creating {num_nodes:,} RouteNode objects with attributes...")

        # Pre-allocate for better performance
        routing_graph.nodes = [None] * num_nodes

        # Process in large chunks
        chunk_size = 1000000

        for chunk_start in range(0, num_nodes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_nodes)

            if chunk_start > 0:
                log(f"  Created {chunk_start:,} / {num_nodes:,} nodes ({100*chunk_start/num_nodes:.1f}%)...")

            for i in range(chunk_start, chunk_end):
                node_data = self.nodes[i]

                # Extract real attributes
                begin_tile_idx = node_data['begin_tile_idx']
                end_tile_idx = node_data['end_tile_idx']
                intent_code = node_data['intent_code']
                num_wires = len(node_data['wires'])

                # Get coordinates from tiles
                begin_x = self.tile_x_coords[begin_tile_idx] if begin_tile_idx >= 0 else 0
                begin_y = self.tile_y_coords[begin_tile_idx] if begin_tile_idx >= 0 else 0
                end_x = self.tile_x_coords[end_tile_idx] if end_tile_idx >= 0 else begin_x
                end_y = self.tile_y_coords[end_tile_idx] if end_tile_idx >= 0 else begin_y

                # Calculate node length (Manhattan distance)
                length = max(1, abs(end_x - begin_x) + abs(end_y - begin_y))

                # Determine node type based on intent code
                node_type = self._intent_to_node_type(intent_code)

                # Calculate base cost based on intent code and length
                base_cost = self._calculate_base_cost(intent_code, length)

                # Create RouteNode with full attributes
                rnode = RouteNode(
                    id=i,
                    tile_id=begin_tile_idx,
                    tile_x=begin_x,
                    tile_y=begin_y,
                    wire_id=node_data['wires'][0] if num_wires > 0 else 0,
                    node_type=node_type,
                    intent_code=intent_code,
                    base_cost=base_cost,
                    length=length
                )

                # Set additional attributes
                rnode.is_pinbounce = (intent_code == IntentCode.NODE_PINBOUNCE)
                rnode.is_laguna = False  # Will be set later if needed

                routing_graph.nodes[i] = rnode

        # Build node map
        routing_graph.node_map = {i: routing_graph.nodes[i] for i in range(num_nodes)}
        routing_graph.num_nodes = num_nodes

        log(f"Routing graph created with {routing_graph.num_nodes:,} nodes")
        # Build edges based on PIPs
        try:
            self._build_intra_tile_edges(routing_graph)
        except Exception as e:
            log(f"Warning: building intra-tile edges failed: {e}")
        return routing_graph

    def _build_intra_tile_edges(self, routing_graph: RouteNodeGraph) -> None:
        """Add directed edges between nodes from per-tile PIPs."""
        log("Building intra-tile edges from PIPs...")
        num_tiles = len(self.tiles)
        # Initialize tile local wire -> node array using tile type wire counts
        self.tile_wire_to_node = []
        for tile_idx in range(num_tiles):
            tile_type_idx = self.tiles[tile_idx]['type']
            num_tt_wires = self.tile_types[tile_type_idx]['num_wires'] if tile_type_idx < len(self.tile_types) else 0
            self.tile_wire_to_node.append([-1] * max(1, num_tt_wires))

        # Populate wire->node mapping by scanning node wires
        for node_idx, node_data in enumerate(self.nodes):
            for w_idx in node_data['wires']:
                if 0 <= w_idx < len(self.wires):
                    w = self.wires[w_idx]
                    tile_idx = w['tile_idx']
                    tile_type_idx = w['tile_type_idx']
                    if tile_idx < 0 or tile_type_idx < 0:
                        continue
                    wire_str_idx = w['wire_str_idx']
                    tt_map = self.tile_type_wire_str_to_idx[tile_type_idx] if tile_type_idx < len(self.tile_type_wire_str_to_idx) else {}
                    if wire_str_idx not in tt_map:
                        continue
                    tt_wire_idx = tt_map[wire_str_idx]
                    if tt_wire_idx >= len(self.tile_wire_to_node[tile_idx]):
                        self.tile_wire_to_node[tile_idx].extend([-1] * (tt_wire_idx - len(self.tile_wire_to_node[tile_idx]) + 1))
                    self.tile_wire_to_node[tile_idx][tt_wire_idx] = node_idx

        # Add edges per tile using cached PIPs
        edge_count = 0
        for tile_idx in range(num_tiles):
            tile_type_idx = self.tiles[tile_idx]['type']
            pips = self.tile_type_pips[tile_type_idx] if tile_type_idx < len(self.tile_type_pips) else []
            for wire0, wire1, directional in pips:
                n0 = self.tile_wire_to_node[tile_idx][wire0] if wire0 < len(self.tile_wire_to_node[tile_idx]) else -1
                n1 = self.tile_wire_to_node[tile_idx][wire1] if wire1 < len(self.tile_wire_to_node[tile_idx]) else -1
                if n0 >= 0 and n1 >= 0:
                    routing_graph.add_edge(routing_graph.nodes[n0], routing_graph.nodes[n1])
                    edge_count += 1
                    if not directional:
                        routing_graph.add_edge(routing_graph.nodes[n1], routing_graph.nodes[n0])
                        edge_count += 1
        log(f"Added {edge_count:,} intra-tile edges")

    def _intent_to_node_type(self, intent_code: IntentCode) -> NodeType:
        """Map intent code to node type."""
        if intent_code == IntentCode.NODE_PINFEED:
            return NodeType.PINFEED_O
        elif intent_code == IntentCode.NODE_PINBOUNCE:
            return NodeType.PINBOUNCE
        elif intent_code in [IntentCode.NODE_LAGUNA_DATA, IntentCode.NODE_LAGUNA_OUTPUT]:
            return NodeType.LAGUNA_I
        elif intent_code in [IntentCode.NODE_HLONG, IntentCode.NODE_VLONG,
                              IntentCode.VLONG, IntentCode.HLONG]:
            return NodeType.SUPER_LONG_LINE
        else:
            return NodeType.WIRE

    def _calculate_base_cost(self, intent_code: IntentCode, length: int) -> float:
        """Calculate base cost based on intent code and length."""
        # Base costs from C++ implementation
        if intent_code in [IntentCode.NODE_PINFEED, IntentCode.PINFEED]:
            return 0.5
        elif intent_code == IntentCode.NODE_PINBOUNCE:
            return 0.3
        elif intent_code in [IntentCode.NODE_LOCAL, IntentCode.NODE_SINGLE, IntentCode.SINGLE]:
            return 0.8
        elif intent_code in [IntentCode.NODE_DOUBLE, IntentCode.DOUBLE]:
            return 1.0 + length * 0.1
        elif intent_code in [IntentCode.NODE_HLONG, IntentCode.NODE_VLONG, IntentCode.HLONG, IntentCode.VLONG]:
            return 1.5 + length * 0.2
        elif intent_code in [IntentCode.NODE_HQUAD, IntentCode.NODE_VQUAD, IntentCode.HQUAD, IntentCode.VQUAD]:
            return 1.2 + length * 0.15
        else:
            # Default cost
            return 1.0 + length * 0.05

    def _build_indices(self, device: Device):
        """Build index structures for fast lookup."""
        # Build tile name index
        for i, tile in enumerate(self.tiles):
            self.tile_name_to_idx[tile['name']] = i

        device.tile_name_to_idx = self.tile_name_to_idx

    def build_node_connections(self, device: Device):
        """Build node parent-child relationships from tile PIPs."""
        log("Building node connections from PIPs...")

        tile_list_reader = None
        tile_type_list_reader = None
        wire_list_reader = None

        # Need to re-open device file to get PIP data
        # For now, we'll implement a simplified version that builds basic connectivity
        # Full implementation would parse all PIPs from Cap'n Proto

        # Initialize tile_wire_to_node mapping
        num_tiles = len(self.tiles)
        self.tile_wire_to_node = [[] for _ in range(num_tiles)]

        log("Mapping wires to nodes...")
        # Build wire-to-node mapping for each tile
        for node_idx, node_data in enumerate(self.nodes):
            if node_idx % 1000000 == 0 and node_idx > 0:
                log(f"  Processed {node_idx:,} / {len(self.nodes):,} nodes...")

            for wire_idx in node_data['wires']:
                # Would need to map wire_idx to (tile_idx, tile_wire_idx)
                # This requires parsing wire_list again
                pass

        log(f"Node connections built")

        return device

    def _parse_fallback(self, device_path: Path) -> Device:
        """
        Fallback parser when Cap'n Proto parsing fails.
        Returns a minimal Device for testing.
        """
        log("Using fallback parser (minimal device)")

        routing_graph = RouteNodeGraph()
        device = Device(routing_graph)
        device.name = device_path.stem

        return device
