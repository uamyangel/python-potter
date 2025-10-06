"""Netlist parsing and writing."""

import pickle
import gzip
from pathlib import Path
from typing import List, Dict
from .route_node import Connection, Net, RouteResult, RouteNode
from .route_node_graph import RouteNodeGraph
from .device import Device
from ..global_defs import Box, NodeType
from ..utils.log import log


class Netlist:
    """Manages circuit netlist with nets and connections."""

    def __init__(
        self,
        device: Device,
        nets: List[Net],
        indirect_connections: List[Connection],
        direct_connections: List[Connection],
        preserved_nodes: List[bool],
        routing_graph: RouteNodeGraph,
        layout: Box
    ):
        self.device = device
        self.nets = nets
        self.indirect_connections = indirect_connections
        self.direct_connections = direct_connections
        self.preserved_nodes = preserved_nodes
        self.routing_graph = routing_graph
        self.layout = layout
        self.num_thread = 16
        # Original input path for writeback
        self._input_netlist_path: str = ""
        # Optional: original strings from netlist
        self._orig_strs: List[str] = []
        # Map net name -> our Net id
        self.net_name_to_id = {}

    def read(self, netlist_file: str) -> None:
        """Read netlist file."""
        log(f"Reading netlist file: {netlist_file}")

        netlist_path = Path(netlist_file)
        if not netlist_path.exists():
            raise FileNotFoundError(f"Netlist file not found: {netlist_file}")

        # Always parse netlist fresh to avoid heavy object caching
        self._input_netlist_path = str(netlist_path)
        log("Parsing netlist file...")
        self._parse_netlist_file(netlist_path)

        log(f"Netlist loaded: {len(self.nets)} nets, "
            f"{len(self.indirect_connections)} indirect connections, "
            f"{len(self.direct_connections)} direct connections")

    def _parse_netlist_file(self, netlist_path: Path) -> None:
        """Parse netlist file from FPGA interchange format."""
        try:
            import gzip
            import capnp
            from ..schemas import PhysicalNetlist as PN

            log("Parsing PhysicalNetlist Cap'n Proto file...")

            # Decompress gzip
            with gzip.open(str(netlist_path), 'rb') as f:
                data = f.read()

            log(f"Decompressed netlist size: {len(data) / 1024 / 1024:.2f} MB")

            # Parse Cap'n Proto with high traversal limit
            with PN.PhysNetlist.from_bytes(data, traversal_limit_in_words=2**63-1) as netlist_reader:
                # Extract string list
                str_list = netlist_reader.strList
                self.strings = [str(s) for s in str_list]
                self._orig_strs = list(self.strings)
                log(f"Strings: {len(self.strings)}")

                # Extract physical nets
                phys_nets = netlist_reader.physNets
                log(f"Physical nets: {len(phys_nets)}")

                # Process each net
                net_id = 0
                indirect_conn_id = 0
                direct_conn_id = 0

                for phys_net_idx, phys_net in enumerate(phys_nets):
                    if phys_net_idx % 10000 == 0 and phys_net_idx > 0:
                        log(f"  Processed {phys_net_idx:,} / {len(phys_nets):,} nets...")

                    net_type = phys_net.type
                    sources = phys_net.sources
                    stubs = phys_net.stubs

                    # Skip if not a signal net
                    if net_type != PN.PhysNetlist.NetType.signal:
                        continue

                    if len(stubs) == 0:
                        continue

                    # Extract source pins
                    source_pins = self._extract_site_pins(sources, str_list)

                    # Extract sink pins (one per stub)
                    sink_pins = self._extract_site_pins_one_by_one(stubs, str_list)

                    if len(sink_pins) == 0 or len(source_pins) == 0:
                        continue

                    # Create net
                    net = Net(id=net_id, name=self.strings[phys_net.name] if phys_net.name < len(self.strings) else "")
                    if net.name:
                        self.net_name_to_id[net.name] = net_id

                    # Get source nodes
                    src_node_candidates = []
                    for site_idx, pin_idx in source_pins:
                        site_name = self.strings[site_idx]
                        pin_name = self.strings[pin_idx]
                        src_node_idx = self._get_site_pin_node(site_name, pin_name)
                        if src_node_idx is not None:
                            src_node_candidates.append(src_node_idx)

                    if len(src_node_candidates) == 0:
                        continue

                    # Get primary source
                    src_node_idx = src_node_candidates[0]
                    src_int_node_idx = self._project_output_to_int(src_node_idx)

                    # Process each sink
                    for sink_site_idx, sink_pin_idx in sink_pins:
                        sink_site_name = self.strings[sink_site_idx]
                        sink_pin_name = self.strings[sink_pin_idx]
                        sink_node_idx = self._get_site_pin_node(sink_site_name, sink_pin_name)

                        if sink_node_idx is None:
                            continue

                        sink_int_node_idx = self._project_input_to_int(sink_node_idx)

                        # Determine if indirect or direct connection
                        is_indirect = (sink_int_node_idx is not None) and (src_int_node_idx is not None)

                        if is_indirect:
                            # Create indirect connection
                            src_rnode = self.routing_graph.get_node(src_int_node_idx)
                            sink_rnode = self.routing_graph.get_node(sink_int_node_idx)

                            if src_rnode and sink_rnode:
                                # Calculate bounding box
                                x_min = min(src_rnode.tile_x, sink_rnode.tile_x)
                                x_max = max(src_rnode.tile_x, sink_rnode.tile_x)
                                y_min = min(src_rnode.tile_y, sink_rnode.tile_y)
                                y_max = max(src_rnode.tile_y, sink_rnode.tile_y)
                                center_x = (x_min + x_max) / 2.0
                                center_y = (y_min + y_max) / 2.0
                                hpwl = abs(x_max - x_min) + abs(y_max - y_min)

                                conn = Connection(
                                    id=indirect_conn_id,
                                    net_id=net_id,
                                    source_node=src_rnode,
                                    sink_node=sink_rnode,
                                    x_min=x_min,
                                    x_max=x_max,
                                    y_min=y_min,
                                    y_max=y_max,
                                    center_x=center_x,
                                    center_y=center_y,
                                    hpwl=hpwl,
                                    is_indirect=True
                                )

                                self.indirect_connections.append(conn)
                                net.connections.append(conn)
                                indirect_conn_id += 1

                                # Mark nodes as PINFEED
                                src_rnode.node_type = NodeType.PINFEED_O
                                sink_rnode.node_type = NodeType.PINFEED_I

                        else:
                            # Create direct connection
                            src_rnode = self.routing_graph.get_node(src_node_idx)
                            sink_rnode = self.routing_graph.get_node(sink_node_idx)

                            if src_rnode and sink_rnode:
                                x_min = min(src_rnode.tile_x, sink_rnode.tile_x)
                                x_max = max(src_rnode.tile_x, sink_rnode.tile_x)
                                y_min = min(src_rnode.tile_y, sink_rnode.tile_y)
                                y_max = max(src_rnode.tile_y, sink_rnode.tile_y)
                                center_x = (x_min + x_max) / 2.0
                                center_y = (y_min + y_max) / 2.0
                                hpwl = abs(x_max - x_min) + abs(y_max - y_min)

                                conn = Connection(
                                    id=direct_conn_id,
                                    net_id=net_id,
                                    source_node=src_rnode,
                                    sink_node=sink_rnode,
                                    x_min=x_min,
                                    x_max=x_max,
                                    y_min=y_min,
                                    y_max=y_max,
                                    center_x=center_x,
                                    center_y=center_y,
                                    hpwl=hpwl,
                                    is_indirect=False
                                )

                                self.direct_connections.append(conn)
                                net.connections.append(conn)
                                direct_conn_id += 1

                                src_rnode.node_type = NodeType.PINFEED_O

                    # Update net bounding box
                    if len(net.connections) > 0:
                        net.x_min = min(c.x_min for c in net.connections)
                        net.x_max = max(c.x_max for c in net.connections)
                        net.y_min = min(c.y_min for c in net.connections)
                        net.y_max = max(c.y_max for c in net.connections)
                        net.center_x = (net.x_min + net.x_max) / 2.0
                        net.center_y = (net.y_min + net.y_max) / 2.0
                        net.hpwl = sum(c.hpwl for c in net.connections)

                        self.nets.append(net)
                        net_id += 1

                log(f"Extracted {net_id} nets, {indirect_conn_id} indirect connections, {direct_conn_id} direct connections")

        except Exception as e:
            log(f"Error parsing netlist: {e}")
            import traceback
            traceback.print_exc()
            log("Netlist parsing failed")

    def _extract_site_pins(self, branches, str_list):
        """Extract all site pins from route branches."""
        from collections import deque

        site_pins = []
        queue = deque(branches)

        while queue:
            branch = queue.popleft()
            route_segment = branch.routeSegment

            # Check if this is a site pin
            if route_segment.which() == 'sitePin':
                sp = route_segment.sitePin
                site_pins.append((sp.site, sp.pin))

            # Add child branches to queue
            for child in branch.branches:
                queue.append(child)

        return site_pins

    def _extract_site_pins_one_by_one(self, branches, str_list):
        """Extract one site pin per branch."""
        from collections import deque

        site_pins = []

        for branch in branches:
            queue = deque([branch])

            while queue:
                b = queue.popleft()
                route_segment = b.routeSegment

                if route_segment.which() == 'sitePin':
                    sp = route_segment.sitePin
                    site_pins.append((sp.site, sp.pin))
                    break

                for child in b.branches:
                    queue.append(child)

        return site_pins

    def _get_site_pin_node(self, site_name: str, pin_name: str):
        """Get node index for a site pin using device mapping."""
        if not hasattr(self.device, 'get_site_pin_node'):
            return None
        return self.device.get_site_pin_node(site_name, pin_name)

    def _project_output_to_int(self, node_idx: int):
        """Project output node to INT tile node."""
        # TODO: Follow graph children until reaching INT tile wire if needed.
        # For now, return original node index to enable routing on available graph.
        return node_idx

    def _project_input_to_int(self, node_idx: int):
        """Project input node to INT tile node."""
        # TODO: Follow graph parents until reaching INT tile wire if needed.
        # For now, return original node index to enable routing on available graph.
        return node_idx

    def _load_cached(self, cache_path: Path) -> None:
        """Deprecated: netlist caching disabled for stability and size."""
        raise RuntimeError("Netlist caching disabled")

    def _save_cached(self, cache_path: Path) -> None:
        """Deprecated: netlist caching disabled for stability and size."""
        return

    def write(self, output_file: str, routing_results: List[RouteResult]) -> None:
        """Write routed netlist to FPGA interchange .phys using Cap'n Proto."""
        log(f"Writing routed netlist to: {output_file}")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import gzip
            from ..schemas import PhysicalNetlist as PN

            # Load original netlist reader to copy metadata and names
            with gzip.open(self._input_netlist_path, 'rb') as f:
                in_bytes = f.read()
            with PN.PhysNetlist.from_bytes(in_bytes, traversal_limit_in_words=2**63-1) as reader:
                # Prepare string table: start from original, extend as needed
                str_list = [s for s in reader.strList]

            def ensure_str(s: str) -> int:
                try:
                    return str_list.index(s)
                except ValueError:
                    str_list.append(s)
                    return len(str_list) - 1

            # Group routing results by net name
            net_results: Dict[str, List[RouteResult]] = {}
            for rr in routing_results:
                net_name = self.nets[rr.net_id].name if rr.net_id < len(self.nets) else ""
                if not net_name:
                    continue
                net_results.setdefault(net_name, []).append(rr)

            # Build new PhysNetlist
            msg = PN.PhysNetlist.new_message()
            msg.part = reader.part

            # Strings
            msg.init('strList', len(str_list))
            for i, s in enumerate(str_list):
                msg.strList[i] = s

            # PhysNets: copy topology by name, attach routed PIPs as stubs
            orig_nets = reader.physNets
            nets_builder = msg.init('physNets', len(orig_nets))
            for i, onet in enumerate(orig_nets):
                nb = nets_builder[i]
                nb.name = onet.name
                nb.type = onet.type
                # Build stubs from routing results for this net
                name = reader.strList[onet.name]
                results = net_results.get(name, [])
                nb.init('sources', len(onet.sources))
                for si, sbranch in enumerate(onet.sources):
                    # Preserve original sources (site pins)
                    sb = nb.sources[si]
                    if sbranch.routeSegment.which() == 'sitePin':
                        sp = sbranch.routeSegment.sitePin
                        sb.routeSegment.init('sitePin')
                        sb.routeSegment.sitePin.site = sp.site
                        sb.routeSegment.sitePin.pin = sp.pin
                    elif sbranch.routeSegment.which() == 'belPin':
                        bp = sbranch.routeSegment.belPin
                        sb.routeSegment.init('belPin')
                        sb.routeSegment.belPin.site = bp.site
                        sb.routeSegment.belPin.bel = bp.bel
                        sb.routeSegment.belPin.pin = bp.pin
                    else:
                        # Default to an empty sitePin if unknown
                        sb.routeSegment.init('sitePin')
                        sb.routeSegment.sitePin.site = 0
                        sb.routeSegment.sitePin.pin = 0
                    sb.init('branches', 0)
                nb.init('stubs', len(results))
                for si, rr in enumerate(results):
                    branch = nb.stubs[si]
                    # Convert node path to pip chain
                    pips = self._route_result_to_pips(rr)
                    self._fill_branch_with_pip_chain(branch, pips, ensure_str)
                # Preserve empty physCells linkage
                nb.init('stubNodes', 0)

            # Minimal other sections left empty; tools may accept this
            msg.init('placements', 0)
            msg.init('physCells', 0)
            msg.init('siteInsts', 0)
            msg.init('properties', 0)
            # Null net: copy as empty
            msg.nullNet.name = reader.nullNet.name if hasattr(reader, 'nullNet') else 0
            msg.nullNet.init('sources', 0)
            msg.nullNet.init('stubs', 0)

            # Write gzipped output
            out_bytes = msg.to_bytes()
            with gzip.open(output_path, 'wb') as f:
                f.write(out_bytes)
            log("Netlist written successfully (.phys)")

        except Exception as e:
            log(f"Error writing netlist: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to pickle for debugging
            with open(output_path.with_suffix(output_path.suffix + '.pkl'), 'wb') as f:
                pickle.dump({'routing_results': routing_results}, f)
            log("Wrote fallback pickle alongside output")

    def _route_result_to_pips(self, rr: RouteResult):
        """Convert a routed connection path to a list of (tile_str_idx, wire0_str_idx, wire1_str_idx, forward)."""
        pips = []
        nodes = rr.nodes
        if not nodes or len(nodes) < 2:
            return pips
        dev = self.device
        for a, b in zip(nodes[:-1], nodes[1:]):
            found = False
            # Try on tile of a then tile of b
            for tile_idx in (a.tile_id, b.tile_id):
                if tile_idx < 0 or tile_idx >= len(dev.tiles):
                    continue
                tile_type_idx = dev.tiles[tile_idx]['type']
                pips_tt = dev.tile_type_pips[tile_type_idx] if tile_type_idx < len(dev.tile_type_pips) else []
                tt_wires = dev.tile_type_wire_idx_to_str[tile_type_idx] if tile_type_idx < len(dev.tile_type_wire_idx_to_str) else []
                wire_to_node = dev.tile_wire_to_node[tile_idx] if tile_idx < len(dev.tile_wire_to_node) else []
                for (w0, w1, directional) in pips_tt:
                    n0 = wire_to_node[w0] if w0 < len(wire_to_node) else -1
                    n1 = wire_to_node[w1] if w1 < len(wire_to_node) else -1
                    if n0 == a.id and n1 == b.id:
                        tile_str_idx = dev.tiles[tile_idx].get('name_idx', None)
                        if tile_str_idx is None:
                            tile_str_idx = 0
                        wire0_str_idx = tt_wires[w0] if w0 < len(tt_wires) else 0
                        wire1_str_idx = tt_wires[w1] if w1 < len(tt_wires) else 0
                        pips.append((tile_str_idx, wire0_str_idx, wire1_str_idx, True))
                        found = True
                        break
                    if not directional and n0 == b.id and n1 == a.id:
                        tile_str_idx = dev.tiles[tile_idx].get('name_idx', None)
                        if tile_str_idx is None:
                            tile_str_idx = 0
                        wire0_str_idx = tt_wires[w0] if w0 < len(tt_wires) else 0
                        wire1_str_idx = tt_wires[w1] if w1 < len(tt_wires) else 0
                        pips.append((tile_str_idx, wire0_str_idx, wire1_str_idx, False))
                        found = True
                        break
                if found:
                    break
            if not found:
                # No direct PIP found; skip this hop
                continue
        return pips

    def _fill_branch_with_pip_chain(self, branch, pips, ensure_str):
        """Fill a RouteBranch and its nested branches with a linear chain of PIPs."""
        # Build nested branches: each element is a pip segment
        if not pips:
            # Empty branch
            branch.init('branches', 0)
            return
        # Initialize current branch node
        cur = branch
        for idx, (tile_si, w0_si, w1_si, forward) in enumerate(pips):
            # Set this segment as PhysPIP
            cur.routeSegment.init('pip')
            cur.routeSegment.pip.tile = tile_si
            cur.routeSegment.pip.wire0 = w0_si
            cur.routeSegment.pip.wire1 = w1_si
            cur.routeSegment.pip.forward = bool(forward)
            cur.routeSegment.pip.isFixed = False
            # Allocate next branch if not last
            if idx < len(pips) - 1:
                cur.init('branches', 1)
                cur = cur.branches[0]
            else:
                cur.init('branches', 0)
