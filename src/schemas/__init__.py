"""Cap'n Proto schemas for FPGA interchange format."""

import os
import sys
from pathlib import Path

# Use independent schema path (no Potter/ dependency)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCHEMA_PATH = PROJECT_ROOT / "fpga-interchange-schema" / "interchange"

if not SCHEMA_PATH.exists():
    raise FileNotFoundError(
        f"FPGA interchange schema not found at {SCHEMA_PATH}. "
        f"Please ensure fpga-interchange-schema/ directory exists in project root."
    )

# Import pycapnp
try:
    import capnp
except ImportError:
    raise ImportError(
        "pycapnp is required but not installed. "
        "Install with: pip install pycapnp"
    )

# Load schemas
DeviceResources = None
PhysicalNetlist = None
LogicalNetlist = None

def load_schemas():
    """Load Cap'n Proto schemas."""
    global DeviceResources, PhysicalNetlist, LogicalNetlist

    if DeviceResources is not None:
        return  # Already loaded

    schema_files = {
        'DeviceResources': SCHEMA_PATH / 'DeviceResources.capnp',
        'PhysicalNetlist': SCHEMA_PATH / 'PhysicalNetlist.capnp',
        'LogicalNetlist': SCHEMA_PATH / 'LogicalNetlist.capnp',
    }

    # Verify all schema files exist
    for name, path in schema_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

    # Load schemas with schema path
    # Need to include parent directory for /capnp/ imports
    import_paths = [str(SCHEMA_PATH), str(SCHEMA_PATH.parent)]

    try:
        DeviceResources = capnp.load(
            str(schema_files['DeviceResources']),
            imports=import_paths
        )
        PhysicalNetlist = capnp.load(
            str(schema_files['PhysicalNetlist']),
            imports=import_paths
        )
        LogicalNetlist = capnp.load(
            str(schema_files['LogicalNetlist']),
            imports=import_paths
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Cap'n Proto schemas: {e}")


# Auto-load on import
load_schemas()
