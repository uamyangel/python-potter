"""Global definitions, constants, and type aliases for Potter router."""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import IntEnum

# Type aliases
StrIdx = int
ObjIdx = int
CostType = int
INVALID_OBJ_IDX = 2**32 - 1
INFTY = 4294967295

# Node attribute indices
class NodeAttrIdx(IntEnum):
    RNODE_ID = 0
    BEG_TILE_ID = 1
    END_TILE_X = 2
    END_TILE_Y = 3
    NODE_BASE_COST = 4
    BEG_TILE_X = 5
    BEG_TILE_Y = 6
    NODE_LEN = 7
    WIRE_ID = 8
    NODE_TYPE = 9
    IS_NODE_PINBOUNCE = 10
    END_TILE_ID = 11
    INTENT_CODE = 12
    TILE_TYPE = 13
    CONSIDERED = 14
    LAGUNA = 15
    PRESERVED = 16
    NODE_ATTR_CNT = 17

# Edge attribute indices
class EdgeAttrIdx(IntEnum):
    START_NODE_ID = 0
    END_NODE_ID = 1
    TILE_X = 2
    TILE_Y = 3
    EDGE_ATTR_CNT = 4

# Connection attribute indices
class ConnAttrIdx(IntEnum):
    HASH_CODE = 0
    SRC_NODE_ID = 1
    SNK_NODE_ID = 2
    X_MIN = 3
    X_MAX = 4
    Y_MIN = 5
    Y_MAX = 6
    NET_ID = 7
    CENTER_X = 8
    CENTER_Y = 9
    DOUBLE_HPWL = 10
    IS_INDIRECT = 11
    CONN_ATTR_CNT = 12

# Tile types
class TileTypes(IntEnum):
    OTHERS = 0
    INT_TILE = 1
    LAG_TILE = 2

# Node types
class NodeType(IntEnum):
    PINFEED_O = 0
    PINFEED_I = 1
    PINBOUNCE = 2
    SUPER_LONG_LINE = 3
    LAGUNA_I = 4
    WIRE = 5

# Intent codes (wire types)
class IntentCode(IntEnum):
    INTENT_DEFAULT = 0
    NODE_OUTPUT = 1
    NODE_DEDICATED = 2
    NODE_GLOBAL_VDISTR = 3
    NODE_GLOBAL_HROUTE = 4
    NODE_GLOBAL_HDISTR = 5
    NODE_PINFEED = 6
    NODE_PINBOUNCE = 7
    NODE_LOCAL = 8
    NODE_HLONG = 9
    NODE_SINGLE = 10
    NODE_DOUBLE = 11
    NODE_HQUAD = 12
    NODE_VLONG = 13
    NODE_VQUAD = 14
    NODE_OPTDELAY = 15
    NODE_GLOBAL_VROUTE = 16
    NODE_GLOBAL_LEAF = 17
    NODE_GLOBAL_BUFG = 18
    NODE_LAGUNA_DATA = 19
    NODE_CLE_OUTPUT = 20
    NODE_INT_INTERFACE = 21
    NODE_LAGUNA_OUTPUT = 22
    GENERIC = 23
    DOUBLE = 24
    BENTQUAD = 26
    SLOWSINGLE = 27
    CLKPIN = 28
    PINFEED = 31
    BOUNCEIN = 32
    LUTINPUT = 33
    IOBOUTPUT = 34
    BOUNCEACROSS = 35
    VLONG = 36
    OUTBOUND = 37
    HLONG = 38
    BUFGROUT = 40
    PINFEEDR = 41
    OPTDELAY = 42
    IOBIN2OUT = 43
    HQUAD = 44
    IOBINPUT = 45
    PADINPUT = 46
    PADOUTPUT = 47
    VLONG12 = 48
    HVCCGNDOUT = 49
    SVLONG = 50
    VQUAD = 51
    SINGLE = 52
    BUFINP2OUT = 53
    REFCLK = 54
    NODE_INTF4 = 55
    NODE_INTF2 = 56
    NODE_CLE_BNODE = 57
    NODE_CLE_CNODE = 58
    NODE_CLE_CTRL = 59
    NODE_HLONG10 = 60
    NODE_HLONG6 = 61
    NODE_VLONG12 = 62
    NODE_VLONG7 = 63
    NODE_SDQNODE = 64
    NODE_IMUX = 65
    NODE_INODE = 66
    NODE_HSINGLE = 67
    NODE_HDOUBLE = 68
    NODE_VSINGLE = 69
    NODE_VDOUBLE = 70
    NODE_INTF_BNODE = 71
    NODE_INTF_CNODE = 72
    NODE_INTF_CTRL = 73
    NODE_IRI = 74
    NODE_OPTDELAY_MUX = 75
    NODE_CLE_LNODE = 76
    NODE_GLOBAL_VDISTR_LVL2 = 77
    NODE_GLOBAL_VDISTR_LVL1 = 78
    NODE_GLOBAL_GCLK = 79
    NODE_GLOBAL_HROUTE_HSR = 80
    NODE_GLOBAL_HDISTR_HSR = 81
    NODE_GLOBAL_HDISTR_LOCAL = 82
    NODE_SLL_INPUT = 83
    NODE_SLL_OUTPUT = 84
    NODE_SLL_DATA = 85
    NODE_GLOBAL_VDISTR_LVL3 = 86
    NODE_GLOBAL_VDISTR_LVL21 = 87
    NODE_GLOBAL_VDISTR_SHARED = 88

# Constants
PRESENT_COST_FACTOR = 1
HISTORY_COST_FACTOR = 1
MAX_NODE_NUM = 30000000
MAX_EDGE_NUM = 130000000
MAX_CONN_NUM = 10000000
MAX_NET_NUM = 1000000


@dataclass
class Box:
    """Bounding box representation."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def overlaps(self, other: 'Box') -> bool:
        return not (self.x_max < other.x_min or other.x_max < self.x_min or
                   self.y_max < other.y_min or other.y_max < self.y_min)
