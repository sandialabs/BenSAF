import re

PAGE_BREAK = re.compile(r'\*{3}\s+AERMOD\s+-\s+VERSION')

# ── Section header patterns ───────────────────────────────────────────────────
# Source group is optional on the header line; concurrent sections omit it.

ANNUAL_AVG_HDR = re.compile(
    r'\*{3}.*?ANNUAL\s+AVERAGE\s+CONCENTRATION'
    r'(?:.*?SOURCE\s+GROUP:\s*(?P<source_group>\S+))?',
    re.IGNORECASE,
)

HIGHEST_HDR = re.compile(
    r'\*{3}.*?(?P<rank>\d+)(?:ST|ND|RD|TH)\s+HIGHEST\s+'
    r'(?P<period>\d+)-HR\s+AVERAGE\s+CONCENTRATION'
    r'(?:.*?SOURCE\s+GROUP:\s*(?P<source_group>\S+))?',
    re.IGNORECASE,
)

CONCURRENT_HDR = re.compile(
    r'\*{3}.*?CONCURRENT\s+(?P<period>\d+)-HR\s+AVERAGE\s+CONCENTRATION'
    r'.*?FOR\s+DAY\s+(?P<day>\d+)\s+OF\s+(?P<year>\d+)',
    re.IGNORECASE,
)

# ── Metadata lines (appear between section header and table) ──────────────────

NETWORK_LINE = re.compile(
    r'NETWORK\s+ID:\s*(?P<id>\S+).*?NETWORK\s+TYPE:\s*(?P<type>\S+)',
    re.IGNORECASE,
)

POLLUTANT_LINE = re.compile(
    r'\*{2}\s*CONC\s+OF\s+(?P<pollutant>\S+)\s+IN\s+(?P<unit>\S+)',
    re.IGNORECASE,
)

ORIGIN_LINE = re.compile(
    r'X-ORIG\s*=\s*(?P<x>[\d.]+).*?Y-ORIG\s*=\s*(?P<y>[\d.]+)',
    re.IGNORECASE,
)

# Used as fallback when source group isn't on the section header line.
SOURCE_GROUP_LINE = re.compile(r'SOURCE\s+GROUP:\s*(?P<group>\S+)', re.IGNORECASE)

# ── Table structure markers ───────────────────────────────────────────────────

# GRIDPOLR column header row:  "(DEGREES) |   10485.18   20970.36 ..."
POLAR_COL_HEADER = re.compile(r'\(DEGREES\)\s*\|')

# A polar data row begins with a direction value followed by '|'.
POLAR_ROW = re.compile(r'^\s*(?P<direction>\d+\.?\d*)\s*\|')

# GRIDCART column header row:  "(METERS) |   500000.00   500050.00 ..." (X-coordinates).
# Mirrors POLAR_COL_HEADER: a table of Y-rows x X-columns, X values listed here.
CART_COL_HEADER = re.compile(r'\(METERS\)\s*\|')

# A Cartesian data row begins with a Y-coordinate value followed by '|'.
CART_ROW = re.compile(r'^\s*(?P<y>-?\d+\.?\d*)\s*\|')

# Matches a decimal number with an optional AERMOD date stamp.
# Handles both plain values ("0.00017") and dated values ("0.01376 (22110820)").
DECIMAL_VALUE = re.compile(r'(?P<value>\d+\.\d+)\s*[a-z]?\s*(?:\((?P<date>\d{8})\))?')
