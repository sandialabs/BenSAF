from collections.abc import Iterator
from enum import Enum, auto
from pathlib import Path
import re

from ._patterns import (
    PAGE_BREAK,
    ANNUAL_AVG_HDR, HIGHEST_HDR, CONCURRENT_HDR,
    NETWORK_LINE, POLLUTANT_LINE, ORIGIN_LINE, SOURCE_GROUP_LINE,
    POLAR_COL_HEADER, POLAR_ROW,
    CART_COL_HEADER, CART_ROW,
    DECIMAL_VALUE,
)
from ._sections import AermodSection


class _State(Enum):
    SCAN       = auto()   # looking for the next section header
    IN_SECTION = auto()   # inside a section, collecting metadata before the table
    IN_TABLE   = auto()   # reading concentration table rows


def parse_file(path: str | Path) -> list[AermodSection]:
    """Parse an AERMOD .ADO, .OUT, or .ADI file and return all result sections found."""
    lines = Path(path).read_text(errors='replace').splitlines()
    return list(_run(lines))


def _run(lines: list[str]) -> Iterator[AermodSection]:
    state = _State.SCAN
    section: AermodSection | None = None

    for line in lines:
        # Page breaks always close the current section regardless of state.
        if PAGE_BREAK.search(line):
            if section is not None:
                yield section
                section = None
            state = _State.SCAN
            continue

        if state == _State.SCAN:
            new_section = _try_section_header(line)
            if new_section is not None:
                section = new_section
                state = _State.IN_SECTION

        elif state == _State.IN_SECTION:
            _update_meta(section, line)
            if POLAR_COL_HEADER.search(line):
                section.distances = _parse_distances(line)
                state = _State.IN_TABLE
            elif CART_COL_HEADER.search(line):
                section.distances = _parse_distances(line)
                state = _State.IN_TABLE

        elif state == _State.IN_TABLE:
            # GRIDPOLR and GRIDCART data rows look identical (a leading number then
            # '|'), so dispatch on the network type recorded during IN_SECTION rather
            # than trying both row patterns.
            if section.network_type == 'GRIDCART':
                if CART_COL_HEADER.search(line):
                    # A repeated column header means the table continues on a new page
                    # before a page-break was emitted — refresh x-values in place.
                    section.distances = _parse_distances(line)
                elif m := CART_ROW.match(line):
                    if section.distances:
                        section.records.extend(_parse_cart_row(m, line, section.distances))
            else:
                if POLAR_COL_HEADER.search(line):
                    section.distances = _parse_distances(line)
                elif m := POLAR_ROW.match(line):
                    if section.distances:
                        section.records.extend(_parse_polar_row(m, line, section.distances))
            # All other lines (headings, separators, blank lines) are silently skipped.

    if section is not None:
        yield section


def _try_section_header(line: str) -> AermodSection | None:
    """Return a new AermodSection if the line is a recognised result section header."""
    if m := ANNUAL_AVG_HDR.search(line):
        return AermodSection(
            section_type='ANNUAL_AVERAGE',
            source_group=(m.group('source_group') or '').upper(),
        )
    if m := HIGHEST_HDR.search(line):
        rank = int(m.group('rank'))
        suffix = {1: 'ST', 2: 'ND', 3: 'RD'}.get(rank, 'TH')
        return AermodSection(
            section_type=f'{rank}{suffix}_HIGHEST',
            averaging_period=int(m.group('period')),
            rank=rank,
            source_group=(m.group('source_group') or '').upper(),
        )
    if m := CONCURRENT_HDR.search(line):
        return AermodSection(
            section_type='CONCURRENT_AVERAGE',
            averaging_period=int(m.group('period')),
            day=int(m.group('day')),
            year=int(m.group('year')),
        )
    return None


def _update_meta(section: AermodSection, line: str) -> None:
    """Extract metadata from lines that appear between the section header and the table."""
    if m := NETWORK_LINE.search(line):
        section.network_id = m.group('id')
        section.network_type = m.group('type').upper()
    if m := POLLUTANT_LINE.search(line):
        section.pollutant = m.group('pollutant')
        section.unit = m.group('unit').strip()
    if m := ORIGIN_LINE.search(line):
        section.origin_x = float(m.group('x'))
        section.origin_y = float(m.group('y'))
    # Concurrent section headers don't carry source group; pick it up from metadata lines.
    if not section.source_group:
        if m := SOURCE_GROUP_LINE.search(line):
            section.source_group = m.group('group').upper()


def _parse_distances(line: str) -> list[float]:
    """Extract distance ring values from a polar table column header row."""
    _, _, data = line.partition('|')
    return [float(m.group('value')) for m in DECIMAL_VALUE.finditer(data)]


def _parse_polar_row(match: re.Match, line: str, distances: list[float]) -> list[dict]:
    """Convert one polar data row to a list of per-distance concentration records."""
    direction = float(match.group('direction'))
    _, _, data = line.partition('|')
    values = [(float(m.group('value')), m.group('date')) for m in DECIMAL_VALUE.finditer(data)]
    return [
        {'direction': direction, 'distance': dist, 'concentration': conc, 'date': date}
        for (conc, date), dist in zip(values, distances)
    ]


def _parse_cart_row(match: re.Match, line: str, x_values: list[float]) -> list[dict]:
    """Convert one Cartesian data row to a list of per-x-coordinate concentration records."""
    y = float(match.group('y'))
    _, _, data = line.partition('|')
    values = [(float(m.group('value')), m.group('date')) for m in DECIMAL_VALUE.finditer(data)]
    return [
        {'x': x, 'y': y, 'concentration': conc, 'date': date}
        for (conc, date), x in zip(values, x_values)
    ]
