from dataclasses import dataclass, field
from pathlib import Path
import re
import pandas as pd

from ._patterns import NETWORK_LINE, ORIGIN_LINE, PAGE_BREAK


@dataclass
class NetworkInfo:
    """Origin and type for one receptor network, sourced from RECEPTOR_NETWORK_SUMMARY."""
    network_id: str
    network_type: str       # GRIDPOLR | GRIDCART
    origin_x: float | None = None
    origin_y: float | None = None


@dataclass
class FileMetadata:
    """Non-result metadata extracted from an AERMOD file.

    Populated from the input pathways (CO/SO/RE/ME) and the receptor network
    summary section.  Does not include concentration table data — use
    parse_file() for that.
    """
    pollutant: str = ''
    averaging_periods: list[str] = field(default_factory=list)
    model_options: str = ''
    sources: pd.DataFrame = field(default_factory=pd.DataFrame)
    included_receptor_files: list[str] = field(default_factory=list)
    surface_met_file: str | None = None
    profile_met_file: str | None = None
    networks: dict[str, NetworkInfo] = field(default_factory=dict)


_PATHWAY_START = re.compile(r'^\s*(CO|SO|RE|ME|OU)\s+STARTING\s*$')
_PATHWAY_END   = re.compile(r'^\s*(CO|SO|RE|ME|OU)\s+FINISHED\s*$')
_NETWORK_SUMMARY_HDR = re.compile(r'\*{3}.*?GRIDDED\s+RECEPTOR\s+NETWORK\s+SUMMARY', re.IGNORECASE)


def parse_metadata(path: str | Path) -> FileMetadata:
    """Extract non-result metadata from an AERMOD .ADO, .OUT, or .ADI file."""
    lines = Path(path).read_text(errors='replace').splitlines()
    meta = FileMetadata()

    current_pathway: str | None = None
    pathway_start: int | None = None

    for i, line in enumerate(lines):
        if m := _PATHWAY_START.match(line):
            current_pathway = m.group(1)
            pathway_start = i
            continue

        if m := _PATHWAY_END.match(line):
            if current_pathway is not None and pathway_start is not None:
                _dispatch_pathway(meta, current_pathway, lines[pathway_start : i + 1])
            current_pathway = None
            pathway_start = None
            continue

        if _NETWORK_SUMMARY_HDR.search(line):
            _parse_network_summary(meta, lines, i)

    return meta


def _dispatch_pathway(meta: FileMetadata, pathway: str, lines: list[str]) -> None:
    dispatch = {'CO': _parse_co, 'SO': _parse_so, 'RE': _parse_re, 'ME': _parse_me}
    if fn := dispatch.get(pathway):
        fn(meta, lines)


def _parse_co(meta: FileMetadata, lines: list[str]) -> None:
    for line in lines:
        s = line.strip()
        if s.startswith('POLLUTID'):
            parts = s.split()
            if len(parts) > 1:
                meta.pollutant = parts[1]
        elif s.startswith('AVERTIME'):
            meta.averaging_periods = s.split()[1:]
        elif s.startswith('MODELOPT'):
            meta.model_options = s.replace('MODELOPT', '', 1).strip()


def _parse_so(meta: FileMetadata, lines: list[str]) -> None:
    sources = []
    current_group = ''

    for line in lines:
        s = line.strip()
        if s.startswith('LOCATION'):
            parts = s.split()
            if len(parts) >= 5:
                try:
                    sources.append({
                        'source_id':    parts[1],
                        'source_type':  parts[2],
                        'x':            float(parts[3]),
                        'y':            float(parts[4]),
                        'z':            float(parts[5]) if len(parts) > 5 else None,
                        'source_group': current_group,
                    })
                except ValueError:
                    pass
        elif s.startswith('SRCGROUP'):
            parts = s.split()
            if len(parts) > 1:
                current_group = parts[1]

    meta.sources = pd.DataFrame(sources) if sources else pd.DataFrame()


def _parse_re(meta: FileMetadata, lines: list[str]) -> None:
    for line in lines:
        s = line.strip()
        # INCLUDED references an external receptor file (.rou).
        if s.upper().startswith('INCLUDED'):
            parts = s.split(None, 1)
            if len(parts) > 1:
                meta.included_receptor_files.append(parts[1].strip())


def _parse_me(meta: FileMetadata, lines: list[str]) -> None:
    for line in lines:
        s = line.strip()
        if s.startswith('SURFFILE'):
            parts = s.split(None, 1)
            if len(parts) > 1:
                meta.surface_met_file = parts[1].strip()
        elif s.startswith('PROFFILE'):
            parts = s.split(None, 1)
            if len(parts) > 1:
                meta.profile_met_file = parts[1].strip()


def _parse_network_summary(meta: FileMetadata, lines: list[str], start: int) -> None:
    """Scan the RECEPTOR_NETWORK_SUMMARY section for network origin coordinates."""
    current_id: str | None = None

    for line in lines[start:]:
        if PAGE_BREAK.search(line):
            break

        if m := NETWORK_LINE.search(line):
            current_id = m.group('id')
            ntype = m.group('type').upper()
            if current_id not in meta.networks:
                meta.networks[current_id] = NetworkInfo(
                    network_id=current_id,
                    network_type=ntype,
                )
            else:
                meta.networks[current_id].network_type = ntype

        if current_id:
            if m := ORIGIN_LINE.search(line):
                meta.networks[current_id].origin_x = float(m.group('x'))
                meta.networks[current_id].origin_y = float(m.group('y'))
