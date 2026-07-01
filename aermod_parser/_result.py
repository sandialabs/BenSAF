from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from ._machine import parse_file
from ._metadata import parse_metadata, FileMetadata
from ._sections import AermodSection


@dataclass
class AermodFile:
    """Structured representation of a parsed AERMOD output file.

    Combines metadata (pathways, network origins) with concentration results.
    Tables split across multiple output pages are merged transparently.

    Typical usage::

        f = AermodFile.from_path("takeoff_eastflow.ADO")
        f.annual_average              # merged DataFrame, all directions × distances
        f.n_highest(rank=1, period=1) # 1st highest 1-hr DataFrame
        f.concurrent                  # all per-day averages with day/year columns
        f.metadata.networks           # origin coordinates per network ID
    """

    metadata: FileMetadata
    sections: list[AermodSection]

    @classmethod
    def from_path(cls, path: str | Path) -> 'AermodFile':
        """Parse an AERMOD .ADO, .OUT, or .ADI file."""
        path = Path(path)
        return cls(
            metadata=parse_metadata(path),
            sections=parse_file(path),
        )

    # ── Section inventory ────────────────────────────────────────────────────

    @property
    def section_types(self) -> list[str]:
        """Distinct result section types present in this file."""
        return sorted(set(s.section_type for s in self.sections))

    # ── Result accessors ─────────────────────────────────────────────────────

    @property
    def annual_average(self) -> pd.DataFrame:
        """Annual average concentrations as a single merged DataFrame."""
        return self._merge('ANNUAL_AVERAGE')

    def n_highest(self, rank: int = 1, period: int | None = None) -> pd.DataFrame:
        """N-th highest concentrations.

        Args:
            rank:   Ordinal rank (1 for 1st highest, 2 for 2nd, …).
            period: Averaging period in hours (1, 24, …).
                    Pass None to return all periods for this rank.
        """
        suffix = {1: 'ST', 2: 'ND', 3: 'RD'}.get(rank, 'TH')
        return self._merge(f'{rank}{suffix}_HIGHEST', averaging_period=period)

    @property
    def concurrent(self) -> pd.DataFrame:
        """All concurrent average data with day and year columns."""
        return self._merge('CONCURRENT_AVERAGE')

    def tables(self) -> dict[str, pd.DataFrame]:
        """Every section type as a merged DataFrame, keyed by section_type string."""
        return {st: self._merge(st) for st in self.section_types}

    # ── Export ───────────────────────────────────────────────────────────────

    def export(self, directory: str | Path, fmt: str = 'csv') -> list[Path]:
        """Write all tables and metadata to files.

        Args:
            directory: Output directory; created if it does not exist.
            fmt:       'csv'     — one file per table (default)
                       'parquet' — one file per table, columnar format
                       'excel'   — single .xlsx with one sheet per table
                                   (requires openpyxl: pip install openpyxl)

        Returns:
            Sorted list of paths that were written.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        if fmt == 'excel':
            return self._export_excel(out)

        if fmt not in ('csv', 'parquet'):
            raise ValueError(f"Unknown format {fmt!r}. Choose 'csv', 'parquet', or 'excel'.")

        written: list[Path] = []

        for section_type, df in self.tables().items():
            path = out / f"{section_type.lower()}.{fmt}"
            df.to_csv(path, index=False) if fmt == 'csv' else df.to_parquet(path, index=False)
            written.append(path)

        if not self.metadata.sources.empty:
            path = out / f"sources.{fmt}"
            self.metadata.sources.to_csv(path, index=False) if fmt == 'csv' else self.metadata.sources.to_parquet(path, index=False)
            written.append(path)

        path = out / 'metadata.json'
        path.write_text(json.dumps(self._metadata_dict(), indent=2))
        written.append(path)

        return sorted(written)

    def _export_excel(self, out: Path) -> list[Path]:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError("Excel export requires openpyxl: pip install openpyxl") from None

        path = out / 'aermod_export.xlsx'
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for section_type, df in self.tables().items():
                df.to_excel(writer, sheet_name=section_type.lower()[:31], index=False)
            if not self.metadata.sources.empty:
                self.metadata.sources.to_excel(writer, sheet_name='sources', index=False)
            # Metadata as a two-column key/value sheet.
            meta_rows = [(k, str(v)) for k, v in self._metadata_dict().items()]
            pd.DataFrame(meta_rows, columns=['key', 'value']).to_excel(
                writer, sheet_name='metadata', index=False
            )
        return [path]

    def _metadata_dict(self) -> dict:
        return {
            'pollutant':               self.metadata.pollutant,
            'averaging_periods':       self.metadata.averaging_periods,
            'model_options':           self.metadata.model_options,
            'surface_met_file':        self.metadata.surface_met_file,
            'profile_met_file':        self.metadata.profile_met_file,
            'included_receptor_files': self.metadata.included_receptor_files,
            'networks': {
                nid: {
                    'network_type': n.network_type,
                    'origin_x':     n.origin_x,
                    'origin_y':     n.origin_y,
                }
                for nid, n in self.metadata.networks.items()
            },
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _merge(self, section_type: str, **filters) -> pd.DataFrame:
        """Merge all sections of the given type, applying optional field filters."""
        matching = [s for s in self.sections if s.section_type == section_type]
        for key, val in filters.items():
            if val is not None:
                matching = [s for s in matching if getattr(s, key, None) == val]
        if not matching:
            return pd.DataFrame()
        return pd.concat([s.to_dataframe() for s in matching], ignore_index=True)
