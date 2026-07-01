from dataclasses import dataclass, field
import pandas as pd


@dataclass
class AermodSection:
    """One result block from an AERMOD output file.

    A single physical page of output maps to one section. Tables that span
    multiple pages produce multiple sections with identical metadata; callers
    can concatenate their DataFrames to recover the full table.
    """

    section_type: str           # ANNUAL_AVERAGE | N[ST|ND|RD|TH]_HIGHEST | CONCURRENT_AVERAGE
    network_type: str = ''      # GRIDPOLR | GRIDCART
    source_group: str = ''
    network_id: str = ''
    pollutant: str = ''
    unit: str = ''
    averaging_period: int | None = None
    rank: int | None = None     # ordinal for N-highest sections (1, 2, 3 …)
    day: int | None = None      # CONCURRENT sections only
    year: int | None = None     # CONCURRENT sections only
    origin_x: float | None = None
    origin_y: float | None = None
    distances: list[float] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return records as a tidy DataFrame with section metadata prepended as columns."""
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        df.insert(0, 'section_type', self.section_type)
        df.insert(1, 'source_group', self.source_group)
        df.insert(2, 'network_id', self.network_id)
        df.insert(3, 'pollutant', self.pollutant)
        df.insert(4, 'unit', self.unit)
        df.insert(5, 'averaging_period', self.averaging_period)
        df.insert(6, 'rank', self.rank)
        df.insert(7, 'day', self.day)
        df.insert(8, 'year', self.year)
        return df
