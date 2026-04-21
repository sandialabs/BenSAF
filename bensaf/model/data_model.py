"""
Data model for SAF health impact assessment.

- AnalysisConfig: Analysis configuration parameters
- AnalysisInputs: All geospatial input data (loaded once, treated as read-only after loading)
- AnalysisResults: Post-analysis container holding inputs + scenario results
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import logging

from bensaf.model.domain import ScenarioResult

logger = logging.getLogger(__name__)


def validate_geoid_alignment(
    df: pd.DataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    data_name: str = "data",
) -> pd.DataFrame:
    """
    Validate and align a DataFrame's GEOIDs against the tract GeoDataFrame.

    Returns the DataFrame reindexed to match tracts_gdf (int GEOID index).
    Raises ValueError if any tract GEOIDs are missing from df.
    Drops extra GEOIDs with a warning.
    """
    tract_geoids = set(tracts_gdf.index)

    if 'GEOID' in df.columns:
        df_geoids = set(df['GEOID'].astype(int))
    elif df.index.name == 'GEOID' or isinstance(df.index, pd.Index):
        df_geoids = set(df.index.astype(int))
    else:
        raise ValueError(f"{data_name} must have GEOID as column or index")

    missing = tract_geoids - df_geoids
    if missing:
        raise ValueError(
            f"{data_name} is missing GEOIDs: "
            f"{sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}"
        )

    extra = df_geoids - tract_geoids
    if extra:
        logger.warning(f"{data_name} has {len(extra)} extra GEOIDs, dropping them")
        if 'GEOID' in df.columns:
            df = df[df['GEOID'].isin(tract_geoids)].copy()
        else:
            df = df[df.index.isin(tract_geoids)].copy()

    if 'GEOID' in df.columns:
        df = df.set_index('GEOID')

    df.index = df.index.astype(int)
    df = df.reindex(tract_geoids)

    return df


@dataclass
class AnalysisConfig:
    """Configuration parameters for the analysis."""
    saf_scenarios: List[float] = field(default_factory=lambda: [5, 25, 50])
    airport_coordinates: Optional[tuple] = None
    crs: str = 'EPSG:4326'


class AnalysisInputs:
    """
    Input data container for a SAF health impact analysis.

    Holds all geospatial data required before scenarios are run. After all
    load_* methods have been called the object should be treated as read-only.

    Core columns (required for analysis):
    - tract_geometries: GeoDataFrame, GEOID (int) index, geometry column only
    - demographics_core: DataFrame, GEOID (int) index, 'population' column
    - baseline_exposure: DataFrame, GEOID (int) index, pollutant columns
    - incidence: DataFrame, GEOID (int) index, incidence rate columns
    - preterm_birth_core: DataFrame, GEOID (int) index, 'baseline_preterm_births' column
    - economic_core: DataFrame, GEOID (int) index, 'per_capita_consumption' for tract mortality valuation

    Optional preterm birth economics (no file): set via ``set_preterm_birth_economic_parameters``.

    Covariate columns (optional, for benefit-distribution analysis):
    - demographics_covariates: DataFrame, all demographic columns except 'population'
    - preterm_birth_covariates: DataFrame, all PTB columns except 'baseline_preterm_births'

    Derived inputs:
    - derived_inputs: DataFrame, computed metrics such as pct_low_income
    """

    def __init__(self, crs: str = 'EPSG:4326'):
        self.crs = crs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.tract_geometries: Optional[gpd.GeoDataFrame] = None
        self.demographics_core: Optional[pd.DataFrame] = None
        self.demographics_covariates: Optional[pd.DataFrame] = None
        self.baseline_exposure: Optional[pd.DataFrame] = None
        self.incidence: Optional[pd.DataFrame] = None
        self.preterm_birth_core: Optional[pd.DataFrame] = None
        self.preterm_birth_covariates: Optional[pd.DataFrame] = None
        self.economic_core: Optional[pd.DataFrame] = None
        self.economic_covariates: Optional[pd.DataFrame] = None
        self.derived_inputs: Optional[pd.DataFrame] = None

        self.preterm_birth_odds_ratio: Optional[float] = None
        self.monetary_value_per_ptb: Optional[float] = None

        # Cached merged views for backward compatibility
        self._demographics: Optional[pd.DataFrame] = None
        self._preterm_birth: Optional[pd.DataFrame] = None
        self._economic: Optional[pd.DataFrame] = None

    @property
    def demographics(self) -> Optional[pd.DataFrame]:
        if self._demographics is not None:
            return self._demographics
        if self.demographics_core is None:
            return None
        if self.demographics_covariates is not None:
            return pd.concat([self.demographics_core, self.demographics_covariates], axis=1)
        return self.demographics_core

    @property
    def preterm_birth(self) -> Optional[pd.DataFrame]:
        if self._preterm_birth is not None:
            return self._preterm_birth
        if self.preterm_birth_core is None:
            return None
        if self.preterm_birth_covariates is not None:
            return pd.concat([self.preterm_birth_core, self.preterm_birth_covariates], axis=1)
        return self.preterm_birth_core

    @property
    def economic(self) -> Optional[pd.DataFrame]:
        if self._economic is not None:
            return self._economic
        if self.economic_core is None:
            return None
        if self.economic_covariates is not None:
            return pd.concat([self.economic_core, self.economic_covariates], axis=1)
        return self.economic_core

    @property
    def is_ready(self) -> bool:
        return (
            self.tract_geometries is not None
            and self.baseline_exposure is not None
            and self.incidence is not None
        )

    def load_tract_geometries(self, tracts_gdf: gpd.GeoDataFrame) -> None:
        self.logger.info("Loading tract geometries")

        for col in ('GEOID', 'geometry'):
            if col not in tracts_gdf.columns:
                raise ValueError(f"Missing required column in tract data: {col}")

        tracts_gdf = tracts_gdf.copy()
        tracts_gdf['GEOID'] = tracts_gdf['GEOID'].astype(int)
        
        # Use CRS from file metadata; do not reproject to a separate dashboard CRS
        if tracts_gdf.crs is None:
            self.logger.warning("Tract data has no CRS in file metadata; assuming EPSG:4326")
            tracts_gdf.set_crs("EPSG:4326", inplace=True)
        self.crs = str(tracts_gdf.crs)
        
        # Set GEOID as index and keep only geometry
        self.tract_geometries = tracts_gdf[['GEOID', 'geometry']].set_index('GEOID')
        self.logger.info(f"Loaded {len(self.tract_geometries)} census tract geometries")

    def load_demographics(self, demographics_df: pd.DataFrame) -> None:
        self.logger.info("Loading demographic data")

        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")

        demographics_df = demographics_df.copy()
        demographics_df = validate_geoid_alignment(
            demographics_df, self.tract_geometries, "demographic data"
        )

        if 'Population' in demographics_df.columns and 'population' not in demographics_df.columns:
            demographics_df['population'] = demographics_df['Population']

        if 'population' not in demographics_df.columns:
            raise ValueError("Demographics data must contain 'population' column")

        core_columns = ['population']
        covariate_columns = [c for c in demographics_df.columns if c not in core_columns]

        self.demographics_core = demographics_df[core_columns].copy()
        self.demographics_covariates = (
            demographics_df[covariate_columns].copy() if covariate_columns else None
        )
        self._demographics = demographics_df

        self.logger.info(
            f"Loaded demographic data: {len(core_columns)} core column(s), "
            f"{len(covariate_columns)} covariate column(s)"
        )

    def load_baseline_exposure(
        self,
        exposure_df: pd.DataFrame,
        pollutant_columns: Optional[List[str]] = None,
    ) -> None:
        self.logger.info("Loading baseline exposure data")

        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")

        exposure_df = exposure_df.copy()
        exposure_df = validate_geoid_alignment(
            exposure_df, self.tract_geometries, "baseline exposure data"
        )

        if pollutant_columns is None:
            pollutant_columns = exposure_df.select_dtypes(include=[float, int]).columns.tolist()

        missing_cols = [c for c in pollutant_columns if c not in exposure_df.columns]
        if missing_cols:
            raise ValueError(f"Missing pollutant columns: {missing_cols}")

        self.baseline_exposure = exposure_df[pollutant_columns].copy()
        self.logger.info(
            f"Loaded baseline exposure data with {len(pollutant_columns)} pollutants: {pollutant_columns}"
        )

    def load_incidence_data(
        self,
        incidence_df: pd.DataFrame,
        endpoint_columns: Optional[List[str]] = None,
    ) -> None:
        self.logger.info("Loading incidence data")

        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")

        incidence_df = incidence_df.copy()
        incidence_df = validate_geoid_alignment(
            incidence_df, self.tract_geometries, "incidence data"
        )

        if endpoint_columns is None:
            endpoint_columns = incidence_df.select_dtypes(include=[float, int]).columns.tolist()

        missing_cols = [c for c in endpoint_columns if c not in incidence_df.columns]
        if missing_cols:
            raise ValueError(f"Missing endpoint columns: {missing_cols}")

        self.incidence = incidence_df[endpoint_columns].copy()
        self.logger.info(
            f"Loaded incidence data with {len(endpoint_columns)} endpoints: {endpoint_columns}"
        )

    def load_preterm_birth_data(self, preterm_birth_df: pd.DataFrame) -> None:
        self.logger.info("Loading preterm birth data")

        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")

        preterm_birth_df = preterm_birth_df.copy()

        if 'baseline_preterm_births' not in preterm_birth_df.columns:
            if 'preterm_births' in preterm_birth_df.columns:
                preterm_birth_df['baseline_preterm_births'] = preterm_birth_df['preterm_births']
            else:
                raise ValueError(
                    "Missing required column 'baseline_preterm_births' or 'preterm_births'"
                )

        preterm_birth_df = validate_geoid_alignment(
            preterm_birth_df, self.tract_geometries, "preterm birth data"
        )

        core_columns = ['baseline_preterm_births']
        covariate_columns = [c for c in preterm_birth_df.columns if c not in core_columns]

        self.preterm_birth_core = preterm_birth_df[core_columns].copy()
        self.preterm_birth_covariates = (
            preterm_birth_df[covariate_columns].copy() if covariate_columns else None
        )
        self._preterm_birth = preterm_birth_df

        self.logger.info(
            f"Loaded preterm birth data: {len(core_columns)} core column(s), "
            f"{len(covariate_columns)} covariate column(s)"
        )

    def set_preterm_birth_economic_parameters(
        self, odds_ratio: float, monetary_value_per_ptb: float
    ) -> None:
        """Set odds ratio and dollar value per preterm birth for the PTB economic pipeline."""
        self.preterm_birth_odds_ratio = odds_ratio
        self.monetary_value_per_ptb = monetary_value_per_ptb

    def load_mortality_economic_tract_data(self, economic_df: pd.DataFrame) -> None:
        """
        Load tract-level inputs for mortality economic valuation.

        Required column: ``per_capita_consumption``. Optional: ``life_years_gained`` per tract;
        if omitted, a default of 10.0 life years per tract is used when computing benefits.
        """
        self.logger.info("Loading tract-level mortality economic data")

        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")

        economic_df = economic_df.copy()

        if 'per_capita_consumption' not in economic_df.columns:
            raise ValueError("Economic tract data must contain 'per_capita_consumption'")

        economic_df = validate_geoid_alignment(
            economic_df, self.tract_geometries, "mortality economic tract data"
        )

        core_columns = ['per_capita_consumption']
        if 'life_years_gained' in economic_df.columns:
            core_columns.append('life_years_gained')

        covariate_columns = [c for c in economic_df.columns if c not in core_columns]

        self.economic_core = economic_df[core_columns].copy()
        self.economic_covariates = (
            economic_df[covariate_columns].copy() if covariate_columns else None
        )
        self._economic = economic_df
        self.logger.info(
            f"Loaded mortality economic data: {len(core_columns)} core column(s), "
            f"{len(covariate_columns)} other column(s)"
        )

    def compute_derived_inputs(self) -> None:
        """
        Compute derived input metrics from loaded demographic data.

        This must be called explicitly after load_demographics if derived metrics
        such as pct_low_income are needed. It is not called automatically.
        """
        demographics_merged = self.demographics
        if demographics_merged is None:
            return

        self.logger.info("Computing derived inputs")
        derived = {}

        if (
            'households_below_poverty' in demographics_merged.columns
            and 'total_households' in demographics_merged.columns
        ):
            derived['pct_low_income'] = (
                demographics_merged['households_below_poverty']
                / demographics_merged['total_households']
                * 100
            )
        elif 'poverty_rate' in demographics_merged.columns:
            derived['pct_low_income'] = demographics_merged['poverty_rate'] * 100

        if derived:
            if self.derived_inputs is None:
                self.derived_inputs = pd.DataFrame(index=demographics_merged.index)
            for name, values in derived.items():
                self.derived_inputs[name] = values
                self.logger.info(f"Computed derived input: {name}")

    def validate(self) -> bool:
        if self.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded")
        if self.baseline_exposure is None:
            raise ValueError("Baseline exposure data must be loaded")
        if self.incidence is None:
            raise ValueError("Incidence data must be loaded")
        return True


class AnalysisResults:
    """
    Post-analysis container: combines inputs with computed scenario results.

    Scenarios are added as they are computed. Provides merged GeoDataFrames
    for downstream visualisation and export.
    """

    def __init__(self, inputs: AnalysisInputs):
        self.inputs = inputs
        self.scenarios: Dict[int, ScenarioResult] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_scenario(self, result: ScenarioResult) -> None:
        if self.inputs.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded first")

        tract_index = self.inputs.tract_geometries.index
        result_index = result.delta_concentration.index
        if not result_index.equals(tract_index):
            raise ValueError(
                f"ScenarioResult index does not match tract geometries index.\n"
                f"  Result index sample: {list(result_index[:5])}\n"
                f"  Tract index sample:  {list(tract_index[:5])}"
            )

        self.scenarios[result.spec.scenario_id] = result
        self.logger.info(
            f"Added scenario result: {result.spec.scenario_label} (ID: {result.spec.scenario_id})"
        )

    def get_scenario(self, scenario_id: int) -> Optional[ScenarioResult]:
        return self.scenarios.get(scenario_id)

    def get_merged_data(self, core_only: bool = False) -> gpd.GeoDataFrame:
        """
        Return a GeoDataFrame with tract geometries joined to all input and result data.

        Args:
            core_only: If True, omit covariate columns from demographics and preterm birth.
        """
        inputs = self.inputs
        if inputs.tract_geometries is None:
            raise ValueError("Tract geometries must be loaded")

        merged = inputs.tract_geometries.copy()

        if inputs.demographics_core is not None:
            merged = merged.join(inputs.demographics_core, how='left')
        if not core_only and inputs.demographics_covariates is not None:
            merged = merged.join(inputs.demographics_covariates, how='left')
        if inputs.baseline_exposure is not None:
            merged = merged.join(inputs.baseline_exposure, how='left')
        if inputs.incidence is not None:
            merged = merged.join(inputs.incidence, how='left')
        if inputs.preterm_birth_core is not None:
            merged = merged.join(inputs.preterm_birth_core, how='left')
        if not core_only and inputs.preterm_birth_covariates is not None:
            merged = merged.join(inputs.preterm_birth_covariates, how='left')
        if inputs.derived_inputs is not None:
            merged = merged.join(inputs.derived_inputs, how='left')
        if inputs.economic_core is not None:
            merged = merged.join(inputs.economic_core, how='left')
        if not core_only and inputs.economic_covariates is not None:
            merged = merged.join(inputs.economic_covariates, how='left')

        for result in self.scenarios.values():
            scenario_df = result.to_dataframe()
            merged = merged.join(
                scenario_df.add_prefix(f"{result.spec.scenario_label}_"), how='left'
            )

        return merged
