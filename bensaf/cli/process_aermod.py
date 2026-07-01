"""
Interactive CLI for processing AERMOD .ADO files.

Two workflows are available:
  - Parse & export: extract section data to CSV/JSON for inspection
  - Generate baseline surface: run the full landing/takeoff → calibrate → aggregate pipeline,
    producing a GEOID,ufp CSV ready for upload to the BenSAF dashboard

Run with:
    bensaf-process-aermod <folder>
    python scripts/process_aermod.py <folder>
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import questionary
from questionary import Style

logger = logging.getLogger(__name__)

# Minimal style: keep the prompts readable without extra visual noise
CLI_STYLE = Style(
    [
        ("qmark", "fg:#5f9ea0 bold"),
        ("question", "bold"),
        ("answer", "fg:#5f9ea0 bold"),
        ("pointer", "fg:#5f9ea0 bold"),
        ("highlighted", "fg:#5f9ea0 bold"),
        ("selected", "fg:#5f9ea0"),
        ("separator", "fg:#5f9ea0"),
        ("instruction", "fg:#888888"),
    ]
)

SECTION_LABELS = {
    "ANNUAL_AVERAGE": "Annual Average",
    "1ST_HIGHEST": "1st Highest",
    "2ND_HIGHEST": "2nd Highest",
    "3RD_HIGHEST": "3rd Highest",
    "CONCURRENT_AVERAGE": "Concurrent Average",
    "SOURCE_PATHWAY": "Source Pathway",
    "RECEPTOR_PATHWAY": "Receptor Pathway",
    "CONTROL_PATHWAY": "Control Pathway",
    "METEOROLOGY_PATHWAY": "Meteorology Pathway",
    "MODEL_SETUP_SUMMARY": "Model Setup Summary",
    "SOURCE_GROUPS_SUMMARY": "Source Groups Summary",
    "RECEPTOR_NETWORK_SUMMARY": "Receptor Network Summary",
}

# Default CRS for AERMOD coordinate data (UTM Zone 16N, common for US Midwest airports)
DEFAULT_AERMOD_CRS = "EPSG:32616"

# Calibration file path relative to the project root
DEFAULT_CALIBRATION_REL = Path("data") / "aermod_calibration_coefficients.json"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler])


def _project_root() -> Path:
    """Return the project root (two levels above bensaf/cli/)."""
    return Path(__file__).parent.parent.parent


def _default_calibration_path() -> Path | None:
    path = _project_root() / DEFAULT_CALIBRATION_REL
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_directory(folder: Path) -> list[Path]:
    """Return all .ADO and .OUT files in the given folder (non-recursive)."""
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.upper() in (".ADO", ".OUT")
    )
    return files


def _file_label(path: Path) -> str:
    size_kb = path.stat().st_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.0f} KB"
    return f"{path.name}  ({size_str})"


# ---------------------------------------------------------------------------
# CRS sanity check
# ---------------------------------------------------------------------------

def _crs_looks_geographic(crs_str: str) -> bool:
    """Return True if the CRS string appears to be geographic (lat/lon)."""
    try:
        from pyproj import CRS
        return CRS(crs_str).is_geographic
    except Exception:
        return "4326" in crs_str


def _coords_look_like_utm(gdf: gpd.GeoDataFrame) -> bool:
    """Return True if x/y values look like projected meters (UTM range)."""
    xs = [g.x for g in gdf.geometry if g is not None]
    if not xs:
        return False
    return any(abs(x) > 180 for x in xs[:20])


def _suggest_crs(gdf: gpd.GeoDataFrame, declared_crs: str) -> str | None:
    """
    Return a warning string if the declared CRS seems inconsistent with the
    coordinate magnitudes, otherwise None.
    """
    if _crs_looks_geographic(declared_crs) and _coords_look_like_utm(gdf):
        return (
            f"  Warning: CRS is set to {declared_crs} (geographic) but coordinates "
            f"look like projected meters. Did you mean EPSG:32616 or similar?"
        )
    return None


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def _config_path(folder: Path) -> Path:
    return folder / "aermod_config.json"


def save_config(config: dict, folder: Path) -> None:
    path = _config_path(folder)
    # Convert Path objects to strings for JSON serialisation
    serialisable = json.loads(
        json.dumps(config, default=lambda o: str(o) if isinstance(o, Path) else o)
    )
    path.write_text(json.dumps(serialisable, indent=2))
    print(f"\n  Settings saved to {path}")


def load_config(folder: Path) -> dict | None:
    path = _config_path(folder)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Workflow: Parse & export
# ---------------------------------------------------------------------------

def _prompt_parse_export_config() -> dict:
    section_choices = [
        questionary.Choice(label, value=key, checked=(key == "ANNUAL_AVERAGE"))
        for key, label in SECTION_LABELS.items()
    ]
    section_types = questionary.checkbox(
        "Which section types would you like to extract?",
        choices=section_choices,
        style=CLI_STYLE,
    ).ask()

    if not section_types:
        print("No section types selected. Exiting.")
        sys.exit(0)

    fmt = questionary.select(
        "Output format:",
        choices=["CSV", "JSON", "Both"],
        style=CLI_STYLE,
    ).ask()

    out_dir = questionary.text(
        "Output folder:",
        default="aermod_parsed",
        style=CLI_STYLE,
    ).ask()

    return {"section_types": section_types, "format": fmt, "output_dir": Path(out_dir)}


def _metadata_section_df(metadata, section_type: str) -> pd.DataFrame:
    """Build a DataFrame for non-result (pathway/summary) section types from FileMetadata.

    The new aermod_parser package doesn't preserve these as raw text sections the way
    the old AermodParser did; this reconstructs an equivalent table from the structured
    metadata it does extract.
    """
    if section_type == "SOURCE_PATHWAY":
        return metadata.sources
    if section_type in ("CONTROL_PATHWAY", "MODEL_SETUP_SUMMARY"):
        return pd.DataFrame([{
            "pollutant": metadata.pollutant,
            "averaging_periods": ", ".join(metadata.averaging_periods),
            "model_options": metadata.model_options,
        }])
    if section_type in ("RECEPTOR_PATHWAY", "RECEPTOR_NETWORK_SUMMARY"):
        rows = [
            {"network_id": nid, "network_type": n.network_type, "origin_x": n.origin_x, "origin_y": n.origin_y}
            for nid, n in metadata.networks.items()
        ]
        rows += [{"included_receptor_file": f} for f in metadata.included_receptor_files]
        return pd.DataFrame(rows)
    if section_type == "METEOROLOGY_PATHWAY":
        return pd.DataFrame([{
            "surface_met_file": metadata.surface_met_file,
            "profile_met_file": metadata.profile_met_file,
        }])
    if section_type == "SOURCE_GROUPS_SUMMARY":
        if metadata.sources.empty or "source_group" not in metadata.sources.columns:
            return pd.DataFrame()
        return pd.DataFrame({"source_group": metadata.sources["source_group"].unique()})
    return pd.DataFrame()


def run_parse_export(files: list[Path], config: dict) -> None:
    from aermod_parser import AermodFile

    out_dir: Path = config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    section_types: list[str] = config["section_types"]
    fmt: str = config["format"]

    # Accumulate DataFrames per section type across all files
    section_frames: dict[str, list[pd.DataFrame]] = {st: [] for st in section_types}
    summary_rows = []

    result_types = {"ANNUAL_AVERAGE", "1ST_HIGHEST", "2ND_HIGHEST", "3RD_HIGHEST", "CONCURRENT_AVERAGE"}

    for path in files:
        print(f"\n  Parsing {path.name} ...")
        f = AermodFile.from_path(path)

        for st in section_types:
            if st in result_types:
                if st == "ANNUAL_AVERAGE":
                    data = f.annual_average
                elif st == "CONCURRENT_AVERAGE":
                    data = f.concurrent
                else:
                    data = f.n_highest(rank=int(st[0]))
            else:
                data = _metadata_section_df(f.metadata, st)

            if data is not None and len(data) > 0:
                data = data.copy()
                data["source_file"] = path.name
                section_frames[st].append(data)
                summary_rows.append({"file": path.name, "section": st, "rows": len(data)})

    # Write outputs
    wrote_any = False
    for st, frames in section_frames.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        label = st.lower()

        if fmt in ("CSV", "Both"):
            csv_path = out_dir / f"{label}.csv"
            combined.to_csv(csv_path, index=False)
            print(f"  Wrote {csv_path}  ({len(combined)} rows)")
            wrote_any = True

        if fmt in ("JSON", "Both"):
            json_path = out_dir / f"{label}.json"
            combined.to_json(json_path, orient="records", indent=2)
            print(f"  Wrote {json_path}  ({len(combined)} rows)")
            wrote_any = True

    # Always write a summary of what was found
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "sections_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Wrote {summary_path}")

    if not wrote_any:
        print("\n  No data found for the selected section types in these files.")
    else:
        print(f"\n  Done. Output written to {out_dir}/")


# ---------------------------------------------------------------------------
# Workflow: Generate baseline surface
# ---------------------------------------------------------------------------

def _prompt_surface_config(files: list[Path], folder: Path) -> dict:
    # Assign roles
    role_map: dict[str, str] = {}
    print("\n  Assign a role to each file (landing, takeoff, or skip).\n")
    for path in files:
        role = questionary.select(
            f"  {path.name}",
            choices=["landing", "takeoff", "skip"],
            style=CLI_STYLE,
        ).ask()
        role_map[str(path)] = role

    landing_paths = [Path(p) for p, r in role_map.items() if r == "landing"]
    takeoff_paths = [Path(p) for p, r in role_map.items() if r == "takeoff"]

    if not landing_paths and not takeoff_paths:
        print("No files assigned as landing or takeoff. Exiting.")
        sys.exit(0)

    # Weights
    landing_weights = _prompt_weights(landing_paths, "landing")
    takeoff_weights = _prompt_weights(takeoff_paths, "takeoff")

    # CRS
    crs = questionary.text(
        "AERMOD coordinate system (CRS):",
        default=DEFAULT_AERMOD_CRS,
        instruction="(e.g. EPSG:32616 for UTM Zone 16N)",
        style=CLI_STYLE,
    ).ask()

    # Aggregation method
    aggregation = questionary.select(
        "Aggregation method:",
        choices=[
            questionary.Choice("Spatial join (recommended for most cases)", value="spatial_join"),
            questionary.Choice("IDW interpolation", value="idw_interpolation"),
            questionary.Choice("Polar (nearest/mean rule)", value="polar"),
        ],
        style=CLI_STYLE,
    ).ask()

    # Tracts file
    tracts_path = questionary.path(
        "Census tracts file (GeoJSON, shapefile, or GeoPackage):",
        style=CLI_STYLE,
    ).ask()

    # Calibration file
    default_cal = str(_default_calibration_path() or "")
    calibration_path = questionary.path(
        "Calibration coefficients file (JSON):",
        default=default_cal,
        style=CLI_STYLE,
    ).ask()

    # Output path
    output_path = questionary.text(
        "Output CSV path:",
        default="aermod_exposure.csv",
        style=CLI_STYLE,
    ).ask()

    return {
        "landing": [(p, w) for p, w in zip(landing_paths, landing_weights)],
        "takeoff": [(p, w) for p, w in zip(takeoff_paths, takeoff_weights)],
        "crs": crs,
        "aggregation": aggregation,
        "tracts_path": tracts_path,
        "calibration_path": calibration_path,
        "output_path": output_path,
    }


def _prompt_weights(paths: list[Path], role: str) -> list[float]:
    """Prompt for a weight for each file. Defaults to equal weighting."""
    if not paths:
        return []
    n = len(paths)
    default_w = round(1.0 / n, 4)
    weights = []
    if n > 1:
        print(f"\n  Set weights for {role} files (default: equal weighting).\n")
    for path in paths:
        raw = questionary.text(
            f"  Weight for {path.name}:",
            default=str(default_w),
            style=CLI_STYLE,
        ).ask()
        try:
            weights.append(float(raw))
        except ValueError:
            print(f"  Could not parse '{raw}' as a number; using {default_w}.")
            weights.append(default_w)

    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        normalised = [w / total for w in weights]
        print(f"  Weights sum to {total:.4f}; normalising to 1.0.")
        return normalised
    return weights


def run_generate_surface(config: dict) -> None:
    from bensaf.core.exposure_generation import generate_exposure_from_aermod

    tracts_gdf = gpd.read_file(config["tracts_path"])
    print(f"\n  Loaded {len(tracts_gdf)} census tracts from {config['tracts_path']}")

    landing = config["landing"] or None
    takeoff = config["takeoff"] or None

    if landing:
        print(f"  Landing files:  {[p.name for p, _ in landing]}")
    if takeoff:
        print(f"  Takeoff files:  {[p.name for p, _ in takeoff]}")
    print(f"  CRS: {config['crs']}  |  Aggregation: {config['aggregation']}")
    print(f"  Calibration: {config['calibration_path']}")
    print()

    exposure_df = generate_exposure_from_aermod(
        landing_files=landing,
        takeoff_files=takeoff,
        tracts_gdf=tracts_gdf,
        calibration_file=config["calibration_path"],
        aermod_crs=config["crs"],
        aggregation_method=config["aggregation"],
    )

    out_path = Path(config["output_path"])
    exposure_df.to_csv(out_path, index=False)

    ufp_col = exposure_df["ufp"] if "ufp" in exposure_df.columns else exposure_df.iloc[:, 1]
    print(f"  Wrote {out_path}  ({len(exposure_df)} tracts)")
    print(f"  UFP range: {ufp_col.min():.4f} – {ufp_col.max():.4f}")
    print(
        "\n  The output CSV (GEOID, ufp) can be uploaded directly to the BenSAF dashboard "
        "via the CSV exposure route."
    )


# ---------------------------------------------------------------------------
# Workflow: Diagnose
# ---------------------------------------------------------------------------

def run_diagnose(files: list[Path], crs: str) -> None:
    from aermod_parser import AermodFile
    from bensaf.core.exposure_generation import extract_annual_average

    print(f"\n  Running diagnostic on {len(files)} file(s) with CRS {crs}.\n")

    all_clear = True

    for path in files:
        print(f"  {'─' * 60}")
        print(f"  {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

        # Section inventory
        try:
            f = AermodFile.from_path(path)
        except Exception as e:
            print(f"    Could not read file: {e}")
            all_clear = False
            continue

        counts: dict[str, int] = {st: len([s for s in f.sections if s.section_type == st]) for st in f.section_types}
        if not f.metadata.sources.empty:
            counts["SOURCE_PATHWAY"] = 1
        if f.metadata.networks:
            counts["RECEPTOR_NETWORK_SUMMARY"] = len(f.metadata.networks)

        if not counts:
            print("    No sections detected.")
            all_clear = False
            continue

        print("    Sections found:")
        for stype, count in counts.items():
            label = SECTION_LABELS.get(stype, stype)
            print(f"      {label}: {count}")

        if "ANNUAL_AVERAGE" not in counts:
            print("    Warning: no ANNUAL_AVERAGE section found — this file cannot be used for surface generation.")
            all_clear = False

        # Coordinate probe on ANNUAL_AVERAGE
        try:
            gdf = extract_annual_average(path, aermod_crs=crs)
        except Exception as e:
            print(f"    Could not extract annual average: {e}")
            all_clear = False
            continue

        if gdf is None or len(gdf) == 0:
            print("    No receptor coordinates found in ANNUAL_AVERAGE.")
            all_clear = False
            continue

        xs = [g.x for g in gdf.geometry]
        ys = [g.y for g in gdf.geometry]
        print(f"    Receptors: {len(gdf)}")
        print(f"    X range:   {min(xs):.1f} – {max(xs):.1f}")
        print(f"    Y range:   {min(ys):.1f} – {max(ys):.1f}")

        warning = _suggest_crs(gdf, crs)
        if warning:
            print(warning)
            all_clear = False

    print(f"\n  {'─' * 60}")
    if all_clear:
        print("  All files look ready to process.")
    else:
        print("  One or more issues were found — review the warnings above before running.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bensaf-process-aermod",
        description="Interactive tool for processing AERMOD .ADO files.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .ADO or .OUT files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Load saved settings from a JSON file and skip interactive prompts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed log output.",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    folder: Path = args.folder.resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    # Scan for files
    files = scan_directory(folder)
    if not files:
        print(f"No .ADO or .OUT files found in {folder}.")
        sys.exit(0)

    print(f"\n  Found {len(files)} file(s) in {folder}\n")

    # Offer to load a saved config
    saved_config = None
    if args.config:
        saved_config = load_config(args.config.parent) if args.config.is_file() else None
    else:
        existing = load_config(folder)
        if existing:
            use_saved = questionary.confirm(
                "A saved config was found for this folder. Load it?",
                default=True,
                style=CLI_STYLE,
            ).ask()
            if use_saved:
                saved_config = existing

    # File selection (skipped if replaying a saved config)
    if saved_config and "selected_files" in saved_config:
        selected = [folder / name for name in saved_config["selected_files"]]
        selected = [p for p in selected if p.exists()]
        print(f"  Using saved file selection: {[p.name for p in selected]}")
    else:
        choices = [
            questionary.Choice(_file_label(f), value=f)
            for f in files
        ]
        selected = questionary.checkbox(
            "Select files to process:",
            choices=choices,
            style=CLI_STYLE,
        ).ask()

    if not selected:
        print("No files selected. Exiting.")
        sys.exit(0)

    # Workflow selection
    if saved_config and "workflow" in saved_config:
        workflow = saved_config["workflow"]
        print(f"  Using saved workflow: {workflow}")
    else:
        workflow = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("Parse and export file contents (CSV/JSON)", value="parse_export"),
                questionary.Choice("Generate baseline exposure surface", value="generate_surface"),
                questionary.Choice("Diagnose files (dry-run report)", value="diagnose"),
            ],
            style=CLI_STYLE,
        ).ask()

    # Run selected workflow
    if workflow == "parse_export":
        if saved_config and "parse_config" in saved_config:
            config = saved_config["parse_config"]
            config["output_dir"] = Path(config["output_dir"])
        else:
            config = _prompt_parse_export_config()
        run_parse_export(selected, config)

    elif workflow == "generate_surface":
        if saved_config and "surface_config" in saved_config:
            config = saved_config["surface_config"]
            # Restore Path objects
            config["landing"] = [(Path(p), w) for p, w in config.get("landing", [])]
            config["takeoff"] = [(Path(p), w) for p, w in config.get("takeoff", [])]
        else:
            config = _prompt_surface_config(selected, folder)
        run_generate_surface(config)

    elif workflow == "diagnose":
        crs = questionary.text(
            "AERMOD coordinate system (CRS) to check against:",
            default=DEFAULT_AERMOD_CRS,
            style=CLI_STYLE,
        ).ask()
        run_diagnose(selected, crs)

    # Offer to save config (not for diagnose, which has nothing to replay)
    if workflow != "diagnose":
        save = questionary.confirm(
            "Save these settings for next time?",
            default=False,
            style=CLI_STYLE,
        ).ask()
        if save:
            cfg_to_save = {
                "selected_files": [p.name for p in selected],
                "workflow": workflow,
            }
            if workflow == "parse_export":
                cfg_to_save["parse_config"] = {
                    **config,
                    "output_dir": str(config["output_dir"]),
                }
            elif workflow == "generate_surface":
                cfg_to_save["surface_config"] = {
                    **config,
                    "landing": [(str(p), w) for p, w in (config.get("landing") or [])],
                    "takeoff": [(str(p), w) for p, w in (config.get("takeoff") or [])],
                }
            save_config(cfg_to_save, folder)


if __name__ == "__main__":
    main()
