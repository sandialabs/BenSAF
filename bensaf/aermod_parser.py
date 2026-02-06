import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from collections import defaultdict


class AermodParser:
    """Comprehensive parser for AERMOD .OUT/.ADO/.ADI files.
    
    Extracts and organizes all data from AERMOD files including:
    - Configuration (Control pathway)
    - Sources (Source pathway)
    - Receptors (Receptor pathway)
    - Meteorology (Meteorology pathway)
    - Output settings (Output pathway)
    - Results (concentration data from various averaging periods) - .ADO/.OUT only
    - Summaries (model setup, source groups, receptor networks)
    
    Supports both input files (.ADI) and output files (.ADO/.OUT).
    """
    
    def __init__(self, file_path: str):
        """Initialize parser with AERMOD file path.
        
        Args:
            file_path: Path to AERMOD .OUT, .ADO, or .ADI file
        """
        self.file_path = Path(file_path)
        self.lines = None
        self.sections = []
        self.network_info = {}  # Dictionary mapping network_id to network configuration
        self.file_type = None  # Will be set to 'ADI' or 'ADO' after file is loaded
        
    def _load_file(self):
        """Load and read all lines from the file."""
        if self.lines is None:
            with open(self.file_path, 'r', encoding='latin-1') as f:
                self.lines = f.readlines()
            # Detect file type
            self.file_type = self._detect_file_type()
    
    def _detect_file_type(self) -> str:
        """Detect whether file is .ADI (input) or .ADO/.OUT (output).
        
        Returns:
            'ADI' for input files, 'ADO' for output files
        """
        # Check file extension first
        ext = self.file_path.suffix.upper()
        if ext == '.ADI':
            return 'ADI'
        elif ext in ['.ADO', '.OUT']:
            return 'ADO'
        
        # Fallback: check for concentration result sections (only in .ADO/.OUT)
        # Look for concentration result markers in first 1000 lines
        for i, line in enumerate(self.lines[:min(1000, len(self.lines))]):
            if '*** THE' in line and ('CONCENTRATION' in line or 'AVERAGE' in line):
                return 'ADO'
            if '*** CONCURRENT' in line and 'AVERAGE CONCENTRATION' in line:
                return 'ADO'
        
        # If no concentration sections found, likely .ADI
        return 'ADI'
    
    def _identify_sections(self) -> List[Dict]:
        """First pass: Identify all sections in the file.
        
        Returns:
            List of section dictionaries with type, title, and line numbers
        """
        self._load_file()
        sections = []
        current_section = None
        
        for i, line in enumerate(self.lines):
            line_stripped = line.strip()
            
            # Control pathway
            if line_stripped == 'CO STARTING':
                current_section = {
                    'type': 'CONTROL_PATHWAY',
                    'title': 'AERMOD Control Pathway',
                    'line_start': i,
                    'line_end': None
                }
            elif line_stripped == 'CO FINISHED' and current_section and current_section['type'] == 'CONTROL_PATHWAY':
                current_section['line_end'] = i
                sections.append(current_section)
                current_section = None
            
            # Source pathway
            elif line_stripped == 'SO STARTING':
                current_section = {
                    'type': 'SOURCE_PATHWAY',
                    'title': 'AERMOD Source Pathway',
                    'line_start': i,
                    'line_end': None
                }
            elif line_stripped == 'SO FINISHED' and current_section and current_section['type'] == 'SOURCE_PATHWAY':
                current_section['line_end'] = i
                sections.append(current_section)
                current_section = None
            
            # Receptor pathway
            elif line_stripped == 'RE STARTING':
                current_section = {
                    'type': 'RECEPTOR_PATHWAY',
                    'title': 'AERMOD Receptor Pathway',
                    'line_start': i,
                    'line_end': None
                }
            elif line_stripped == 'RE FINISHED' and current_section and current_section['type'] == 'RECEPTOR_PATHWAY':
                current_section['line_end'] = i
                sections.append(current_section)
                current_section = None
            
            # Meteorology pathway
            elif line_stripped == 'ME STARTING':
                current_section = {
                    'type': 'METEOROLOGY_PATHWAY',
                    'title': 'AERMOD Meteorology Pathway',
                    'line_start': i,
                    'line_end': None
                }
            elif line_stripped == 'ME FINISHED' and current_section and current_section['type'] == 'METEOROLOGY_PATHWAY':
                current_section['line_end'] = i
                sections.append(current_section)
                current_section = None
            
            # Output pathway
            elif line_stripped == 'OU STARTING':
                current_section = {
                    'type': 'OUTPUT_PATHWAY',
                    'title': 'AERMOD Output Pathway',
                    'line_start': i,
                    'line_end': None
                }
            elif line_stripped == 'OU FINISHED' and current_section and current_section['type'] == 'OUTPUT_PATHWAY':
                current_section['line_end'] = i
                sections.append(current_section)
                current_section = None
            
            # Results sections - concentration data
            # Check for concurrent sections first (they don't have "*** THE")
            elif '*** CONCURRENT' in line and 'AVERAGE CONCENTRATION' in line:
                section_type = 'CONCURRENT_AVERAGE'
                section_title = 'Concurrent Average Concentration'
                averaging_period = None
                period_match = re.search(r'(\d+)-HR AVERAGE', line)
                if period_match:
                    averaging_period = int(period_match.group(1))
                
                current_section = {
                    'type': section_type,
                    'title': section_title,
                    'line_start': i,
                    'line_end': None,
                    'averaging_period': averaging_period,
                    'pollutant': None,
                    'source_group': None,
                    'network_id': None,
                    'network_type': None,
                    'origin_x': None,
                    'origin_y': None,
                    'date': None,
                    'day': None,
                    'year': None
                }
            
            elif '*** THE' in line:
                section_type = None
                section_title = None
                averaging_period = None
                
                if 'ANNUAL AVERAGE CONCENTRATION' in line:
                    section_type = 'ANNUAL_AVERAGE'
                    section_title = 'Annual Average Concentration'
                    years_match = re.search(r'AVERAGED OVER\s+(\d+)\s+YEARS', line)
                    if years_match:
                        averaging_period = int(years_match.group(1))
                elif 'HIGHEST' in line and 'CONCENTRATION' in line:
                    if '1ST HIGHEST' in line:
                        section_type = '1ST_HIGHEST'
                        section_title = '1st Highest Concentration'
                    elif '2ND HIGHEST' in line:
                        section_type = '2ND_HIGHEST'
                        section_title = '2nd Highest Concentration'
                    elif '3RD HIGHEST' in line:
                        section_type = '3RD_HIGHEST'
                        section_title = '3rd Highest Concentration'
                    
                    period_match = re.search(r'(\d+)-HR AVERAGE', line)
                    if period_match:
                        averaging_period = int(period_match.group(1))
                
                if section_type:
                    current_section = {
                        'type': section_type,
                        'title': section_title,
                        'line_start': i,
                        'line_end': None,
                        'averaging_period': averaging_period,
                        'pollutant': None,
                        'source_group': None,
                        'network_id': None,
                        'network_type': None,
                        'origin_x': None,
                        'origin_y': None,
                        'date': None,
                        'day': None,
                        'year': None
                    }
            
            # Summary sections
            elif '*** MODEL SETUP OPTIONS SUMMARY' in line:
                current_section = {
                    'type': 'MODEL_SETUP_SUMMARY',
                    'title': 'Model Setup Options Summary',
                    'line_start': i,
                    'line_end': None
                }
            elif '*** SOURCE IDs DEFINING SOURCE GROUPS' in line:
                current_section = {
                    'type': 'SOURCE_GROUPS_SUMMARY',
                    'title': 'Source IDs Defining Source Groups',
                    'line_start': i,
                    'line_end': None
                }
            elif '*** GRIDDED RECEPTOR NETWORK SUMMARY' in line:
                current_section = {
                    'type': 'RECEPTOR_NETWORK_SUMMARY',
                    'title': 'Gridded Receptor Network Summary',
                    'line_start': i,
                    'line_end': None
                }
            
            # Source emission rate scalars
            elif '* SOURCE EMISSION RATE SCALARS' in line:
                current_section = {
                    'type': 'SOURCE_EMISSION_SCALARS',
                    'title': 'Source Emission Rate Scalars',
                    'line_start': i,
                    'line_end': None
                }
            
            # Update current section metadata
            if current_section:
                # Extract pollutant
                if '** CONC OF' in line:
                    pollutant_match = re.search(r'CONC OF\s+(\w+)', line)
                    if pollutant_match:
                        current_section['pollutant'] = pollutant_match.group(1)
                
                # Extract source group
                if 'SOURCE GROUP:' in line:
                    group_match = re.search(r'SOURCE GROUP:\s+(\w+)', line)
                    if group_match:
                        current_section['source_group'] = group_match.group(1)
                
                # Extract network info
                if 'NETWORK ID:' in line:
                    id_match = re.search(r'NETWORK ID:\s+(\w+)', line)
                    if id_match:
                        current_section['network_id'] = id_match.group(1)
                if 'NETWORK TYPE:' in line:
                    type_match = re.search(r'NETWORK TYPE:\s+(\w+)', line)
                    if type_match:
                        current_section['network_type'] = type_match.group(1)
                
                # Extract origin for polar networks
                if 'ORIGIN FOR POLAR NETWORK' in line:
                    for check_idx in range(i, min(i + 5, len(self.lines))):
                        check_line = self.lines[check_idx]
                        x_match = re.search(r'X-ORIG\s*=\s*([\d\.\-]+)', check_line)
                        y_match = re.search(r'Y-ORIG\s*=\s*([\d\.\-]+)', check_line)
                        if x_match:
                            current_section['origin_x'] = float(x_match.group(1))
                        if y_match:
                            current_section['origin_y'] = float(y_match.group(1))
                
                # Extract date info for concurrent averages
                if current_section.get('type') == 'CONCURRENT_AVERAGE':
                    # Check current line and previous lines for day/year info
                    day_match = re.search(r'DAY\s+(\d+)\s+OF\s+(\d+)', line)
                    if day_match:
                        current_section['day'] = int(day_match.group(1))
                        current_section['year'] = int(day_match.group(2))
                    # Also check the section start line
                    elif current_section['line_start'] == i:
                        start_line = self.lines[current_section['line_start']]
                        day_match = re.search(r'DAY\s+(\d+)\s+OF\s+(\d+)', start_line)
                        if day_match:
                            current_section['day'] = int(day_match.group(1))
                            current_section['year'] = int(day_match.group(2))
                
                # End section markers
                if '*** AERMOD - VERSION' in line and i > current_section['line_start'] + 10:
                    if current_section['line_end'] is None:
                        current_section['line_end'] = i
                        sections.append(current_section)
                        current_section = None
        
        # Close any open sections at end of file
        if current_section:
            current_section['line_end'] = len(self.lines) - 1
            sections.append(current_section)
        
        return sections
    
    def _parse_control_pathway(self, section: Dict) -> Dict:
        """Parse control pathway section.
        
        Args:
            section: Section dictionary
            
        Returns:
            Dictionary with control pathway data
        """
        start = section['line_start']
        end = section.get('line_end', len(self.lines))
        
        data = {
            'title': section.get('title'),
            'options': {}
        }
        
        for i in range(start, min(end, len(self.lines))):
            line = self.lines[i]
            
            if line.strip().startswith('TITLEONE'):
                data['title_one'] = line.replace('TITLEONE', '').strip()
            elif line.strip().startswith('MODELOPT'):
                data['model_options'] = line.replace('MODELOPT', '').strip()
            elif line.strip().startswith('AVERTIME'):
                parts = line.replace('AVERTIME', '').strip().split()
                if parts:
                    data['averaging_times'] = parts
            elif line.strip().startswith('POLLUTID'):
                data['pollutant_id'] = line.replace('POLLUTID', '').strip()
            elif line.strip().startswith('ERRORFIL'):
                data['error_file'] = line.replace('ERRORFIL', '').strip()
        
        return data
    
    def _parse_source_pathway(self, section: Dict) -> pd.DataFrame:
        """Parse source pathway section.
        
        Parses LOCATION lines and SRCPARAM lines (for .ADI files).
        Source parameters are merged into the DataFrame when available.
        
        Args:
            section: Section dictionary
            
        Returns:
            DataFrame with source locations and parameters.
            Columns include: source_id, source_type, x_coord, y_coord, z_coord, source_group.
            For .ADI files, also includes: srcparam1, srcparam2, srcparam3, srcparam4.
        """
        start = section['line_start']
        end = section.get('line_end', len(self.lines))
        
        sources = []
        srcparams = {}  # Dictionary mapping source_id to parameters
        current_source_group = None
        
        for i in range(start, min(end, len(self.lines))):
            line = self.lines[i]
            
            # Source location
            if line.strip().startswith('LOCATION'):
                parts = line.strip().split()
                if len(parts) >= 5:
                    source_id = parts[1]
                    source_type = parts[2]
                    try:
                        x_coord = float(parts[3])
                        y_coord = float(parts[4])
                        z_coord = float(parts[5]) if len(parts) > 5 else None
                        
                        sources.append({
                            'source_id': source_id,
                            'source_type': source_type,
                            'x_coord': x_coord,
                            'y_coord': y_coord,
                            'z_coord': z_coord,
                            'source_group': current_source_group
                        })
                    except (ValueError, IndexError):
                        continue
            
            # Source parameters (SRCPARAM) - for .ADI files
            elif line.strip().startswith('SRCPARAM'):
                srcparam_data = self._parse_srcparam(line)
                if srcparam_data:
                    source_id = srcparam_data['source_id']
                    srcparams[source_id] = srcparam_data
            
            # Source group definition
            if 'SOURCE GROUP:' in line or line.strip().startswith('SRCGROUP'):
                group_match = re.search(r'(?:SOURCE GROUP:|SRCGROUP)\s+(\w+)', line)
                if group_match:
                    current_source_group = group_match.group(1)
        
        df = pd.DataFrame(sources) if sources else pd.DataFrame()
        
        # Merge source parameters if available
        if len(df) > 0 and srcparams:
            param_rows = []
            for source_id, params in srcparams.items():
                param_rows.append({
                    'source_id': source_id,
                    **{k: v for k, v in params.items() if k != 'source_id'}
                })
            if param_rows:
                param_df = pd.DataFrame(param_rows)
                df = df.merge(param_df, on='source_id', how='left')
        
        return df
    
    def _parse_srcparam(self, line: str) -> Optional[Dict]:
        """Parse SRCPARAM line.
        
        Format: SRCPARAM <source_id> <param1> <param2> <param3> <param4>
        Parameters vary by source type (VOLUME, POINT, AREA, etc.)
        
        Args:
            line: Line containing SRCPARAM data
            
        Returns:
            Dictionary with source_id and parameters, or None if parsing fails
        """
        pattern = r'SRCPARAM\s+(\w+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)'
        match = re.search(pattern, line)
        if match:
            try:
                return {
                    'source_id': match.group(1),
                    'srcparam1': float(match.group(2)),
                    'srcparam2': float(match.group(3)),
                    'srcparam3': float(match.group(4)),
                    'srcparam4': float(match.group(5))
                }
            except ValueError:
                return None
        return None
    
    def _parse_gridcart_definition(self, start: int, end: int, network_id: str) -> pd.DataFrame:
        """Parse GRIDCART grid definition and generate receptor coordinates.
        
        Parses XYINC, ELEV, and HILL lines to generate all receptor locations.
        
        Args:
            start: Start line number of grid definition
            end: End line number of grid definition
            network_id: Network identifier
            
        Returns:
            DataFrame with receptor coordinates (receptor_id, x_coord, y_coord, elevation, network_id, network_type)
        """
        x_origin = None
        nx = None
        x_spacing = None
        y_origin = None
        ny = None
        y_spacing = None
        
        elevations = {}  # Dictionary mapping (x_idx, y_idx) to elevation
        hills = {}  # Dictionary mapping (x_idx, y_idx) to hill_id
        
        # Track ELEV/HILL lines to determine y_idx
        elev_y_idx = 0
        hill_y_idx = 0
        
        for i in range(start, min(end, len(self.lines))):
            line = self.lines[i]
            line_stripped = line.strip()
            
            # Parse XYINC line: XYINC <x_origin> <nx> <x_spacing> <y_origin> <ny> <y_spacing>
            if line_stripped.startswith('XYINC'):
                parts = line_stripped.split()
                if len(parts) >= 7:
                    try:
                        x_origin = float(parts[1])
                        nx = int(parts[2])
                        x_spacing = float(parts[3])
                        y_origin = float(parts[4])
                        ny = int(parts[5])
                        y_spacing = float(parts[6])
                    except (ValueError, IndexError):
                        pass
            
            # Parse ELEV line: ELEV <terrain_type> <elevation_values...>
            elif line_stripped.startswith('ELEV'):
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        terrain_type = int(parts[1])
                        # Parse elevation values (one per x coordinate)
                        elev_values = [float(p) for p in parts[2:] if self._is_numeric(p)]
                        if elev_values and nx is not None:
                            # Use sequential y_idx based on order of ELEV lines
                            for x_idx, elev in enumerate(elev_values):
                                if x_idx < nx:
                                    elevations[(x_idx, elev_y_idx)] = elev
                            # Increment y_idx for next ELEV line
                            elev_y_idx += 1
                    except (ValueError, IndexError):
                        pass
            
            # Parse HILL line: HILL <hill_id> <hill_values...>
            elif line_stripped.startswith('HILL'):
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        hill_id = int(parts[1])
                        # Parse hill values (one per x coordinate)
                        hill_values = [float(p) for p in parts[2:] if self._is_numeric(p)]
                        if hill_values and nx is not None:
                            # Use sequential y_idx based on order of HILL lines
                            for x_idx, hill_val in enumerate(hill_values):
                                if x_idx < nx:
                                    hills[(x_idx, hill_y_idx)] = hill_id
                            # Increment y_idx for next HILL line
                            hill_y_idx += 1
                    except (ValueError, IndexError):
                        pass
        
        # Generate receptor coordinates
        receptors = []
        if x_origin is not None and nx is not None and x_spacing is not None and \
           y_origin is not None and ny is not None and y_spacing is not None:
            
            receptor_id = 0
            for y_idx in range(ny):
                y_coord = y_origin + y_idx * y_spacing
                for x_idx in range(nx):
                    x_coord = x_origin + x_idx * x_spacing
                    
                    elevation = elevations.get((x_idx, y_idx), None)
                    hill_id = hills.get((x_idx, y_idx), None)
                    
                    receptors.append({
                        'receptor_id': f'{network_id}_{receptor_id}',
                        'x_coord': x_coord,
                        'y_coord': y_coord,
                        'elevation': elevation,
                        'hill_id': hill_id,
                        'network_id': network_id,
                        'network_type': 'GRIDCART'
                    })
                    receptor_id += 1
        
        return pd.DataFrame(receptors) if receptors else pd.DataFrame()
    
    def _parse_gridpolr_definition(self, start: int, end: int, network_id: str) -> pd.DataFrame:
        """Parse GRIDPOLR grid definition and generate receptor coordinates.
        
        Parses ORIG, DIST, and DIR lines to generate all receptor locations.
        
        Args:
            start: Start line number of grid definition
            end: End line number of grid definition
            network_id: Network identifier
            
        Returns:
            DataFrame with receptor coordinates (receptor_id, x_coord, y_coord, distance, direction, network_id, network_type)
        """
        origin_x = 0.0
        origin_y = 0.0
        distances = []
        directions = []
        
        for i in range(start, min(end, len(self.lines))):
            line = self.lines[i]
            line_stripped = line.strip()
            
            # Parse origin coordinates
            if 'X-ORIG' in line or 'Y-ORIG' in line:
                x_match = re.search(r'X-ORIG\s*=\s*([\d\.\-]+)', line)
                y_match = re.search(r'Y-ORIG\s*=\s*([\d\.\-]+)', line)
                if x_match:
                    origin_x = float(x_match.group(1))
                if y_match:
                    origin_y = float(y_match.group(1))
            
            # Parse DIST line: DIST <dist1> <dist2> ...
            elif line_stripped.startswith('DIST'):
                parts = line_stripped.split()
                distances = [float(p) for p in parts[1:] if self._is_numeric(p)]
            
            # Parse DIR line: DIR <dir1> <dir2> ...
            elif line_stripped.startswith('DIR'):
                parts = line_stripped.split()
                directions = [float(p) for p in parts[1:] if self._is_numeric(p)]
        
        # Generate receptor coordinates
        receptors = []
        if distances and directions:
            receptor_id = 0
            for distance in distances:
                for direction in directions:
                    # Convert polar to Cartesian
                    # AERMOD uses meteorological convention: 0° = North, clockwise
                    direction_rad = np.radians(direction)
                    x_coord = origin_x + distance * np.sin(direction_rad)
                    y_coord = origin_y + distance * np.cos(direction_rad)
                    
                    receptors.append({
                        'receptor_id': f'{network_id}_{receptor_id}',
                        'x_coord': x_coord,
                        'y_coord': y_coord,
                        'distance': distance,
                        'direction': direction,
                        'network_id': network_id,
                        'network_type': 'GRIDPOLR'
                    })
                    receptor_id += 1
        
        return pd.DataFrame(receptors) if receptors else pd.DataFrame()
    
    def _is_numeric(self, s: str) -> bool:
        """Check if string represents a numeric value.
        
        Args:
            s: String to check
            
        Returns:
            True if string is numeric, False otherwise
        """
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _parse_receptor_pathway(self, section: Dict) -> Dict:
        """Parse receptor pathway section and store network information.
        
        For .ADI files, also parses grid definitions (GRIDCART/GRIDPOLR) and generates
        receptor coordinates.
        
        Args:
            section: Section dictionary
            
        Returns:
            Dictionary with receptor network information and optionally receptor coordinates DataFrame
        """
        start = section['line_start']
        end = section.get('line_end', len(self.lines))
        
        data = {
            'title': section.get('title'),
            'networks': []
        }
        
        current_network = None
        origin_x = None
        origin_y = None
        receptor_dfs = []  # List of DataFrames with receptor coordinates (for .ADI files)
        
        # Track grid definitions for .ADI files
        grid_start = None
        grid_network_id = None
        grid_type = None
        
        for i in range(start, min(end, len(self.lines))):
            line = self.lines[i]
            line_stripped = line.strip()
            
            # Detect GRIDCART or GRIDPOLR definitions (for .ADI files)
            # Check for END first to avoid detecting END as a new start
            if grid_start is not None:
                is_end = False
                if grid_type == 'GRIDCART' and 'GRIDCART' in line_stripped and 'END' in line_stripped:
                    is_end = True
                elif grid_type == 'GRIDPOLR' and 'GRIDPOLR' in line_stripped and 'END' in line_stripped:
                    is_end = True
                
                if is_end:
                    # Parse the grid definition
                    if grid_type == 'GRIDCART':
                        grid_df = self._parse_gridcart_definition(grid_start, i, grid_network_id)
                        if len(grid_df) > 0:
                            receptor_dfs.append(grid_df)
                    elif grid_type == 'GRIDPOLR':
                        grid_df = self._parse_gridpolr_definition(grid_start, i, grid_network_id)
                        if len(grid_df) > 0:
                            receptor_dfs.append(grid_df)
                    
                    grid_start = None
                    grid_network_id = None
                    grid_type = None
                    continue  # Skip processing this line further
            
            # Detect GRIDCART or GRIDPOLR start (only if not already in a grid)
            if grid_start is None:
                if line_stripped.startswith('GRIDCART'):
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        grid_network_id = parts[1]  # Network ID is the second token
                        grid_type = 'GRIDCART'
                        grid_start = i
                        # Store network info
                        if grid_network_id not in self.network_info:
                            self.network_info[grid_network_id] = {
                                'network_type': 'GRIDCART',
                                'origin_x': 0.0,
                                'origin_y': 0.0
                            }
                elif line_stripped.startswith('GRIDPOLR'):
                    parts = line_stripped.split()
                    if len(parts) >= 2:
                        grid_network_id = parts[1]  # Network ID is the second token
                        grid_type = 'GRIDPOLR'
                        grid_start = i
                        # Store network info
                        if grid_network_id not in self.network_info:
                            self.network_info[grid_network_id] = {
                                'network_type': 'GRIDPOLR',
                                'origin_x': 0.0,
                                'origin_y': 0.0
                            }
            
            # Extract origin for polar networks
            if 'ORIGIN FOR POLAR NETWORK' in line or 'X-ORIG' in line or 'Y-ORIG' in line:
                for check_idx in range(i, min(i + 5, len(self.lines))):
                    check_line = self.lines[check_idx]
                    x_match = re.search(r'X-ORIG\s*=\s*([\d\.\-]+)', check_line)
                    y_match = re.search(r'Y-ORIG\s*=\s*([\d\.\-]+)', check_line)
                    if x_match:
                        origin_x = float(x_match.group(1))
                    if y_match:
                        origin_y = float(y_match.group(1))
            
            if 'NETWORK ID:' in line:
                if current_network:
                    data['networks'].append(current_network)
                    # Store in class attribute for lookup during concentration parsing
                    if current_network['network_id']:
                        self.network_info[current_network['network_id']] = {
                            'network_type': current_network['network_type'],
                            'origin_x': origin_x if origin_x is not None else 0.0,
                            'origin_y': origin_y if origin_y is not None else 0.0
                        }
                
                id_match = re.search(r'NETWORK ID:\s+(\w+)', line)
                type_match = re.search(r'NETWORK TYPE:\s+(\w+)', line)
                
                current_network = {
                    'network_id': id_match.group(1) if id_match else None,
                    'network_type': type_match.group(1) if type_match else None,
                    'origin_x': origin_x if origin_x is not None else 0.0,
                    'origin_y': origin_y if origin_y is not None else 0.0
                }
                # Reset origin for next network
                origin_x = None
                origin_y = None
        
        # Handle grid definition that extends to end of section
        if grid_start is not None:
            if grid_type == 'GRIDCART':
                grid_df = self._parse_gridcart_definition(grid_start, end, grid_network_id)
                if len(grid_df) > 0:
                    receptor_dfs.append(grid_df)
            elif grid_type == 'GRIDPOLR':
                grid_df = self._parse_gridpolr_definition(grid_start, end, grid_network_id)
                if len(grid_df) > 0:
                    receptor_dfs.append(grid_df)
        
        if current_network:
            data['networks'].append(current_network)
            # Store in class attribute
            if current_network['network_id']:
                self.network_info[current_network['network_id']] = {
                    'network_type': current_network['network_type'],
                    'origin_x': current_network.get('origin_x', 0.0),
                    'origin_y': current_network.get('origin_y', 0.0)
                }
        
        # For .ADI files, add receptor coordinates DataFrame
        if self.file_type == 'ADI' and receptor_dfs:
            data['receptors'] = pd.concat(receptor_dfs, ignore_index=True)
        
        return data
    
    def _parse_concentration_results(self, section: Dict) -> pd.DataFrame:
        """Parse concentration results section (annual, highest, concurrent).
        
        Requires network information from RECEPTOR_PATHWAY to determine coordinate system.
        
        Args:
            section: Section dictionary
            
        Returns:
            DataFrame with concentration data
            
        Raises:
            ValueError: If network information is not available for the network_id
        """
        start = section['line_start']
        end = section.get('line_end', len(self.lines))
        
        # Get network_id from section
        network_id = section.get('network_id')
        
        # Look up network information from RECEPTOR_PATHWAY
        network_type = None
        origin_x = 0.0
        origin_y = 0.0
        
        if network_id and network_id in self.network_info:
            network_info = self.network_info[network_id]
            network_type = network_info['network_type']
            origin_x = network_info.get('origin_x', 0.0)
            origin_y = network_info.get('origin_y', 0.0)
        elif section.get('network_type'):
            # Fallback to section metadata if network_id lookup fails
            network_type = section.get('network_type')
            origin_x = section.get('origin_x', 0.0)
            origin_y = section.get('origin_y', 0.0)
        else:
            # Try to detect from section content as last resort
            for i in range(start, min(start + 100, end, len(self.lines))):
                line = self.lines[i]
                if 'NETWORK TYPE:' in line:
                    type_match = re.search(r'NETWORK TYPE:\s+(\w+)', line)
                    if type_match:
                        network_type = type_match.group(1)
                        break
                elif 'GRIDPOLR' in line or 'DIRECTION' in line:
                    network_type = 'GRIDPOLR'
                    break
                elif 'GRIDCART' in line or 'X-COORD (METERS)' in line:
                    network_type = 'GRIDCART'
                    break
        
        # Require network_type to be determined
        if not network_type:
            # Last resort: try to extract from section content
            for i in range(start, min(start + 50, end, len(self.lines))):
                line = self.lines[i]
                if 'NETWORK TYPE:' in line:
                    type_match = re.search(r'NETWORK TYPE:\s+(\w+)', line)
                    if type_match:
                        network_type = type_match.group(1)
                        # Store for future use
                        if network_id and network_id not in self.network_info:
                            self.network_info[network_id] = {
                                'network_type': network_type,
                                'origin_x': 0.0,
                                'origin_y': 0.0
                            }
                        break
            
            if not network_type:
                raise ValueError(
                    f"Could not determine network type for concentration section. "
                    f"Network ID: {network_id}. "
                    f"RECEPTOR_PATHWAY or RECEPTOR_NETWORK_SUMMARY must be parsed before concentration results. "
                    f"Ensure these sections are included in section_types or parse all sections."
                )
        
        # Update section with network info
        section['network_type'] = network_type
        if origin_x != 0.0 or origin_y != 0.0:
            section['origin_x'] = origin_x
            section['origin_y'] = origin_y
        
        results = []
        
        if network_type == 'GRIDPOLR':
            results = self._extract_polar_data(section, start, end)
        elif network_type == 'GRIDCART':
            results = self._extract_cartesian_data(section, start, end)
        else:
            raise ValueError(f"Unknown network type: {network_type}. Expected GRIDCART or GRIDPOLR.")
        
        df = pd.DataFrame(results)
        
        # Convert polar to cartesian if needed
        if len(df) > 0 and 'direction' in df.columns and 'distance' in df.columns:
            direction_rad = np.radians(df['direction'].values)
            
            if 'origin_x' in df.columns:
                origin_x = df['origin_x'].astype(float).fillna(0.0).values
            else:
                origin_x = np.zeros(len(df))
            
            if 'origin_y' in df.columns:
                origin_y = df['origin_y'].astype(float).fillna(0.0).values
            else:
                origin_y = np.zeros(len(df))
            
            # AERMOD uses meteorological convention: 0° = North, clockwise
            # Convert to standard: 0° = East, counterclockwise
            # Standard: x = r*cos(θ), y = r*sin(θ)
            # AERMOD: direction is from North, clockwise
            # So: x = r*sin(θ), y = r*cos(θ) where θ is from North
            df['x_coord'] = origin_x + df['distance'].values * np.sin(direction_rad)
            df['y_coord'] = origin_y + df['distance'].values * np.cos(direction_rad)
            
            df = df.drop(columns=['direction', 'distance', 'origin_x', 'origin_y'], errors='ignore')
        
        return df
    
    def _extract_polar_data(self, section: Dict, start: int, end: int) -> List[Dict]:
        """Extract data from a polar coordinate section.
        
        Args:
            section: Section dictionary
            start: Start line number
            end: End line number
            
        Returns:
            List of data dictionaries
        """
        results = []
        distances = None
        origin_x = section.get('origin_x', 0.0)
        origin_y = section.get('origin_y', 0.0)
        
        for i in range(start, min(end, start + 1000, len(self.lines))):
            line = self.lines[i]
            
            if 'DISTANCE (METERS)' in line:
                next_idx = i + 1
                if next_idx < len(self.lines):
                    next_line = self.lines[next_idx]
                    distances = re.findall(r'(\d+\.\d{2})', next_line)
                    distances = [float(d) for d in distances if float(d) > 100]
            
            elif distances and re.match(r'^\s*\d+\.\d+\s+\|', line):
                dir_match = re.match(r'^\s*(\d+\.\d+)\s+\|', line)
                if dir_match:
                    direction = float(dir_match.group(1))
                    data_part = line.split('|')[1] if '|' in line else ''
                    
                    pattern_with_date = r'(\d+\.\d+)([a-z]?)\((\d{8})\)'
                    pattern_simple = r'(\d+\.\d+)'
                    
                    matches_with_date = re.findall(pattern_with_date, data_part)
                    if matches_with_date:
                        for idx, (conc_str, flag, date_str) in enumerate(matches_with_date):
                            if idx < len(distances):
                                try:
                                    conc = float(conc_str)
                                    result = {
                                        'section_type': section['type'],
                                        'pollutant': section.get('pollutant'),
                                        'source_group': section.get('source_group'),
                                        'network_id': section.get('network_id'),
                                        'direction': direction,
                                        'distance': distances[idx],
                                        'concentration': conc,
                                        'flag': flag if flag else None,
                                        'date': date_str if date_str else None,
                                        'origin_x': origin_x,
                                        'origin_y': origin_y,
                                        'day': section.get('day'),
                                        'year': section.get('year'),
                                        'averaging_period': section.get('averaging_period')
                                    }
                                    results.append(result)
                                except ValueError:
                                    continue
                    else:
                        conc_values = re.findall(pattern_simple, data_part)
                        for idx, conc_str in enumerate(conc_values):
                            if idx < len(distances):
                                try:
                                    conc = float(conc_str)
                                    result = {
                                        'section_type': section['type'],
                                        'pollutant': section.get('pollutant'),
                                        'source_group': section.get('source_group'),
                                        'network_id': section.get('network_id'),
                                        'direction': direction,
                                        'distance': distances[idx],
                                        'concentration': conc,
                                        'flag': None,
                                        'date': None,
                                        'origin_x': origin_x,
                                        'origin_y': origin_y,
                                        'day': section.get('day'),
                                        'year': section.get('year'),
                                        'averaging_period': section.get('averaging_period')
                                    }
                                    results.append(result)
                                except ValueError:
                                    continue
            
            elif '*** AERMOD - VERSION' in line and i > start + 10:
                break
        
        return results
    
    def _extract_cartesian_data(self, section: Dict, start: int, end: int) -> List[Dict]:
        """Extract data from a Cartesian coordinate section.
        
        Args:
            section: Section dictionary
            start: Start line number
            end: End line number
            
        Returns:
            List of data dictionaries
        """
        results = []
        x_coords = None
        
        for i in range(start, min(end, start + 1000, len(self.lines))):
            line = self.lines[i]
            
            if 'X-COORD (METERS)' in line:
                next_idx = i + 1
                if next_idx < len(self.lines):
                    next_line = self.lines[next_idx]
                    x_coords = re.findall(r'(\d+\.\d{2})', next_line)
                    x_coords = [float(x) for x in x_coords if float(x) > 1000]
            
            elif x_coords and re.match(r'^\s*\d+\.\d+\s+\|', line):
                y_match = re.match(r'^\s*(\d+\.\d+)\s+\|', line)
                if y_match:
                    y_coord = float(y_match.group(1))
                    data_part = line.split('|')[1] if '|' in line else ''
                    
                    pattern_with_date = r'(\d+\.\d+)([a-z]?)\((\d{8})\)'
                    pattern_simple = r'(\d+\.\d+)'
                    
                    matches_with_date = re.findall(pattern_with_date, data_part)
                    if matches_with_date:
                        for idx, (conc_str, flag, date_str) in enumerate(matches_with_date):
                            if idx < len(x_coords):
                                try:
                                    conc = float(conc_str)
                                    result = {
                                        'section_type': section['type'],
                                        'pollutant': section.get('pollutant'),
                                        'source_group': section.get('source_group'),
                                        'network_id': section.get('network_id'),
                                        'x_coord': x_coords[idx],
                                        'y_coord': y_coord,
                                        'concentration': conc,
                                        'flag': flag if flag else None,
                                        'date': date_str if date_str else None,
                                        'day': section.get('day'),
                                        'year': section.get('year'),
                                        'averaging_period': section.get('averaging_period')
                                    }
                                    results.append(result)
                                except ValueError:
                                    continue
                    else:
                        conc_values = re.findall(pattern_simple, data_part)
                        for idx, conc_str in enumerate(conc_values):
                            if idx < len(x_coords):
                                try:
                                    conc = float(conc_str)
                                    result = {
                                        'section_type': section['type'],
                                        'pollutant': section.get('pollutant'),
                                        'source_group': section.get('source_group'),
                                        'network_id': section.get('network_id'),
                                        'x_coord': x_coords[idx],
                                        'y_coord': y_coord,
                                        'concentration': conc,
                                        'flag': None,
                                        'date': None,
                                        'day': section.get('day'),
                                        'year': section.get('year'),
                                        'averaging_period': section.get('averaging_period')
                                    }
                                    results.append(result)
                                except ValueError:
                                    continue
            
            elif '*** AERMOD - VERSION' in line and i > start + 10:
                break
        
        return results
    
    def _parse_summary_section(self, section: Dict) -> Dict:
        """Parse summary section.
        
        Args:
            section: Section dictionary
            
        Returns:
            Dictionary with summary data
        """
        start = section['line_start']
        end = section.get('line_end', len(self.lines))
        
        data = {
            'type': section['type'],
            'title': section.get('title'),
            'content': []
        }
        
        for i in range(start, min(end, start + 200, len(self.lines))):
            line = self.lines[i].strip()
            if line and not line.startswith('***') and not line.startswith('**'):
                data['content'].append(line)
        
        return data
    
    def parse(self, section_types: Optional[List[str]] = None) -> Dict[str, Union[pd.DataFrame, Dict, List]]:
        """Parse the AERMOD file and extract all data.
        
        Supports both input files (.ADI) and output files (.ADO/.OUT).
        For .ADI files, extracts source parameters (SRCPARAM) and generates receptor coordinates
        from grid definitions. For .ADO/.OUT files, extracts concentration results.
        
        RECEPTOR_PATHWAY is automatically parsed first if concentration results are requested,
        as it contains required network coordinate system information.
        
        Args:
            section_types: Optional list of section types to extract.
                          If None, extracts all sections.
                          Options: 'CONTROL_PATHWAY', 'SOURCE_PATHWAY', 'RECEPTOR_PATHWAY',
                                  'METEOROLOGY_PATHWAY', 'OUTPUT_PATHWAY', 'ANNUAL_AVERAGE',
                                  '1ST_HIGHEST', '2ND_HIGHEST', '3RD_HIGHEST', 'CONCURRENT_AVERAGE',
                                  'MODEL_SETUP_SUMMARY', 'SOURCE_GROUPS_SUMMARY', 
                                  'RECEPTOR_NETWORK_SUMMARY', 'SOURCE_EMISSION_SCALARS'
                          Note: Concentration result types (ANNUAL_AVERAGE, etc.) are only
                          available in .ADO/.OUT files.
        
        Returns:
            Dictionary with keys for each section type containing extracted data.
            For .ADI files, RECEPTOR_PATHWAY entries include a 'receptors' DataFrame with
            generated receptor coordinates. SOURCE_PATHWAY includes source parameters (srcparam1-4).
        """
        self._load_file()
        sections = self._identify_sections()
        self.sections = sections
        
        # Concentration result types that require network information
        concentration_types = ['ANNUAL_AVERAGE', '1ST_HIGHEST', '2ND_HIGHEST', '3RD_HIGHEST', 'CONCURRENT_AVERAGE']
        
        # Check if we need to parse receptor network info first
        needs_receptor_info = False
        if section_types:
            # Check if any concentration types are requested
            if any(ct in section_types for ct in concentration_types):
                needs_receptor_info = True
                # Ensure RECEPTOR_PATHWAY and RECEPTOR_NETWORK_SUMMARY are included
                required_sections = ['RECEPTOR_PATHWAY', 'RECEPTOR_NETWORK_SUMMARY']
                for req_section in required_sections:
                    if req_section not in section_types:
                        section_types = list(section_types) + [req_section]
        else:
            # If parsing all, we need receptor info for concentration results
            needs_receptor_info = True
        
        if section_types:
            sections = [s for s in sections if s['type'] in section_types]
        
        results = {}
        
        # Parse RECEPTOR_PATHWAY first if needed
        receptor_sections = [s for s in sections if s['type'] == 'RECEPTOR_PATHWAY']
        for section in receptor_sections:
            if section['type'] == 'RECEPTOR_PATHWAY':
                if section['type'] not in results:
                    results[section['type']] = []
                results[section['type']].append(self._parse_receptor_pathway(section))
        
        # Also parse RECEPTOR_NETWORK_SUMMARY for network info (often contains network details)
        # This is critical as RECEPTOR_PATHWAY may just reference external files
        network_summary_sections = [s for s in sections if s['type'] == 'RECEPTOR_NETWORK_SUMMARY']
        for section in network_summary_sections:
            start = section['line_start']
            end = section.get('line_end', len(self.lines))
            current_network_id = None
            
            # Extract network info directly from section lines
            for i in range(start, min(end, start + 200, len(self.lines))):
                line = self.lines[i]
                
                # Find network ID and type together
                id_match = re.search(r'NETWORK ID:\s+(\w+)', line)
                type_match = re.search(r'NETWORK TYPE:\s+(\w+)', line)
                
                if id_match:
                    current_network_id = id_match.group(1)
                    network_type = type_match.group(1) if type_match else None
                    # Initialize if not already in network_info
                    if current_network_id not in self.network_info:
                        self.network_info[current_network_id] = {
                            'network_type': network_type,
                            'origin_x': 0.0,
                            'origin_y': 0.0
                        }
                    elif network_type and not self.network_info[current_network_id]['network_type']:
                        # Update network_type if we found it
                        self.network_info[current_network_id]['network_type'] = network_type
                
                # Extract origin coordinates (can appear before or after network ID)
                x_match = re.search(r'X-ORIG\s*=\s*([\d\.\-]+)', line)
                y_match = re.search(r'Y-ORIG\s*=\s*([\d\.\-]+)', line)
                if x_match or y_match:
                    # If we have a current network, use it; otherwise try to find one
                    target_network = current_network_id
                    if not target_network:
                        # Look for network ID in nearby lines
                        for check_i in range(max(start, i - 20), min(end, i + 20, len(self.lines))):
                            check_line = self.lines[check_i]
                            check_id = re.search(r'NETWORK ID:\s+(\w+)', check_line)
                            if check_id:
                                target_network = check_id.group(1)
                                break
                    
                    if target_network:
                        if target_network not in self.network_info:
                            self.network_info[target_network] = {
                                'network_type': None,
                                'origin_x': 0.0,
                                'origin_y': 0.0
                            }
                        if x_match:
                            self.network_info[target_network]['origin_x'] = float(x_match.group(1))
                        if y_match:
                            self.network_info[target_network]['origin_y'] = float(y_match.group(1))
        
        # Also extract network info from concentration sections as fallback
        # This ensures we have network info even if RECEPTOR_NETWORK_SUMMARY is missing
        if not self.network_info:
            concentration_sections = [s for s in sections if s['type'] in concentration_types]
            for section in concentration_sections[:5]:  # Check first few sections
                network_id = section.get('network_id')
                network_type = section.get('network_type')
                if network_id and network_type:
                    if network_id not in self.network_info:
                        self.network_info[network_id] = {
                            'network_type': network_type,
                            'origin_x': section.get('origin_x', 0.0),
                            'origin_y': section.get('origin_y', 0.0)
                        }
        
        # Now parse remaining sections
        for section in sections:
            section_type = section['type']
            
            if section_type == 'CONTROL_PATHWAY':
                results[section_type] = self._parse_control_pathway(section)
            
            elif section_type == 'SOURCE_PATHWAY':
                df = self._parse_source_pathway(section)
                if section_type not in results:
                    results[section_type] = df
                else:
                    results[section_type] = pd.concat([results[section_type], df], ignore_index=True)
            
            elif section_type == 'RECEPTOR_PATHWAY':
                # Already parsed above, skip here
                if section_type not in results:
                    results[section_type] = []
                    results[section_type].append(self._parse_receptor_pathway(section))
            
            elif section_type in ['METEOROLOGY_PATHWAY', 'OUTPUT_PATHWAY']:
                if section_type not in results:
                    results[section_type] = []
                results[section_type].append({
                    'title': section.get('title'),
                    'line_start': section['line_start'],
                    'line_end': section.get('line_end')
                })
            
            elif section_type in ['ANNUAL_AVERAGE', '1ST_HIGHEST', '2ND_HIGHEST', '3RD_HIGHEST', 'CONCURRENT_AVERAGE']:
                df = self._parse_concentration_results(section)
                if section_type not in results:
                    results[section_type] = df
                else:
                    results[section_type] = pd.concat([results[section_type], df], ignore_index=True)
            
            elif section_type == 'RECEPTOR_NETWORK_SUMMARY':
                # Already processed above for network info, but also store summary
                if section_type not in results:
                    results[section_type] = []
                results[section_type].append(self._parse_summary_section(section))
            
            elif section_type in ['MODEL_SETUP_SUMMARY', 'SOURCE_GROUPS_SUMMARY', 'SOURCE_EMISSION_SCALARS']:
                if section_type not in results:
                    results[section_type] = []
                results[section_type].append(self._parse_summary_section(section))
        
        return results
    
    def get_sections_info(self) -> pd.DataFrame:
        """Get information about all sections in the file.
        
        Returns:
            DataFrame with section metadata
        """
        self._load_file()
        sections = self._identify_sections()
        return pd.DataFrame(sections)
    
    def extract_concentration_data(self, section_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract only concentration data (results sections).
        
        Args:
            section_types: Optional list of result section types to extract.
                          If None, extracts all result types.
                          Options: 'ANNUAL_AVERAGE', '1ST_HIGHEST', '2ND_HIGHEST', 
                                  '3RD_HIGHEST', 'CONCURRENT_AVERAGE'
            
        Returns:
            DataFrame with concentration data
        """
        if section_types is None:
            section_types = ['ANNUAL_AVERAGE', '1ST_HIGHEST', '2ND_HIGHEST', '3RD_HIGHEST', 'CONCURRENT_AVERAGE']
        
        all_data = self.parse(section_types=section_types)
        
        dfs = []
        for section_type in section_types:
            if section_type in all_data and isinstance(all_data[section_type], pd.DataFrame):
                if len(all_data[section_type]) > 0:
                    dfs.append(all_data[section_type])
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

