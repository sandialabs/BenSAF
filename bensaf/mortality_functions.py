"""
Library for managing mortality function parameters.

This module provides a class to load, manage, and modify mortality function
parameters stored in a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MortalityFunctionLibrary:
    """
    Library for managing mortality function parameters.
    
    This class loads mortality function parameters from a JSON file and provides
    methods to add, remove, and retrieve functions.
    """
    
    def __init__(self, json_path: Optional[Path] = None):
        """
        Initialize the mortality function library.
        
        Args:
            json_path: Path to the JSON file containing mortality function parameters.
                      If None, uses default path: bensaf/data/mortality_functions.json
        """
        if json_path is None:
            # Default to data directory in the repo
            repo_root = Path(__file__).parent.parent
            json_path = repo_root / "data" / "mortality_functions.json"
        
        self.json_path = Path(json_path)
        self.functions: Dict[int, Dict[str, Any]] = {}
        
        # Create directory if it doesn't exist
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing functions or create default
        if self.json_path.exists():
            self._load()
        else:
            self._create_default()
            self._save()
    
    def _load(self) -> None:
        """Load mortality functions from JSON file."""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            
            # Convert string keys to integers
            self.functions = {int(k): v for k, v in data.items()}
            logger.info(f"Loaded {len(self.functions)} mortality functions from {self.json_path}")
        except Exception as e:
            logger.error(f"Error loading mortality functions: {e}")
            raise
    
    def _save(self) -> None:
        """Save mortality functions to JSON file."""
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.functions, f, indent=2)
            logger.info(f"Saved {len(self.functions)} mortality functions to {self.json_path}")
        except Exception as e:
            logger.error(f"Error saving mortality functions: {e}")
            raise
    
    def _create_default(self) -> None:
        """Create default mortality functions."""
        self.functions = {
            0: {
                "title": "Bouma et al. Default",
                "mean_rr": 1.012,
                "lower_rr": 1.010,
                "upper_rr": 1.015,
                "unit_increase": 2723.0
            }
        }
        logger.info("Created default mortality function")
    
    def _get_next_id(self) -> int:
        """Get the next available ID."""
        if not self.functions:
            return 0
        return max(self.functions.keys()) + 1
    
    def add_function(self, title: str, mean_rr: float, lower_rr: float, 
                     upper_rr: float, unit_increase: float, 
                     function_id: Optional[int] = None) -> int:
        """
        Add a new mortality function to the library.
        
        Args:
            title: Title/name of the mortality function
            mean_rr: Mean relative risk
            lower_rr: Lower bound of relative risk (95% CI)
            upper_rr: Upper bound of relative risk (95% CI)
            unit_increase: Unit increase in pollutant concentration for the relative risk
            function_id: Optional ID for the function. If None, uses next available ID.
        
        Returns:
            The ID of the added function
        """
        if function_id is None:
            function_id = self._get_next_id()
        elif function_id in self.functions:
            raise ValueError(f"Function with ID {function_id} already exists")
        
        self.functions[function_id] = {
            "title": title,
            "mean_rr": mean_rr,
            "lower_rr": lower_rr,
            "upper_rr": upper_rr,
            "unit_increase": unit_increase
        }
        
        self._save()
        logger.info(f"Added mortality function '{title}' with ID {function_id}")
        return function_id
    
    def remove_function(self, function_id: int) -> bool:
        """
        Remove a mortality function from the library.
        
        Args:
            function_id: ID of the function to remove
        
        Returns:
            True if function was removed, False if it didn't exist
        """
        if function_id not in self.functions:
            logger.warning(f"Function with ID {function_id} not found")
            return False
        
        title = self.functions[function_id]["title"]
        del self.functions[function_id]
        self._save()
        logger.info(f"Removed mortality function '{title}' with ID {function_id}")
        return True
    
    def get_function(self, function_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a mortality function by ID.
        
        Args:
            function_id: ID of the function to retrieve
        
        Returns:
            Dictionary containing the function parameters, or None if not found
        """
        return self.functions.get(function_id)
    
    def get_all_functions(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all mortality functions.
        
        Returns:
            Dictionary mapping function IDs to function parameters
        """
        return self.functions.copy()
    
    def list_functions(self) -> List[Dict[str, Any]]:
        """
        Get a list of all functions with their IDs and titles.
        
        Returns:
            List of dictionaries with 'id' and 'title' keys
        """
        return [
            {"id": func_id, "title": func_data["title"]}
            for func_id, func_data in sorted(self.functions.items())
        ]
    
    def update_function(self, function_id: int, title: Optional[str] = None,
                       mean_rr: Optional[float] = None,
                       lower_rr: Optional[float] = None,
                       upper_rr: Optional[float] = None,
                       unit_increase: Optional[float] = None) -> bool:
        """
        Update parameters of an existing mortality function.
        
        Args:
            function_id: ID of the function to update
            title: New title (optional)
            mean_rr: New mean relative risk (optional)
            lower_rr: New lower relative risk (optional)
            upper_rr: New upper relative risk (optional)
            unit_increase: New unit increase (optional)
        
        Returns:
            True if function was updated, False if it didn't exist
        """
        if function_id not in self.functions:
            logger.warning(f"Function with ID {function_id} not found")
            return False
        
        if title is not None:
            self.functions[function_id]["title"] = title
        if mean_rr is not None:
            self.functions[function_id]["mean_rr"] = mean_rr
        if lower_rr is not None:
            self.functions[function_id]["lower_rr"] = lower_rr
        if upper_rr is not None:
            self.functions[function_id]["upper_rr"] = upper_rr
        if unit_increase is not None:
            self.functions[function_id]["unit_increase"] = unit_increase
        
        self._save()
        logger.info(f"Updated mortality function with ID {function_id}")
        return True

