import os
import unittest
import pandas as pd
from bensaf.aermod_parser import extract_receptor_values


class TestAermodParser(unittest.TestCase):
    
    def test_aermod_baldwin45(self):
        """Test parsing aermod-baldwin45.out and check dataframe is non-empty."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'aermod-examples', 'aermod-baldwin45.out')
        df = extract_receptor_values(file_path)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertGreater(len(df), 0, "DataFrame should have at least one row")
    
    def test_westflow_takeoff_data(self):
        """Test parsing westflow_takeoff.ADO from data directory and check dataframe is non-empty."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'aermod-examples', 'westflow_takeoff.ADO')
        df = extract_receptor_values(file_path)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertGreater(len(df), 0, "DataFrame should have at least one row")
    
    def test_westflow_takeoff_compare_csv(self):
        """Test parsing westflow_takeoff.ADO and compare to CSV."""
        ado_path = os.path.join(os.path.dirname(__file__), '..', 'kirklocal', 'AERMOD-elena', 'westflow_takeoff.ADO')
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'kirklocal', 'AERMOD-elena', 'westflow_takeoff.csv')
        
        df_parsed = extract_receptor_values(ado_path)
        df_csv = pd.read_csv(csv_path, header=None, names=['x_coord', 'y_coord', 'concentration', 'col4', 'col5', 'col6'])
        
        self.assertFalse(df_parsed.empty, "Parsed DataFrame should not be empty")
        self.assertFalse(df_csv.empty, "CSV DataFrame should not be empty")
        
        df_csv_subset = df_csv[['x_coord', 'y_coord', 'concentration']].copy()
        
        merged = pd.merge(
            df_parsed,
            df_csv_subset,
            on=['x_coord', 'y_coord'],
            how='inner',
            suffixes=('_parsed', '_csv')
        )
        
        self.assertGreater(len(merged), 0, "Should have matching coordinates between parsed and CSV data")
        
        tolerance = 1e-5
        concentration_match = (merged['concentration_parsed'] - merged['concentration_csv']).abs() < tolerance
        self.assertTrue(concentration_match.all(), "Concentrations should match within tolerance")


if __name__ == '__main__':
    unittest.main()
