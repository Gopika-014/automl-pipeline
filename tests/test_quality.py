import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check

def test_quality():
    df, col_types, problem_type = ingest("data/Titanic-dataset.csv", target_col="Survived")
    quality_report = run_quality_check(df, col_types, "Survived", problem_type)

    # Score should be between 0 and 100
    assert 0 <= quality_report["quality_score"] <= 100

    # Should have warnings list
    assert isinstance(quality_report["warnings"], list)

    # Should detect missing values (Titanic has missing Age and Cabin)
    assert len(quality_report["missing_values"]) > 0

    # Should detect class distribution
    assert quality_report["class_distribution"] is not None

    print("\nAll quality tests passed.")

test_quality()