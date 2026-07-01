import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.eda import run_eda

def test_eda():
    df, col_types, problem_type = ingest("data/Titanic-Dataset.csv", target_col="Survived")
    plot_paths, stats = run_eda(df, col_types, "Survived", problem_type)

    assert stats["total_rows"] == 891
    assert stats["total_columns"] == 12
    assert stats["target_unique_values"] == 2
    assert isinstance(plot_paths, dict)
    assert len(plot_paths) > 0

    print("\nAll EDA tests passed.")

test_eda()