import os
import json
import tempfile
import pytest
from validate_results import validate_experiment_results

def test_validator_with_valid_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_json = {
            "precision_bits": 16,
            "dataset": "wikitext-2",
            "seed": 42,
            "logit_cosine_similarity": 0.999,
            "perplexity": 24.0
        }
        with open(os.path.join(tmpdir, "16bit.json"), "w") as f:
            json.dump(sample_json, f)

        assert validate_experiment_results(tmpdir) is True

def test_validator_detects_out_of_bounds_cossim():
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_json = {
            "precision_bits": 16,
            "dataset": "wikitext-2",
            "seed": 42,
            "logit_cosine_similarity": 1.05,
            "perplexity": 24.0
        }
        with open(os.path.join(tmpdir, "bad.json"), "w") as f:
            json.dump(bad_json, f)

        assert validate_experiment_results(tmpdir) is False
