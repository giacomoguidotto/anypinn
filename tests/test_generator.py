"""Tests for the variant generator module."""

from __future__ import annotations

from anypinn.cli._generator import extract_variants


class TestExtractVariants:
    """Tests for extract_variants()."""

    def test_no_markers_returns_unchanged(self) -> None:
        source = "x = 1\ny = 2\n"
        assert extract_variants(source, {}) == source

    def test_selects_first_variant(self) -> None:
        source = """\
before
# --- VARIANT: source/synthetic ---
validation_synthetic = "syn"
# --- VARIANT: source/csv ---
validation_csv = "csv"
# --- END VARIANT ---
after
"""
        result = extract_variants(source, {"source": "synthetic"})
        assert 'validation = "syn"' in result
        assert "csv" not in result
        assert "before" in result
        assert "after" in result

    def test_selects_second_variant(self) -> None:
        source = """\
before
# --- VARIANT: source/synthetic ---
validation_synthetic = "syn"
# --- VARIANT: source/csv ---
validation_csv = "csv"
# --- END VARIANT ---
after
"""
        result = extract_variants(source, {"source": "csv"})
        assert 'validation = "csv"' in result
        assert "syn" not in result

    def test_removes_marker_comments(self) -> None:
        source = """\
# --- VARIANT: source/synthetic ---
x_synthetic = 1
# --- VARIANT: source/csv ---
x_csv = 2
# --- END VARIANT ---
"""
        result = extract_variants(source, {"source": "synthetic"})
        assert "VARIANT" not in result
        assert "END VARIANT" not in result

    def test_strips_suffix_from_identifiers(self) -> None:
        source = """\
# --- VARIANT: source/synthetic ---
validation_synthetic = {}
def create_data_module_synthetic(hp):
    return DataModule(validation=validation_synthetic)
# --- VARIANT: source/csv ---
validation_csv = {}
def create_data_module_csv(hp):
    return DataModule(validation=validation_csv)
# --- END VARIANT ---
"""
        result = extract_variants(source, {"source": "synthetic"})
        assert "validation_synthetic" not in result
        assert "create_data_module_synthetic" not in result
        assert "validation = {}" in result
        assert "def create_data_module(hp):" in result
        assert "validation=validation)" in result

    def test_multiple_axes_independent(self) -> None:
        source = """\
# --- VARIANT: source/synthetic ---
data_synthetic = "gen"
# --- VARIANT: source/csv ---
data_csv = "load"
# --- END VARIANT ---
shared = True
# --- VARIANT: direction/forward ---
params_forward = {}
# --- VARIANT: direction/inverse ---
params_inverse = {"beta": 1}
# --- END VARIANT ---
"""
        result = extract_variants(source, {"source": "csv", "direction": "inverse"})
        assert 'data = "load"' in result
        assert "gen" not in result
        assert 'params = {"beta": 1}' in result
        assert "forward" not in result
        assert "shared = True" in result

    def test_preserves_indentation(self) -> None:
        source = """\
def func():
    # --- VARIANT: source/synthetic ---
    x_synthetic = 1
    # --- VARIANT: source/csv ---
    x_csv = 2
    # --- END VARIANT ---
    return x
"""
        result = extract_variants(source, {"source": "synthetic"})
        assert "    x = 1" in result

    def test_multiline_variant_block(self) -> None:
        source = """\
# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp):
    gen_props = ODEProperties(
        ode=SIR_unscaled,
        y0=torch.tensor([N_POP - 1, 1]),
    )
    return SIRInvDataModule(hp=hp, gen_props=gen_props)
# --- VARIANT: source/csv ---
def create_data_module_csv(hp):
    return SIRInvDataModule(hp=hp)
# --- END VARIANT ---
"""
        result = extract_variants(source, {"source": "synthetic"})
        assert "def create_data_module(hp):" in result
        assert "gen_props = ODEProperties(" in result
        assert "gen_props=gen_props)" in result

    def test_no_matching_axis_keeps_nothing(self) -> None:
        source = """\
before
# --- VARIANT: source/synthetic ---
x_synthetic = 1
# --- VARIANT: source/csv ---
x_csv = 2
# --- END VARIANT ---
after
"""
        result = extract_variants(source, {"other": "value"})
        assert "x" not in result.replace("after", "").replace("before", "")
        assert "before" in result
        assert "after" in result

    def test_suffix_not_stripped_from_unrelated_words(self) -> None:
        """Ensure _synthetic suffix is only stripped from identifiers, not arbitrary text."""
        source = """\
# --- VARIANT: source/synthetic ---
# Generate synthetic data using true parameter values
data_synthetic = generate()
# --- VARIANT: source/csv ---
data_csv = load()
# --- END VARIANT ---
"""
        result = extract_variants(source, {"source": "synthetic"})
        # "synthetic" in the comment should remain (it's not a _synthetic suffix)
        assert "synthetic data" in result
        # But data_synthetic should become data
        assert "data = generate()" in result

    def test_empty_selections_removes_all_variant_blocks(self) -> None:
        source = """\
shared = True
# --- VARIANT: source/synthetic ---
x_synthetic = 1
# --- END VARIANT ---
"""
        result = extract_variants(source, {})
        assert "shared = True" in result
        assert "x" not in result.replace("shared", "")

    def test_preserves_blank_lines_between_sections(self) -> None:
        source = """\
before

# --- VARIANT: source/synthetic ---
x_synthetic = 1
# --- VARIANT: source/csv ---
x_csv = 2
# --- END VARIANT ---

after
"""
        result = extract_variants(source, {"source": "synthetic"})
        assert "before" in result
        assert "after" in result
        assert "x = 1" in result
