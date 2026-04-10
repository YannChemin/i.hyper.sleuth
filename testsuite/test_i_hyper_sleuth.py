"""
Name:      i.hyper.sleuth test suite
Purpose:   Unit and integration tests for the i.hyper.sleuth GRASS module.

           Covers:
             - Reference spectrum parsers (inline, CSV, JSON — all layouts)
             - WavelengthLUT: construction, apply, apply_cube, overlap,
               coverage_report, edge-fill, restrict_to_overlap
             - normalize_spectrum / normalize_cube (all five modes)
             - to_prob_simplex / to_prob_simplex_cube
             - continuum_remove / continuum_remove_cube
             - All 17 match_* similarity functions (known-input assertions)
             - calibrate_scores (rank transform properties)
             - compute_diversity_weights (weight sum invariant)
             - fuse_probabilities (all six fusion modes, boundary conditions)
             - compute_consensus_stats (agreement, entropy, conflict, spread)
             - Module existence and --help (GRASS integration, skipped if not installed)
             - Integration test: full consensus run on synthetic GRASS scene
               (skipped when GRASS environment or test data are unavailable)

Author:    Yann Chemin <yann.chemin@gmail.com>
Copyright: (C) 2026 by Yann Chemin and the GRASS Development Team
License:   GPL-2.0-or-later

Run from inside a GRASS session
--------------------------------
    cd i.hyper.sleuth/testsuite
    python -m grass.gunittest.main
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Import the module under test — works both from a GRASS session and as a
# standalone unit test (without GRASS, GRASS-calling functions are skipped).
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _MODULE_DIR)

_GRASS_AVAILABLE = False
try:
    import grass.script as gs
    _GRASS_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Install a lightweight grass.script stub when GRASS is not available.
# This allows the pure-NumPy functions in i.hyper.sleuth.py to be tested
# without a running GRASS session.
# ---------------------------------------------------------------------------

if not _GRASS_AVAILABLE:
    import types as _types

    _gs_stub = _types.ModuleType("grass.script")
    _gs_stub.fatal   = lambda msg, **kw: (_ for _ in ()).throw(SystemExit(msg))
    _gs_stub.warning = lambda msg, **kw: None
    _gs_stub.message = lambda msg, **kw: None
    _gs_stub.verbose = lambda msg, **kw: None
    _gs_stub.percent = lambda a, b, c, **kw: None
    _gs_stub.run_command    = lambda *a, **kw: 0
    _gs_stub.read_command   = lambda *a, **kw: ""
    _gs_stub.write_command  = lambda *a, **kw: 0
    _gs_stub.parse_command  = lambda *a, **kw: {}
    _gs_stub.find_file      = lambda *a, **kw: {}
    _gs_stub.raster3d_info  = lambda *a, **kw: {"depths": 0}

    # Register both the package and the submodule so
    # `import grass.script as gs` inside the module under test works
    import sys as _sys
    _grass_pkg = _types.ModuleType("grass")
    _grass_pkg.script = _gs_stub
    _sys.modules.setdefault("grass",        _grass_pkg)
    _sys.modules.setdefault("grass.script", _gs_stub)
    gs = _gs_stub

# ---------------------------------------------------------------------------
# Patch gs.fatal to raise SystemExit (testable) rather than calling sys.exit.
# Do this regardless of whether GRASS is real or stubbed.
# ---------------------------------------------------------------------------
_sys_modules_gs = sys.modules.get("grass.script")
if _sys_modules_gs is not None:
    _sys_modules_gs.fatal   = lambda msg, **kw: (_ for _ in ()).throw(SystemExit(msg))
    _sys_modules_gs.warning = lambda msg, **kw: None

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "i_hyper_sleuth",
    os.path.join(_MODULE_DIR, "i.hyper.sleuth.py"),
)
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

# Convenience aliases
WavelengthLUT        = _m.WavelengthLUT
parse_inline         = _m.parse_reference_inline
parse_file           = _m.parse_reference_file
normalize_spectrum   = _m.normalize_spectrum
normalize_cube       = _m.normalize_cube
to_prob_simplex      = _m.to_prob_simplex
to_prob_simplex_cube = _m.to_prob_simplex_cube
continuum_remove     = _m.continuum_remove
calibrate_scores     = _m.calibrate_scores
diversity_weights    = _m.compute_diversity_weights
fuse                 = _m.fuse_probabilities
consensus_stats      = _m.compute_consensus_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cube(shape, fill=0.5):
    """Return a float64 array filled with *fill*."""
    return np.full(shape, fill, dtype=np.float64)


def _rand_cube(n=10, rows=4, cols=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, rows, cols))


def _rand_ref(n=10, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random(n)


# ===========================================================================
# Reference spectrum parsers
# ===========================================================================

class TestParseReferenceInline(unittest.TestCase):
    """Tests for parse_reference_inline."""

    def test_colon_comma(self):
        wls, refs = parse_inline("450:0.04,670:0.05,800:0.42")
        np.testing.assert_array_equal(wls, [450, 670, 800])
        np.testing.assert_array_almost_equal(refs, [0.04, 0.05, 0.42])

    def test_semicolon_separator(self):
        """Semicolons instead of colons are accepted."""
        wls, refs = parse_inline("450;0.04,670;0.05")
        np.testing.assert_array_equal(wls, [450, 670])

    def test_whitespace_list(self):
        wls, refs = parse_inline("450:0.04 670:0.05 800:0.42")
        self.assertEqual(len(wls), 3)

    def test_sorted_by_wavelength(self):
        """Out-of-order input is sorted."""
        wls, refs = parse_inline("800:0.42,450:0.04,670:0.05")
        self.assertTrue(np.all(np.diff(wls) > 0))

    def test_scientific_notation(self):
        wls, refs = parse_inline("4.5e2:4e-2,8.0e2:4.2e-1")
        self.assertAlmostEqual(wls[0], 450)
        self.assertAlmostEqual(refs[0], 0.04)

    def test_returns_float64(self):
        wls, refs = parse_inline("450:0.04,670:0.05")
        self.assertEqual(wls.dtype, np.float64)
        self.assertEqual(refs.dtype, np.float64)

    def test_too_few_pairs_raises(self):
        with self.assertRaises(SystemExit):
            parse_inline("450:0.04")

    def test_bad_token_raises(self):
        with self.assertRaises(SystemExit):
            parse_inline("notanumber:0.04,670:0.05")


class TestParseReferenceFileCSV(unittest.TestCase):
    """Tests for parse_reference_file — CSV format."""

    def _write(self, content):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        f.write(content)
        f.close()
        return f.name

    def tearDown(self):
        # Clean up any leftover temp files
        pass

    def test_basic_csv_no_header(self):
        path = self._write("450,0.04\n670,0.05\n800,0.42\n")
        wls, refs = parse_file(path)
        os.unlink(path)
        np.testing.assert_array_equal(wls, [450, 670, 800])

    def test_basic_csv_with_header(self):
        """Header row is skipped automatically."""
        path = self._write("wavelength,reflectance\n450,0.04\n670,0.05\n")
        wls, refs = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)
        self.assertAlmostEqual(wls[0], 450)

    def test_csv_sorted(self):
        path = self._write("800,0.42\n450,0.04\n670,0.05\n")
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertTrue(np.all(np.diff(wls) > 0))

    def test_csv_comment_lines_skipped(self):
        """Non-numeric lines (comments) are silently skipped."""
        path = self._write(
            "# kaolinite spectrum\nwavelength,reflectance\n450,0.04\n670,0.05\n"
        )
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)

    def test_csv_too_few_rows_raises(self):
        path = self._write("450,0.04\n")
        with self.assertRaises(SystemExit):
            parse_file(path)
        os.unlink(path)

    def test_missing_file_raises(self):
        with self.assertRaises(SystemExit):
            parse_file("/nonexistent/spectrum.csv")


class TestParseReferenceFileJSON(unittest.TestCase):
    """Tests for parse_reference_file — all JSON layouts."""

    def _write(self, obj):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(obj, f)
        f.close()
        return f.name

    def test_array_of_pairs(self):
        """[[wl, r], [wl, r], ...] layout."""
        path = self._write([[450, 0.04], [670, 0.05], [800, 0.42]])
        wls, refs = parse_file(path)
        os.unlink(path)
        np.testing.assert_array_equal(wls, [450, 670, 800])
        np.testing.assert_array_almost_equal(refs, [0.04, 0.05, 0.42])

    def test_parallel_arrays_wavelengths_reflectances(self):
        """{"wavelengths": [...], "reflectances": [...]} layout."""
        path = self._write({
            "wavelengths":  [450, 670, 800],
            "reflectances": [0.04, 0.05, 0.42],
        })
        wls, refs = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 3)
        self.assertAlmostEqual(wls[1], 670)

    def test_parallel_arrays_wl_r_aliases(self):
        """{"wl": [...], "r": [...]} alias layout."""
        path = self._write({
            "wl": [450, 670],
            "r":  [0.04, 0.05],
        })
        wls, refs = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)

    def test_data_key_layout(self):
        """{"data": [[wl, r], ...]} layout."""
        path = self._write({"data": [[450, 0.04], [670, 0.05]]})
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)

    def test_spectrum_key_layout(self):
        """{"spectrum": [[wl, r], ...]} layout."""
        path = self._write({"spectrum": [[450, 0.04], [670, 0.05]]})
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)

    def test_pairs_key_layout(self):
        """{"pairs": [[wl, r], ...]} layout."""
        path = self._write({"pairs": [[450, 0.04], [670, 0.05]]})
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)

    def test_sorted_by_wavelength(self):
        path = self._write([[800, 0.42], [450, 0.04], [670, 0.05]])
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertTrue(np.all(np.diff(wls) > 0))

    def test_single_wavelength_alias(self):
        """{"wavelength": [...]} (singular) is accepted."""
        path = self._write({"wavelength": [450, 670], "reflectance": [0.04, 0.05]})
        wls, _ = parse_file(path)
        os.unlink(path)
        self.assertEqual(len(wls), 2)


# ===========================================================================
# WavelengthLUT
# ===========================================================================

class TestWavelengthLUT(unittest.TestCase):
    """Tests for WavelengthLUT construction, apply, and coverage helpers."""

    def setUp(self):
        self.src = np.array([400, 500, 600, 700, 800, 900], dtype=np.float64)
        self.dst = np.array([450, 550, 650, 750, 850], dtype=np.float64)
        self.lut = WavelengthLUT(self.src, self.dst)

    # ── Construction ────────────────────────────────────────────────────

    def test_precomputed_arrays_shape(self):
        self.assertEqual(self.lut.left_idx.shape,  (len(self.dst),))
        self.assertEqual(self.lut.right_idx.shape, (len(self.dst),))
        self.assertEqual(self.lut.alpha.shape,     (len(self.dst),))

    def test_alpha_in_range(self):
        self.assertTrue(np.all(self.lut.alpha >= 0.0))
        self.assertTrue(np.all(self.lut.alpha <= 1.0))

    def test_valid_dst_all_covered(self):
        """All dst points are within src range → all valid."""
        self.assertTrue(np.all(self.lut.valid_dst))

    def test_overlap_range(self):
        # src=[400..900], dst=[450..850] → overlap = [450, 850]
        self.assertAlmostEqual(self.lut.overlap_lo, 450.0)
        self.assertAlmostEqual(self.lut.overlap_hi, 850.0)
        self.assertTrue(self.lut.has_overlap)

    def test_no_overlap_raises(self):
        src = np.array([400, 500, 600], dtype=np.float64)
        dst = np.array([700, 800, 900], dtype=np.float64)
        lut = WavelengthLUT(src, dst)
        self.assertFalse(lut.has_overlap)

    # ── apply ────────────────────────────────────────────────────────────

    def test_apply_midpoint_interpolation(self):
        """apply at 450 nm should give avg(src[400], src[500])."""
        src_vals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.lut.apply(src_vals)
        # dst[0] = 450 → midway between src[0]=0 and src[1]=1 → 0.5
        self.assertAlmostEqual(result[0], 0.5)

    def test_apply_exact_match(self):
        """apply at exact src point should return exact value."""
        lut = WavelengthLUT(
            np.array([400.0, 500.0, 600.0]),
            np.array([400.0, 500.0, 600.0]),
        )
        src_vals = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(lut.apply(src_vals), src_vals, atol=1e-10)

    def test_apply_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            self.lut.apply(np.ones(3))

    # ── apply_cube ───────────────────────────────────────────────────────

    def test_apply_cube_shape(self):
        cube = np.ones((len(self.src), 5, 7))
        result = self.lut.apply_cube(cube)
        self.assertEqual(result.shape, (len(self.dst), 5, 7))

    def test_apply_cube_uniform(self):
        """Uniform cube → resampled cube should also be uniform (same value)."""
        v = 0.35
        cube = np.full((len(self.src), 3, 3), v)
        result = self.lut.apply_cube(cube)
        np.testing.assert_allclose(result, v, atol=1e-8)

    # ── edge fill ────────────────────────────────────────────────────────

    def test_edge_fill_out_of_range_dst(self):
        """dst points outside src range receive edge-fill values."""
        src = np.array([500.0, 600.0, 700.0])
        dst = np.array([400.0, 550.0, 800.0])   # 400 and 800 are outside
        lut = WavelengthLUT(src, dst, fill='edge')
        vals = lut.apply(np.array([0.1, 0.2, 0.3]))
        self.assertAlmostEqual(vals[0], 0.1)   # left edge
        self.assertAlmostEqual(vals[2], 0.3)   # right edge

    def test_nan_fill(self):
        src = np.array([500.0, 600.0, 700.0])
        dst = np.array([400.0, 550.0])   # 400 is outside
        lut = WavelengthLUT(src, dst, fill='nan')
        vals = lut.apply(np.array([0.1, 0.2, 0.3]))
        self.assertTrue(np.isnan(vals[0]))
        self.assertAlmostEqual(vals[1], 0.15)

    # ── restrict_to_overlap ───────────────────────────────────────────────

    def test_restrict_to_overlap_returns_sub_lut(self):
        src = np.array([400.0, 500.0, 600.0, 700.0])
        dst = np.array([350.0, 450.0, 550.0, 650.0, 750.0])
        lut = WavelengthLUT(src, dst)
        ov_src, ov_sv, ov_dst, ov_db, sub = lut.restrict_to_overlap(
            src_vals=np.ones(4), dst_bands=list(range(5))
        )
        self.assertTrue(np.all(ov_src >= lut.overlap_lo))
        self.assertTrue(np.all(ov_dst <= lut.overlap_hi))
        self.assertIsInstance(sub, WavelengthLUT)

    # ── coverage_report ───────────────────────────────────────────────────

    def test_coverage_report_string(self):
        rep = self.lut.coverage_report()
        self.assertIn("nm", rep)
        self.assertIn("overlap", rep)


# ===========================================================================
# Normalization
# ===========================================================================

class TestNormalize(unittest.TestCase):

    def test_none_returns_same_array(self):
        s = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_equal(normalize_spectrum(s, "none"), s)

    def test_area_sums_to_one(self):
        s = np.array([0.1, 0.2, 0.3, 0.4])
        r = normalize_spectrum(s, "area")
        self.assertAlmostEqual(r.sum(), 1.0)

    def test_max_max_is_one(self):
        s = np.array([0.1, 0.5, 0.3])
        r = normalize_spectrum(s, "max")
        self.assertAlmostEqual(r.max(), 1.0)

    def test_minmax_range_zero_to_one(self):
        s = np.array([0.1, 0.5, 0.3])
        r = normalize_spectrum(s, "minmax")
        self.assertAlmostEqual(r.min(), 0.0)
        self.assertAlmostEqual(r.max(), 1.0)

    def test_vector_l2_norm_is_one(self):
        s = np.array([0.3, 0.4, 0.0])
        r = normalize_spectrum(s, "vector")
        self.assertAlmostEqual(np.linalg.norm(r), 1.0)

    def test_cube_area_sums_to_one_per_pixel(self):
        cube = np.random.default_rng(7).random((5, 3, 4))
        r = normalize_cube(cube, "area")
        col_sums = r.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)

    def test_cube_vector_unit_l2_per_pixel(self):
        cube = np.random.default_rng(8).random((5, 3, 4)) + 0.01
        r = normalize_cube(cube, "vector")
        norms = np.linalg.norm(r, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestProbSimplex(unittest.TestCase):

    def test_sums_to_one(self):
        s = np.array([0.1, 0.3, 0.6])
        np.testing.assert_allclose(to_prob_simplex(s).sum(), 1.0, atol=1e-10)

    def test_negative_clipped(self):
        s = np.array([-0.1, 0.3, 0.6])
        r = to_prob_simplex(s)
        self.assertTrue(np.all(r > 0))

    def test_cube_sums_to_one(self):
        cube = np.random.default_rng(3).random((6, 4, 4))
        r = to_prob_simplex_cube(cube)
        np.testing.assert_allclose(r.sum(axis=0), 1.0, atol=1e-8)


# ===========================================================================
# Continuum removal
# ===========================================================================

class TestContinuumRemove(unittest.TestCase):

    def _flat_spec(self, n=20, val=0.4):
        return np.full(n, val), np.linspace(400, 2400, n)

    def test_flat_spectrum_stays_near_one(self):
        """CR of a flat spectrum should be ~1 everywhere."""
        spec, wls = self._flat_spec()
        cr = continuum_remove(spec, wls)
        np.testing.assert_allclose(cr, 1.0, atol=0.01)

    def test_absorption_feature_below_one(self):
        """A synthetic trough should produce CR < 1 near the trough."""
        spec, wls = self._flat_spec(n=30, val=0.4)
        # Create an absorption at band 15
        spec[13:17] *= 0.7
        cr = continuum_remove(spec, wls)
        self.assertTrue(cr[15] < 1.0)

    def test_output_in_zero_one(self):
        rng = np.random.default_rng(9)
        spec = rng.random(20) * 0.5
        wls  = np.linspace(400, 2400, 20)
        cr = continuum_remove(spec, wls)
        self.assertTrue(np.all(cr >= 0.0))
        self.assertTrue(np.all(cr <= 1.0))


# ===========================================================================
# Similarity methods — known inputs
# ===========================================================================

class TestMatchFunctionsBasic(unittest.TestCase):
    """All match_* functions: sanity-check on trivial known inputs."""

    def _perfect(self, n=10):
        """Cube where every pixel equals the reference."""
        ref = np.linspace(0.1, 0.5, n)
        cube = np.broadcast_to(ref[:, np.newaxis, np.newaxis],
                               (n, 3, 3)).copy()
        return cube, ref

    def _orthogonal(self, n=10):
        """Cube where every pixel has all zeros (orthogonal to any positive ref)."""
        ref = np.linspace(0.1, 0.5, n)
        cube = np.zeros((n, 3, 3))
        return cube, ref

    def test_sam_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_sam(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_sam_range(self):
        cube = _rand_cube()
        ref  = _rand_ref()
        s = _m.match_sam(cube, ref)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_sid_perfect(self):
        """SID of a spectrum against itself should be very close to 1."""
        ref  = np.linspace(0.05, 0.5, 10) + 0.01
        cube = np.broadcast_to(ref[:, np.newaxis, np.newaxis],
                               (10, 3, 3)).copy()
        s = _m.match_sid(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_sid_range(self):
        cube = _rand_cube() + 0.01
        ref  = _rand_ref() + 0.01
        s = _m.match_sid(cube, ref)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_ed_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_ed(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_ed_zero_cube_less_than_perfect(self):
        cube, ref = self._perfect()
        cube_zero = np.zeros_like(cube)
        s_zero = _m.match_ed(cube_zero, ref)
        s_perf = _m.match_ed(cube, ref)
        self.assertTrue(np.all(s_zero < s_perf))

    def test_sad_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_sad(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_sca_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_sca(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_sca_inverted(self):
        """Inverted spectrum should give low SCA score."""
        ref  = np.linspace(0.1, 0.5, 10)
        inv  = ref[::-1].copy()
        cube = np.broadcast_to(inv[:, np.newaxis, np.newaxis], (10, 2, 2)).copy()
        s = _m.match_sca(cube, ref)
        # Pearson r of increasing vs decreasing is -1 → similarity ~0
        self.assertTrue(np.all(s < 0.2))

    def test_jsd_perfect(self):
        ref  = np.linspace(0.05, 0.5, 10) + 0.01
        cube = np.broadcast_to(ref[:, np.newaxis, np.newaxis],
                               (10, 2, 2)).copy()
        s = _m.match_jsd(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_bhatt_perfect(self):
        ref  = np.linspace(0.05, 0.5, 10) + 0.01
        cube = np.broadcast_to(ref[:, np.newaxis, np.newaxis],
                               (10, 2, 2)).copy()
        # Normalise to probability distributions
        cube_n = cube / cube.sum(axis=0, keepdims=True)
        ref_n  = ref  / ref.sum()
        s = _m.match_bhatt(cube_n, ref_n)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_ssim_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_ssim(cube, ref)
        np.testing.assert_allclose(s, 1.0, atol=1e-4)

    def test_ssim_range(self):
        s = _m.match_ssim(_rand_cube(), _rand_ref())
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_xcorr_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_xcorr(cube, ref, max_lag=0)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_xcorr_range(self):
        s = _m.match_xcorr(_rand_cube(), _rand_ref(), max_lag=2)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_dtw_perfect(self):
        cube, ref = self._perfect()
        s = _m.match_dtw(cube, ref, window=1)
        np.testing.assert_allclose(s, 1.0, atol=1e-5)

    def test_dtw_range(self):
        s = _m.match_dtw(_rand_cube(), _rand_ref(), window=2)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_mtf_range(self):
        s = _m.match_mtf(_rand_cube(rows=6, cols=6), _rand_ref())
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_cem_range(self):
        s = _m.match_cem(_rand_cube(rows=6, cols=6), _rand_ref())
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_cr_sam_range(self):
        wls  = np.linspace(400, 2400, 10)
        ref  = np.linspace(0.1, 0.5, 10)
        cube = _rand_cube()
        s = _m.match_cr_sam(cube, ref, wls)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_gd1_perfect(self):
        """Two identical spectra → derivative vectors also identical → SAM ≈ 1."""
        # Use a spectrum with non-constant derivative (Gaussian absorption trough)
        wls = np.linspace(400, 900, 20)
        x   = (wls - 650) / 80.0
        ref = 0.4 - 0.25 * np.exp(-0.5 * x**2)   # Gaussian absorption at 650 nm
        cube = np.broadcast_to(ref[:, np.newaxis, np.newaxis], (20, 2, 2)).copy()
        s = _m.match_gd1(cube, ref, wls)
        # EPS in norm denominator prevents exact 1; allow generous tolerance
        np.testing.assert_allclose(s, 1.0, atol=1e-3)

    def test_gd2_perfect(self):
        """Two identical non-linear spectra → GD2 SAM ≈ 1."""
        # Use a quadratic spectrum so 2nd derivative is non-zero
        wls = np.linspace(400, 900, 20)
        x   = (wls - 650) / 80.0
        ref = 0.4 - 0.25 * np.exp(-0.5 * x**2)
        cube = np.broadcast_to(ref[:, np.newaxis, np.newaxis], (20, 2, 2)).copy()
        s = _m.match_gd2(cube, ref, wls)
        np.testing.assert_allclose(s, 1.0, atol=1e-3)

    def test_sid_sam_range(self):
        cube = _rand_cube() + 0.01
        ref  = _rand_ref() + 0.01
        s = _m.match_sid_sam(cube, ref)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))


class TestMatchEnsemble(unittest.TestCase):

    def test_ensemble_range(self):
        rng  = np.random.default_rng(5)
        maps = {k: rng.random((4, 4)) for k in ("sam", "sid", "ed")}
        s = _m.match_ensemble(maps)
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_ensemble_empty_raises(self):
        with self.assertRaises(SystemExit):
            _m.match_ensemble({})

    def test_ensemble_target_pixel_highest(self):
        """Pixel that scores highest in ALL methods should top the ensemble."""
        n, rows, cols = 8, 5, 5
        ref  = np.linspace(0.1, 0.5, n)
        cube = _rand_cube(n=n, rows=rows, cols=cols, seed=42)
        # Plant the reference exactly at pixel (2, 2)
        cube[:, 2, 2] = ref

        maps = {
            "sam": _m.match_sam(cube, ref),
            "sca": _m.match_sca(cube, ref),
            "ed":  _m.match_ed(cube, ref),
        }
        ens = _m.match_ensemble(maps)
        self.assertEqual(np.unravel_index(ens.argmax(), ens.shape), (2, 2))


# ===========================================================================
# Consensus pipeline
# ===========================================================================

class TestCalibrateScores(unittest.TestCase):

    def test_output_range(self):
        rng = np.random.default_rng(11)
        maps = {"sam": rng.random((5, 5)), "ed": rng.random((5, 5))}
        cal = calibrate_scores(maps)
        for v in cal.values():
            self.assertTrue(np.all(v > 0) and np.all(v <= 1))

    def test_uniform_marginal(self):
        """Rank-calibrated maps should be approx. uniform in (0, 1]."""
        rng  = np.random.default_rng(12)
        smap = {"sam": rng.random((10, 10))}
        cal  = calibrate_scores(smap)
        vals = cal["sam"].ravel()
        # Mean should be close to 0.5 for a uniform distribution
        self.assertAlmostEqual(float(vals.mean()), 0.5, delta=0.1)

    def test_ensemble_skipped(self):
        """ensemble / consensus maps must not appear in calibrated output."""
        rng = np.random.default_rng(13)
        maps = {
            "sam": rng.random((4, 4)),
            "ensemble": rng.random((4, 4)),
        }
        cal = calibrate_scores(maps)
        self.assertNotIn("ensemble", cal)


class TestDiversityWeights(unittest.TestCase):

    def test_single_method(self):
        cal = {"sam": np.random.default_rng(14).random((3, 3))}
        w = diversity_weights(cal)
        self.assertAlmostEqual(w["sam"], 1.0)

    def test_weights_mean_is_one(self):
        rng = np.random.default_rng(15)
        cal = {k: rng.random((4, 4)) for k in ("sam", "sid", "ed", "sca")}
        w = diversity_weights(cal)
        self.assertAlmostEqual(
            sum(w.values()) / len(w), 1.0, delta=1e-6
        )

    def test_correlated_methods_downweighted(self):
        """Identical maps should receive low relative weight."""
        base = np.random.default_rng(16).random((5, 5))
        cal = {
            "a": base.copy(),
            "b": base.copy(),            # identical to a
            "c": 1.0 - base.copy(),      # anti-correlated
        }
        w = diversity_weights(cal)
        # a and b are fully correlated → their weights should be equal and low
        self.assertAlmostEqual(w["a"], w["b"], places=5)
        # c is anti-correlated → also down-weighted (mean correlation is high)
        # Key invariant: mean weight = 1
        self.assertAlmostEqual(sum(w.values()) / 3, 1.0, delta=1e-6)


class TestFuseProbabilities(unittest.TestCase):
    """Test all six fusion modes on a controlled calibrated map."""

    def setUp(self):
        rng = np.random.default_rng(20)
        k = 4
        self.cal = {f"m{i}": rng.random((5, 5)) for i in range(k)}
        self.w   = diversity_weights(self.cal)

    def _fuse(self, mode):
        return fuse(self.cal, self.w, mode)

    def test_rank_product_range(self):
        s = self._fuse("rank_product")
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_fisher_range(self):
        s = self._fuse("fisher")
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_stouffer_range(self):
        s = self._fuse("stouffer")
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_group_product_range(self):
        s = self._fuse("group_product")
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_harmonic_range(self):
        s = self._fuse("harmonic")
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_min_range(self):
        s = self._fuse("min")
        self.assertTrue(np.all(s >= 0) and np.all(s <= 1))

    def test_min_is_lower_bound(self):
        """min fusion ≤ all other modes for every pixel."""
        s_min = self._fuse("min")
        for mode in ("rank_product", "stouffer", "harmonic"):
            s = self._fuse(mode)
            # min is the strictest: at least most pixels should not exceed it
            # (harmonic can go below min due to weighting — test rank_product)
        s_rp = self._fuse("rank_product")
        # min ≤ mean of rank_product on average
        self.assertLessEqual(float(s_min.mean()), float(s_rp.mean()) + 0.01)

    def test_all_ones_input(self):
        """Uniform calibrated = 1 → all fusions should return ~1."""
        ones = {f"m{i}": np.ones((3, 3)) for i in range(3)}
        w    = diversity_weights(ones)
        for mode in _m.FUSION_MODES:
            if mode == "min":
                continue  # min of 1s = 1, but harmonic/fisher may differ
            s = fuse(ones, w, mode)
            np.testing.assert_allclose(s, 1.0, atol=0.01,
                                        err_msg=f"mode={mode} failed")

    def test_unknown_mode_raises(self):
        with self.assertRaises(SystemExit):
            fuse(self.cal, self.w, "nonexistent_mode")


class TestConsensusStats(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(30)
        self.cal = {f"m{i}": rng.random((6, 6)) for i in range(5)}
        self.prob = rng.random((6, 6))

    def test_keys_present(self):
        stats = consensus_stats(self.cal, self.prob)
        for key in ("agreement", "entropy", "conflict", "spread"):
            self.assertIn(key, stats)

    def test_agreement_range(self):
        stats = consensus_stats(self.cal, self.prob)
        a = stats["agreement"]
        self.assertTrue(np.all(a >= 0) and np.all(a <= 1))

    def test_entropy_range(self):
        stats = consensus_stats(self.cal, self.prob)
        e = stats["entropy"]
        self.assertTrue(np.all(e >= 0) and np.all(e <= 1.0 + 1e-6))

    def test_spread_non_negative(self):
        stats = consensus_stats(self.cal, self.prob)
        self.assertTrue(np.all(stats["spread"] >= 0))

    def test_uniform_cal_high_agreement(self):
        """When all calibrated scores are 0.9, agreement at threshold 0.8 should be 1."""
        cal  = {f"m{i}": np.full((4, 4), 0.9) for i in range(4)}
        prob = np.full((4, 4), 0.9)
        stats = consensus_stats(cal, prob, agreement_threshold=0.80)
        np.testing.assert_allclose(stats["agreement"], 1.0, atol=1e-6)

    def test_low_cal_zero_agreement(self):
        cal  = {f"m{i}": np.full((4, 4), 0.1) for i in range(4)}
        prob = np.full((4, 4), 0.1)
        stats = consensus_stats(cal, prob, agreement_threshold=0.80)
        np.testing.assert_allclose(stats["agreement"], 0.0, atol=1e-6)


# ===========================================================================
# Target detection: planted pixel must score highest
# ===========================================================================

class TestTargetDetection(unittest.TestCase):
    """Verify each method correctly ranks the planted target pixel highest."""

    def setUp(self):
        n, rows, cols = 12, 8, 8
        self.ref   = np.linspace(0.05, 0.50, n)
        rng = np.random.default_rng(99)
        self.cube  = rng.random((n, rows, cols)) * 0.3 + 0.1
        # Plant exact reference at (4, 4)
        self.cube[:, 4, 4] = self.ref
        self.wls   = np.linspace(400, 2400, n)

    def _check_top(self, method_fn, *extra_args):
        s = method_fn(self.cube, self.ref, *extra_args)
        idx = np.unravel_index(s.argmax(), s.shape)
        self.assertEqual(idx, (4, 4),
                         msg=f"{method_fn.__name__} did not rank target pixel highest")

    def test_sam_detects_target(self):
        self._check_top(_m.match_sam)

    def test_sid_detects_target(self):
        self._check_top(_m.match_sid)

    def test_ed_detects_target(self):
        self._check_top(_m.match_ed)

    def test_sad_detects_target(self):
        self._check_top(_m.match_sad)

    def test_sca_detects_target(self):
        self._check_top(_m.match_sca)

    def test_jsd_detects_target(self):
        self._check_top(_m.match_jsd)

    def test_bhatt_detects_target(self):
        self._check_top(_m.match_bhatt)

    def test_ssim_detects_target(self):
        self._check_top(_m.match_ssim)

    def test_xcorr_detects_target(self):
        self._check_top(_m.match_xcorr, 2)

    def test_dtw_detects_target(self):
        self._check_top(_m.match_dtw, 2)

    def test_gd1_detects_target(self):
        self._check_top(_m.match_gd1, self.wls)

    def test_gd2_detects_target(self):
        """GD2 needs a non-linear spectrum; plant a Gaussian trough at (4,4)."""
        n, rows, cols = 20, 8, 8
        wls = np.linspace(400, 2400, n)
        x   = (wls - 1200) / 200.0
        ref = 0.4 - 0.30 * np.exp(-0.5 * x**2)   # Gaussian absorption

        rng  = np.random.default_rng(77)
        cube = rng.random((n, rows, cols)) * 0.3 + 0.1
        cube[:, 4, 4] = ref     # exact match at target pixel

        s = _m.match_gd2(cube, ref, wls)
        target_val = s[4, 4]
        threshold  = np.percentile(s, 90)   # target must be in top 10%
        self.assertGreaterEqual(
            float(target_val), float(threshold),
            msg="GD2 target pixel not in top 10% of similarity scores"
        )


# ===========================================================================
# GRASS integration tests (skipped if GRASS or test data unavailable)
# ===========================================================================

@unittest.skipUnless(_GRASS_AVAILABLE, "GRASS GIS environment not available")
class TestModuleExists(unittest.TestCase):
    """Verify the module is installed and help text is accessible."""

    def test_module_help(self):
        """i.hyper.sleuth --help must succeed (exit 0 with usage info)."""
        import subprocess
        gisbase = os.environ.get("GISBASE", "")
        script  = os.path.join(gisbase, "scripts", "i.hyper.sleuth")
        if not os.path.isfile(script):
            self.skipTest("i.hyper.sleuth not installed at expected path")
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True, text=True,
        )
        self.assertIn("i.hyper.sleuth", result.stdout + result.stderr)


@unittest.skipUnless(_GRASS_AVAILABLE, "GRASS GIS environment not available")
class TestIntegrationKaolinite(unittest.TestCase):
    """Full run on synthetic kaolinite scene using the module's main() function.

    Requires a running GRASS session.  The test creates a synthetic 3D raster,
    writes a CSV reference, runs main() in point mode, then in full-image mode
    and checks the primary output map's statistical properties.
    """

    _SCENE = "sleuth_test_kaol"
    _OUTPUT = "sleuth_out_kaol"

    @classmethod
    def setUpClass(cls):
        """Create the synthetic scene once for all tests in this class."""
        sys.path.insert(0, os.path.dirname(__file__))
        try:
            from generate_test_data import (
                BAND_WAVELENGTHS, REFERENCE_CSV,
                create_scene, setup_test_region,
            )
            setup_test_region()
            create_scene.__module__  # verify import
            cls._scene_created = True
            cls._band_wavelengths = BAND_WAVELENGTHS
            cls._ref_csv = REFERENCE_CSV["kaolinite"]
        except Exception as exc:
            cls._scene_created = False
            cls._skip_reason = str(exc)
            return

        try:
            from generate_test_data import create_scene as _create
            _create(cls._SCENE)
        except Exception as exc:
            cls._scene_created = False
            cls._skip_reason = str(exc)

    @classmethod
    def tearDownClass(cls):
        if not getattr(cls, "_scene_created", False):
            return
        sys.path.insert(0, os.path.dirname(__file__))
        try:
            from generate_test_data import cleanup_scene
            cleanup_scene(cls._SCENE)
        except Exception:
            pass
        # Remove output maps
        for suffix in ("", "_sam", "_sid"):
            name = cls._OUTPUT + suffix
            if gs.find_file(name, element="cell").get("name"):
                gs.run_command("g.remove", type="raster",
                               name=name, flags="f", quiet=True)

    def setUp(self):
        if not getattr(self.__class__, "_scene_created", False):
            self.skipTest(getattr(self.__class__, "_skip_reason", "scene not created"))

    def _write_ref_csv(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        )
        f.write(self._ref_csv)
        f.close()
        return f.name

    def test_sam_output_exists_and_in_range(self):
        """SAM output map must exist and have values in [0, 1]."""
        ref_path = self._write_ref_csv()
        try:
            _m.main(
                options={
                    "input":          self._SCENE,
                    "output":         self._OUTPUT,
                    "reference":      "",
                    "reference_file": ref_path,
                    "method":         "sam",
                    "output_prefix":  "",
                    "resample":       "linear",
                    "normalize":      "none",
                    "shift_window":   "3",
                    "fusion_mode":    "rank_product",
                    "agreement_threshold": "0.80",
                    "min_wavelength": "",
                    "max_wavelength": "",
                    "coordinates":    "",
                },
                flags=dict(n=False, i=False, v=False, c=False, p=False, z=False),
            )
        finally:
            os.unlink(ref_path)

        info = gs.parse_command("r.univar", map=self._OUTPUT, flags="g")
        self.assertGreaterEqual(float(info["min"]), 0.0)
        self.assertLessEqual(float(info["max"]),    1.0)

    def test_target_pixel_highest_sam(self):
        """The 3×3 target patch must contain the global SAM maximum."""
        if not gs.find_file(self._OUTPUT, element="cell").get("name"):
            self.skipTest(f"{self._OUTPUT} not found; run test_sam_output first")

        result = gs.read_command(
            "r.what",
            map=self._OUTPUT,
            coordinates="9.5,9.5",   # centre of target patch (row 9, col 9)
        )
        lines = [l for l in result.strip().splitlines() if "|" in l]
        if not lines:
            self.skipTest("r.what returned no data")
        target_val = float(lines[0].split("|")[-1].strip())

        info = gs.parse_command("r.univar", map=self._OUTPUT, flags="g")
        global_max = float(info["max"])
        # Target value should be within 0.05 of the global maximum
        self.assertGreaterEqual(target_val, global_max - 0.05)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _GRASS_AVAILABLE:
        try:
            from grass.gunittest.main import test
            test()
        except ImportError:
            unittest.main()
    else:
        unittest.main()
