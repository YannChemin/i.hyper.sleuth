#!/usr/bin/env python
##############################################################################
# MODULE:    i.hyper.sleuth
# AUTHOR(S): Hyperspectral Spectral Target Detection
# PURPOSE:   Find pixels in a hyperspectral 3D raster that best match a
#            reference spectrum using a comprehensive set of signal-analysis,
#            information-theoretic, morphological, and subpixel-detection
#            similarity metrics.
# COPYRIGHT: (C) 2026 by the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

# %module
# % description: Spectral target detection — find pixels matching a reference spectrum in a hyperspectral 3D raster
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: spectroscopy
# % keyword: target detection
# % keyword: spectral matching
# % keyword: similarity
# % keyword: SAM
# % keyword: SID
# % keyword: classification
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % required: yes
# % description: Input hyperspectral 3D raster (from i.hyper.import or i.hyper.atcorr)
# % guisection: Input
# %end

# %option G_OPT_R_OUTPUT
# % key: output
# % required: yes
# % description: Output similarity raster map (0=no match, 1=perfect match)
# % guisection: Output
# %end

# %option
# % key: reference
# % type: string
# % required: no
# % description: Reference spectrum as wavelength:reflectance pairs, comma-separated (e.g. 450:0.05,550:0.12,670:0.04,800:0.45)
# % guisection: Reference
# %end

# %option G_OPT_F_INPUT
# % key: reference_file
# % required: no
# % description: Reference spectrum file: CSV (wavelength,reflectance per line) or JSON ([[wl,r],...] or {"wavelengths":[...],"reflectances":[...]})
# % guisection: Reference
# %end

# %option
# % key: method
# % type: string
# % required: no
# % multiple: yes
# % options: sam,sid,sid_sam,ed,sad,sca,cr_sam,cr_ed,gd1,gd2,xcorr,dtw,ssim,jsd,bhatt,mtf,cem,ensemble,consensus
# % answer: sam
# % description: Spectral similarity method(s) — sam: Spectral Angle Mapper | sid: Spectral Information Divergence | sid_sam: SID×tan(SAM) hybrid | ed: Euclidean Distance (L2) | sad: Spectral Absolute Difference (L1) | sca: Spectral Correlation Angle (Pearson r) | cr_sam: Continuum-Removed SAM | cr_ed: Continuum-Removed Euclidean Distance | gd1: 1st-Derivative Shape Matching | gd2: 2nd-Derivative Shape Matching | xcorr: Normalized Cross-Correlation (shift-tolerant) | dtw: Dynamic Time Warping | ssim: Spectral Structural Similarity Index | jsd: Jensen-Shannon Divergence | bhatt: Bhattacharyya Coefficient | mtf: Matched/Tuned Filter | cem: Constrained Energy Minimization | ensemble: Rank-based ensemble fusion | consensus: Full multi-method consensus with calibrated probability fusion (runs all base methods, see fusion_mode=)
# % guisection: Methods
# %end

# %option
# % key: fusion_mode
# % type: string
# % required: no
# % options: rank_product,fisher,stouffer,group_product,harmonic,min
# % answer: rank_product
# % description: Fusion strategy for consensus= method — rank_product: weighted geometric mean of rank fractions (default, robust) | fisher: Fisher chi-squared combined probability test (statistically rigorous) | stouffer: Stouffer weighted Z-score combination (handles diversity weights naturally) | group_product: geometric-mean AND within method-type groups then arithmetic-mean OR across groups | harmonic: harmonic mean (strictest, requires all methods to agree) | min: minimum across methods (absolute strictest)
# % guisection: Methods
# %end

# %option
# % key: agreement_threshold
# % type: double
# % required: no
# % answer: 0.80
# % description: Calibrated-probability threshold for per-pixel agreement count (consensus= only); fraction of methods that must exceed this to count as "agreeing"
# % guisection: Methods
# %end

# %option
# % key: output_prefix
# % type: string
# % required: no
# % description: Output map prefix for individual per-method similarity maps (written alongside the main output)
# % guisection: Output
# %end

# %option
# % key: resample
# % type: string
# % required: no
# % options: linear,cubic,pchip
# % answer: linear
# % description: Interpolation method for resampling reference spectrum to sensor wavelengths
# % guisection: Processing
# %end

# %option
# % key: normalize
# % type: string
# % required: no
# % options: none,area,max,minmax,vector
# % answer: none
# % description: Spectrum normalization before matching: none=raw reflectance; area=divide by band sum; max=divide by maximum; minmax=0-1 range; vector=unit L2 norm
# % guisection: Processing
# %end

# %option
# % key: shift_window
# % type: integer
# % required: no
# % answer: 3
# % description: Maximum band-shift window for shift-tolerant methods (xcorr, dtw); 0 disables shift tolerance
# % guisection: Processing
# %end

# %option
# % key: min_wavelength
# % type: double
# % required: no
# % description: Minimum wavelength to consider (nm); defaults to sensor range
# % guisection: Processing
# %end

# %option
# % key: max_wavelength
# % type: double
# % required: no
# % description: Maximum wavelength to consider (nm); defaults to sensor range
# % guisection: Processing
# %end

# %option
# % key: coordinates
# % type: string
# % required: no
# % description: East,North coordinates for single-pixel point analysis (requires -p flag)
# % guisection: Point mode
# %end

# %flag
# % key: n
# % description: Only use bands marked valid (valid=1) in band metadata
# % guisection: Processing
# %end

# %flag
# % key: i
# % description: Info mode: print band coverage and reference spectrum summary, then exit
# % guisection: Processing
# %end

# %flag
# % key: v
# % description: Verbose: print processing details and per-band information
# % guisection: Processing
# %end

# %flag
# % key: c
# % description: Apply convex-hull continuum removal to both reference and pixel spectra before matching
# % guisection: Processing
# %end

# %flag
# % key: p
# % description: Point mode: analyse a single pixel at coordinates= and print full per-method score table
# % guisection: Point mode
# %end

# %flag
# % key: z
# % description: Normalize both reference and pixel spectra to unit sum (probability simplex) before matching; recommended with sid, jsd, bhatt
# % guisection: Processing
# %end

# %rules
# % required: reference,reference_file
# % exclusive: reference,reference_file
# %end

from __future__ import annotations

import sys
import os
import re
import csv
import json
import ctypes
import ctypes.util
import atexit
from typing import Optional

import numpy as np
import grass.script as gs

# ---------------------------------------------------------------------------
# Method metadata
# ---------------------------------------------------------------------------

ALL_METHODS = [
    'sam', 'sid', 'sid_sam', 'ed', 'sad', 'sca',
    'cr_sam', 'cr_ed', 'gd1', 'gd2',
    'xcorr', 'dtw', 'ssim', 'jsd', 'bhatt',
    'mtf', 'cem', 'ensemble', 'consensus',
]

# Base methods that consensus will run (everything except meta-methods)
BASE_METHODS = [m for m in ALL_METHODS if m not in ('ensemble', 'consensus')]

METHOD_LABELS = {
    'sam':       'Spectral Angle Mapper',
    'sid':       'Spectral Information Divergence',
    'sid_sam':   'SID × tan(SAM) hybrid',
    'ed':        'Euclidean Distance (L2)',
    'sad':       'Spectral Absolute Difference (L1)',
    'sca':       'Spectral Correlation Angle (Pearson r)',
    'cr_sam':    'Continuum-Removed SAM',
    'cr_ed':     'Continuum-Removed Euclidean Distance',
    'gd1':       '1st-Derivative Shape Matching',
    'gd2':       '2nd-Derivative Shape Matching',
    'xcorr':     'Normalized Cross-Correlation',
    'dtw':       'Dynamic Time Warping',
    'ssim':      'Spectral Structural Similarity Index',
    'jsd':       'Jensen-Shannon Divergence',
    'bhatt':     'Bhattacharyya Coefficient',
    'mtf':       'Matched Tuned Filter',
    'cem':       'Constrained Energy Minimization',
    'ensemble':  'Rank-based Ensemble Fusion',
    'consensus': 'Multi-method Consensus (calibrated probability fusion)',
}

# Methods grouped by mathematical basis — used by the group_product fusion mode
# to treat correlated methods as a single evidence source.
METHOD_GROUPS: dict[str, list[str]] = {
    'geometric':   ['sam', 'sca', 'cr_sam', 'gd1', 'gd2'],
    'distance':    ['ed', 'sad', 'cr_ed'],
    'information': ['sid', 'sid_sam', 'jsd'],
    'statistical': ['bhatt', 'ssim'],
    'signal':      ['xcorr', 'dtw'],
    'subpixel':    ['mtf', 'cem'],
}

# Supported fusion modes for consensus analysis
FUSION_MODES = (
    'rank_product',   # weighted geometric mean of rank fractions — default
    'fisher',         # Fisher's combined probability chi-squared test
    'stouffer',       # Stouffer's weighted Z-score combination
    'group_product',  # AND within method groups, OR across groups
    'harmonic',       # harmonic mean (strictest AND)
    'min',            # minimum across all methods (absolute strictest)
)

# Methods that require image-wide statistics (covariance)
GLOBAL_STATS_METHODS = {'mtf', 'cem'}

# ---------------------------------------------------------------------------
# Temporary raster cleanup
# ---------------------------------------------------------------------------

_TMP_RASTERS: list[str] = []


def _cleanup():
    if _TMP_RASTERS:
        gs.run_command('g.remove', type='raster',
                       name=','.join(_TMP_RASTERS), flags='f', quiet=True)


atexit.register(_cleanup)


def _tmp(label: str) -> str:
    name = f"tmp_ihsleuth_{os.getpid()}_{label}"
    _TMP_RASTERS.append(name)
    return name

# ---------------------------------------------------------------------------
# libgrass_g3d: fast band extraction
# ---------------------------------------------------------------------------

_G3D_LIB = None


def _load_g3d_lib():
    global _G3D_LIB
    if _G3D_LIB is not None:
        return _G3D_LIB

    lib_dir = os.path.join(os.environ['GISBASE'], 'lib')

    import re as _re
    _vh = os.path.join(os.environ['GISBASE'], 'include', 'grass', 'version.h')
    with open(_vh) as _f:
        _m = _re.search(r'#define\s+GRASS_HEADERS_VERSION\s+"([^"]+)"', _f.read())
    headers_version = (_m.group(1).encode() if _m else b"")

    gis_lib = ctypes.CDLL(os.path.join(lib_dir, 'libgrass_gis.so'),
                          mode=ctypes.RTLD_GLOBAL)
    gis_lib.G__no_gisinit.restype = None
    gis_lib.G__no_gisinit.argtypes = [ctypes.c_char_p]
    gis_lib.G__no_gisinit(headers_version)

    lib = ctypes.CDLL(os.path.join(lib_dir, 'libgrass_g3d.so'))
    lib.Rast3d_extract_z_slice.restype = ctypes.c_int
    lib.Rast3d_extract_z_slice.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    _G3D_LIB = lib
    return lib


def extract_band(raster3d: str, band_num: int) -> str:
    """Extract band_num (1-based) from raster3d to a temp 2D raster."""
    lib = _load_g3d_lib()
    z = band_num - 1
    base = raster3d.replace('@', '_').replace('#', '_').replace('.', '_')
    tmp_name = _tmp(f"band_{base}_{band_num}")
    name3d, mapset3d = (raster3d.split('@') + [''])[:2]
    ret = lib.Rast3d_extract_z_slice(
        name3d.encode(),
        mapset3d.encode() if mapset3d else None,
        ctypes.c_int(z),
        tmp_name.encode(),
    )
    if ret != 0:
        gs.fatal(f"Rast3d_extract_z_slice failed for band {band_num} of {raster3d}")
    return tmp_name

# ---------------------------------------------------------------------------
# Band metadata helpers (unified with other i.hyper.* addons)
# ---------------------------------------------------------------------------


def _parse_wl_from_r3info(raster3d: str) -> dict[int, tuple]:
    """Parse Band N: WL nm, FWHM: F nm lines from r3.info history."""
    try:
        info_text = gs.read_command('r3.info', flags='h', map=raster3d)
    except Exception:
        try:
            info_text = gs.read_command('r3.info', map=raster3d)
        except Exception:
            return {}
    pat = re.compile(r'Band\s+(\d+):\s+([\d.]+)\s+nm[,\s]+FWHM:\s+([\d.]+)\s+nm')
    bands: dict[int, tuple] = {}
    for line in info_text.split('\n'):
        m = pat.search(line)
        if m:
            bands[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return bands


def _convert_wl_nm(wl: float, unit: str) -> float:
    u = unit.lower().strip()
    if u in ('nm', 'nanometer', 'nanometers'):
        return wl
    if u in ('um', 'µm', 'micrometer', 'micron', 'microns'):
        return wl * 1000.0
    if u in ('m', 'meter', 'meters'):
        return wl * 1e9
    gs.warning(f"Unknown wavelength unit '{unit}'; assuming nm")
    return wl


def get_band_info(raster3d: str, only_valid: bool = False,
                  min_wl: Optional[float] = None,
                  max_wl: Optional[float] = None) -> list[dict]:
    """Return sorted list of band dicts with keys: band, wavelength, fwhm, valid, map_name."""
    info = gs.raster3d_info(raster3d)
    depths = int(info['depths'])

    # Check whether 2D band-slice rasters exist (i.hyper.import workflow)
    base = raster3d.split('@')[0]
    mapset = raster3d.split('@')[1] if '@' in raster3d else None
    slices_exist = bool(gs.find_file(f"{base}#1", element='cell', mapset=mapset).get('name'))

    bands: list[dict] = []

    if slices_exist:
        for i in range(1, depths + 1):
            band_name = f"{raster3d}#{i}"
            wl = fwhm = None
            valid = True
            unit = 'nm'
            try:
                result = gs.read_command('r.info', map=band_name, flags='h')
                for line in result.split('\n'):
                    line = line.strip()
                    if line.startswith('wavelength='):
                        wl = float(line.split('=')[1])
                    elif line.startswith('FWHM='):
                        fwhm = float(line.split('=')[1])
                    elif line.startswith('valid='):
                        valid = int(line.split('=')[1]) == 1
                    elif line.startswith('unit='):
                        unit = line.split('=')[1].strip()
            except Exception:
                pass
            if wl is None:
                continue
            wl_nm = _convert_wl_nm(wl, unit)
            if min_wl is not None and wl_nm < min_wl:
                continue
            if max_wl is not None and wl_nm > max_wl:
                continue
            if only_valid and not valid:
                continue
            bands.append({'band': i, 'wavelength': wl_nm,
                          'fwhm': fwhm or 10.0, 'valid': valid,
                          'map_name': band_name})
    else:
        wl_dict = _parse_wl_from_r3info(raster3d)
        for i in range(1, depths + 1):
            if i not in wl_dict:
                continue
            wl_nm, fwhm = wl_dict[i]
            if min_wl is not None and wl_nm < min_wl:
                continue
            if max_wl is not None and wl_nm > max_wl:
                continue
            if only_valid:
                continue  # can't check validity without slice metadata
            bands.append({'band': i, 'wavelength': wl_nm,
                          'fwhm': fwhm, 'valid': True,
                          'map_name': None})

    if not bands:
        gs.fatal(f"No wavelength metadata found in '{raster3d}'. "
                 "Import data with i.hyper.import or ensure Band N/FWHM lines "
                 "are in r3.info history.")

    bands.sort(key=lambda b: b['wavelength'])
    return bands

# ---------------------------------------------------------------------------
# Reference spectrum I/O
# ---------------------------------------------------------------------------


def parse_reference_inline(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse 'wl1:r1,wl2:r2,...' or 'wl1;r1,wl2;r2,...' inline string."""
    pairs = []
    for tok in re.split(r'[,\s]+', text.strip()):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r'([\d.eE+\-]+)[;:]([\d.eE+\-]+)', tok)
        if not m:
            gs.fatal(f"Cannot parse reference token '{tok}' — expected wl:r format")
        pairs.append((float(m.group(1)), float(m.group(2))))
    if len(pairs) < 2:
        gs.fatal("Reference spectrum must contain at least 2 wavelength:reflectance pairs")
    pairs.sort(key=lambda p: p[0])
    wls = np.array([p[0] for p in pairs], dtype=np.float64)
    refs = np.array([p[1] for p in pairs], dtype=np.float64)
    return wls, refs


def parse_reference_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse CSV or JSON reference spectrum file.

    CSV: wavelength,reflectance rows (with or without header).
    JSON: [[wl, r], ...] or {"wavelengths": [...], "reflectances": [...]}
        or {"data": [[wl, r], ...]} or {"spectrum": [[wl, r], ...]}
    """
    if not os.path.isfile(path):
        gs.fatal(f"Reference file not found: {path}")

    # Try JSON first
    try:
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            pairs = [(float(row[0]), float(row[1])) for row in obj]
        elif isinstance(obj, dict):
            for key in ('data', 'spectrum', 'pairs'):
                if key in obj and isinstance(obj[key], list):
                    pairs = [(float(r[0]), float(r[1])) for r in obj[key]]
                    break
            else:
                wls_j = obj.get('wavelengths') or obj.get('wavelength') or obj.get('wl')
                refs_j = obj.get('reflectances') or obj.get('reflectance') or obj.get('r')
                if wls_j is None or refs_j is None:
                    gs.fatal("JSON reference file missing 'wavelengths'/'reflectances' keys")
                pairs = list(zip(map(float, wls_j), map(float, refs_j)))
        else:
            gs.fatal(f"Unexpected JSON structure in {path}")
        pairs.sort(key=lambda p: p[0])
        return (np.array([p[0] for p in pairs], dtype=np.float64),
                np.array([p[1] for p in pairs], dtype=np.float64))
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass

    # CSV fallback
    pairs = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                w = float(row[0])
                r = float(row[1])
                pairs.append((w, r))
            except ValueError:
                continue  # skip header or comments
    if len(pairs) < 2:
        gs.fatal(f"Could not parse at least 2 wavelength,reflectance pairs from {path}")
    pairs.sort(key=lambda p: p[0])
    return (np.array([p[0] for p in pairs], dtype=np.float64),
            np.array([p[1] for p in pairs], dtype=np.float64))

# ---------------------------------------------------------------------------
# Spectrum resampling
# ---------------------------------------------------------------------------


class WavelengthLUT:
    """Precomputed resampling lookup table between two wavelength grids.

    Build once with ``WavelengthLUT(src_wls, dst_wls)``, then call
    :meth:`apply` or :meth:`apply_cube` instantly for any value array
    on the src grid — no repeated binary-search or weight computation.

    The table also exposes full overlap diagnostics so callers can
    restrict matching to the common wavelength range before scoring.

    Parameters
    ----------
    src_wls : array_like
        Source (e.g. reference) wavelengths, strictly increasing, in nm.
    dst_wls : array_like
        Destination (e.g. sensor band) wavelengths, strictly increasing, in nm.
    fill : 'edge' | 'nan' | float
        Behaviour for dst points that lie outside the src range.
        ``'edge'`` reuses the nearest src endpoint value (default).
        ``'nan'`` writes NaN.  A float fills with that constant.
    """

    def __init__(self, src_wls, dst_wls, fill: str | float = 'edge') -> None:
        src_wls = np.asarray(src_wls, dtype=np.float64)
        dst_wls = np.asarray(dst_wls, dtype=np.float64)

        if src_wls.ndim != 1 or dst_wls.ndim != 1:
            raise ValueError("WavelengthLUT: both wavelength arrays must be 1-D")
        if src_wls.size < 2 or dst_wls.size < 2:
            raise ValueError("WavelengthLUT: need at least 2 points in each grid")
        if np.any(np.diff(src_wls) <= 0) or np.any(np.diff(dst_wls) <= 0):
            raise ValueError("WavelengthLUT: wavelength arrays must be strictly increasing")

        self.src_wls = src_wls
        self.dst_wls = dst_wls
        self.fill = fill
        n_src = len(src_wls)

        # ── Binary-search indices (computed once) ────────────────────────
        # searchsorted('right') → for exact matches the right bracket is
        # one past the hit, so left bracket holds the exact value.
        ridx = np.searchsorted(src_wls, dst_wls, side='right')
        ridx = np.clip(ridx, 1, n_src - 1)
        lidx = ridx - 1

        # ── Linear interpolation weights ─────────────────────────────────
        dx = src_wls[ridx] - src_wls[lidx]          # spacing at each bracket
        alpha = np.where(dx > 0,
                         (dst_wls - src_wls[lidx]) / dx,
                         0.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        self.left_idx  = lidx   # (n_dst,) → index into src
        self.right_idx = ridx   # (n_dst,) → index into src
        self.alpha     = alpha  # (n_dst,) weight towards right bracket

        # ── Validity / overlap masks ──────────────────────────────────────
        # valid_dst[i] = True  iff dst_wls[i] is within src coverage
        self.valid_dst = ((dst_wls >= src_wls[0]) &
                          (dst_wls <= src_wls[-1]))
        # valid_src[j] = True  iff src_wls[j] is within dst coverage (observable)
        self.valid_src = ((src_wls >= dst_wls[0]) &
                          (src_wls <= dst_wls[-1]))

        # ── Overlap range ─────────────────────────────────────────────────
        wl_lo = max(src_wls[0], dst_wls[0])
        wl_hi = min(src_wls[-1], dst_wls[-1])
        self.overlap_lo: float = float(wl_lo)
        self.overlap_hi: float = float(wl_hi)
        self.has_overlap: bool = bool(wl_lo < wl_hi)

        # Precomputed index arrays for the overlap sub-grids
        self.overlap_dst_idx = np.where(self.valid_dst)[0]   # into dst
        self.overlap_src_idx = np.where(self.valid_src)[0]   # into src
        self.overlap_dst_wls = dst_wls[self.overlap_dst_idx]
        self.overlap_src_wls = src_wls[self.overlap_src_idx]

    # ── Core apply methods ────────────────────────────────────────────────

    def apply(self, src_vals: np.ndarray) -> np.ndarray:
        """Resample a 1-D src value array to the dst grid.

        O(n_dst) with no binary-search; uses precomputed indices.

        Parameters
        ----------
        src_vals : ndarray, shape (n_src,)

        Returns
        -------
        ndarray, shape (n_dst,)
        """
        src_vals = np.asarray(src_vals, dtype=np.float64)
        if src_vals.shape[0] != len(self.src_wls):
            raise ValueError(
                f"WavelengthLUT.apply: src_vals length {src_vals.shape[0]} "
                f"!= src_wls length {len(self.src_wls)}"
            )
        result = (src_vals[self.left_idx] * (1.0 - self.alpha) +
                  src_vals[self.right_idx] * self.alpha)
        self._apply_fill(result)
        return result

    def apply_cube(self, cube: np.ndarray) -> np.ndarray:
        """Resample the band axis of a cube from the src to the dst grid.

        Performs a single vectorised gather-and-blend; no Python loops.

        Parameters
        ----------
        cube : ndarray, shape (n_src, ...) — any trailing dimensions

        Returns
        -------
        ndarray, shape (n_dst, ...)
        """
        if cube.shape[0] != len(self.src_wls):
            raise ValueError(
                f"WavelengthLUT.apply_cube: cube band axis {cube.shape[0]} "
                f"!= src_wls length {len(self.src_wls)}"
            )
        left_vals  = cube[self.left_idx]    # (n_dst, ...)
        right_vals = cube[self.right_idx]   # (n_dst, ...)
        alpha = self.alpha
        for _ in range(cube.ndim - 1):
            alpha = alpha[..., np.newaxis]  # broadcast over trailing axes
        result = left_vals * (1.0 - alpha) + right_vals * alpha
        self._apply_fill_nd(result)
        return result

    def _apply_fill(self, result: np.ndarray) -> None:
        oor = ~self.valid_dst
        if not np.any(oor):
            return
        if self.fill == 'edge':
            pass  # left/right clamp already produced edge values
        elif self.fill == 'nan':
            result[oor] = np.nan
        else:
            result[oor] = float(self.fill)

    def _apply_fill_nd(self, result: np.ndarray) -> None:
        oor = ~self.valid_dst
        if not np.any(oor):
            return
        if self.fill == 'edge':
            return
        fill_val = np.nan if self.fill == 'nan' else float(self.fill)
        result[oor] = fill_val

    # ── Overlap helpers ───────────────────────────────────────────────────

    def restrict_to_overlap(
        self,
        src_vals: Optional[np.ndarray] = None,
        dst_bands: Optional[list] = None,
    ) -> tuple:
        """Reduce both grids to the common wavelength overlap.

        Returns a tuple
        ``(ovl_src_wls, ovl_src_vals, ovl_dst_wls, ovl_dst_bands, sub_lut)``

        where *sub_lut* is a new :class:`WavelengthLUT` built on the reduced
        grids — ready for fast application without further clipping.

        Parameters
        ----------
        src_vals : ndarray (n_src,), optional
            If provided, also sliced to the overlap src indices.
        dst_bands : list, optional
            If provided (e.g. the GRASS band-dict list), sliced to the
            overlap dst indices.
        """
        ovl_src_wls = self.overlap_src_wls
        ovl_dst_wls = self.overlap_dst_wls
        ovl_src_vals = (src_vals[self.overlap_src_idx]
                        if src_vals is not None else None)
        ovl_dst_bands = ([dst_bands[i] for i in self.overlap_dst_idx]
                         if dst_bands is not None else None)
        sub_lut = WavelengthLUT(ovl_src_wls, ovl_dst_wls, fill=self.fill)
        return ovl_src_wls, ovl_src_vals, ovl_dst_wls, ovl_dst_bands, sub_lut

    def coverage_report(self) -> str:
        """Return a human-readable coverage summary string."""
        n_dst = len(self.dst_wls)
        n_src = len(self.src_wls)
        n_cov = int(self.valid_dst.sum())
        n_vis = int(self.valid_src.sum())
        pct_cov = 100.0 * n_cov / n_dst if n_dst else 0.0
        pct_vis = 100.0 * n_vis / n_src if n_src else 0.0
        return (
            f"src [{self.src_wls[0]:.1f}–{self.src_wls[-1]:.1f} nm, n={n_src}] → "
            f"dst [{self.dst_wls[0]:.1f}–{self.dst_wls[-1]:.1f} nm, n={n_dst}]  |  "
            f"overlap [{self.overlap_lo:.1f}–{self.overlap_hi:.1f} nm]  |  "
            f"dst covered {n_cov}/{n_dst} ({pct_cov:.1f}%)  "
            f"src visible {n_vis}/{n_src} ({pct_vis:.1f}%)"
        )

    def __repr__(self) -> str:
        return f"WavelengthLUT({self.coverage_report()})"


def resample_reference(ref_wls: np.ndarray, ref_vals: np.ndarray,
                       sensor_wls: np.ndarray,
                       method: str = 'linear',
                       lut: Optional['WavelengthLUT'] = None) -> np.ndarray:
    """Resample reference spectrum to the sensor wavelength grid.

    For ``method='linear'`` a pre-built :class:`WavelengthLUT` can be passed
    to skip index recomputation.  For cubic/pchip the LUT is not used but
    its overlap mask is honoured to avoid extrapolation artefacts.
    """
    if method == 'linear':
        if lut is None:
            lut = WavelengthLUT(ref_wls, sensor_wls)
        return lut.apply(ref_vals)

    from scipy.interpolate import CubicSpline, PchipInterpolator

    clipped = np.clip(sensor_wls, ref_wls[0], ref_wls[-1])
    if method == 'cubic':
        interp = CubicSpline(ref_wls, ref_vals, bc_type='not-a-knot',
                             extrapolate=False)
        result = interp(clipped)
    elif method == 'pchip':
        interp = PchipInterpolator(ref_wls, ref_vals, extrapolate=False)
        result = interp(clipped)
    else:
        gs.fatal(f"Unknown resample method: {method}")

    # NaN guard: fall back to linear for any NaN produced by scipy
    bad = ~np.isfinite(result)
    if np.any(bad):
        fallback = np.interp(clipped[bad], ref_wls, ref_vals)
        result[bad] = fallback

    # Apply fill for out-of-range sensor bands using the LUT mask
    if lut is None:
        lut = WavelengthLUT(ref_wls, sensor_wls)
    oor = ~lut.valid_dst
    if np.any(oor):
        if lut.fill == 'nan':
            result[oor] = np.nan
        elif lut.fill != 'edge':
            result[oor] = float(lut.fill)
        # 'edge': clipped already produced edge values — no action needed

    return result.astype(np.float64)

# ---------------------------------------------------------------------------
# Cube I/O
# ---------------------------------------------------------------------------


def load_cube(bands: list[dict], raster3d: str,
              verbose: bool = False) -> np.ndarray:
    """Load all bands into a float32 numpy array (n_bands, rows, cols).

    Uses garray for 2D raster reading.  Band slice rasters are extracted from
    the 3D raster on demand when map_name is None.
    """
    import grass.script.array as garray

    n = len(bands)
    cube = None

    for idx, b in enumerate(bands):
        if verbose:
            gs.verbose(f"  Loading band {b['band']} ({b['wavelength']:.1f} nm) ...")
        gs.percent(idx, n, 5)

        map_name = b.get('map_name')
        if map_name is None:
            map_name = extract_band(raster3d, b['band'])
            b['map_name'] = map_name

        arr = garray.array()
        arr.read(map_name)
        band_data = np.asarray(arr, dtype=np.float32)

        if cube is None:
            rows, cols = band_data.shape
            cube = np.empty((n, rows, cols), dtype=np.float32)

        cube[idx] = band_data

    gs.percent(n, n, 5)
    return cube


def read_pixel_spectrum(bands: list[dict], raster3d: str,
                        east: float, north: float) -> np.ndarray:
    """Extract the spectrum at a single geographic coordinate."""
    coords = f"{east},{north}"
    spec = []
    for b in bands:
        map_name = b.get('map_name')
        if map_name is None:
            map_name = extract_band(raster3d, b['band'])
            b['map_name'] = map_name
        result = gs.read_command('r.what', map=map_name, coordinates=coords)
        # r.what output: map|x|y|value
        lines = result.strip().split('\n')
        val = np.nan
        for line in lines:
            parts = line.split('|')
            if len(parts) >= 4:
                try:
                    val = float(parts[3])
                except ValueError:
                    val = np.nan
                break
        spec.append(val)
    return np.array(spec, dtype=np.float64)


def write_raster(data: np.ndarray, name: str, overwrite: bool = True) -> None:
    """Write a 2D numpy array as a GRASS raster."""
    import grass.script.array as garray
    arr = garray.array()
    arr[:] = data.astype(np.float64)
    arr.write(name, overwrite=overwrite)

# ---------------------------------------------------------------------------
# Spectrum preprocessing
# ---------------------------------------------------------------------------


def normalize_spectrum(spec: np.ndarray, method: str) -> np.ndarray:
    """Normalize a 1-D spectrum array."""
    if method == 'none':
        return spec
    s = spec.copy()
    if method == 'area':
        total = np.sum(s)
        if total > 0:
            s /= total
    elif method == 'max':
        m = np.max(s)
        if m > 0:
            s /= m
    elif method == 'minmax':
        lo, hi = np.min(s), np.max(s)
        if hi > lo:
            s = (s - lo) / (hi - lo)
        else:
            s = np.zeros_like(s)
    elif method == 'vector':
        n = np.linalg.norm(s)
        if n > 0:
            s /= n
    else:
        gs.fatal(f"Unknown normalize method: {method}")
    return s


def normalize_cube(cube: np.ndarray, method: str) -> np.ndarray:
    """Normalize a (n_bands, rows, cols) cube along the band axis."""
    if method == 'none':
        return cube
    c = cube.copy()
    if method == 'area':
        total = np.sum(c, axis=0, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        c /= total
    elif method == 'max':
        mx = np.max(c, axis=0, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        c /= mx
    elif method == 'minmax':
        lo = np.min(c, axis=0, keepdims=True)
        hi = np.max(c, axis=0, keepdims=True)
        rng = np.where(hi > lo, hi - lo, 1.0)
        c = (c - lo) / rng
    elif method == 'vector':
        norms = np.linalg.norm(c, axis=0, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        c /= norms
    return c


def to_prob_simplex(spec: np.ndarray) -> np.ndarray:
    """Normalize 1-D spectrum to the probability simplex (sum to 1, all ≥ 0)."""
    s = np.maximum(spec, 1e-12)
    return s / s.sum()


def to_prob_simplex_cube(cube: np.ndarray) -> np.ndarray:
    """Normalize cube band-axis to probability simplex per pixel."""
    c = np.maximum(cube, 1e-12)
    totals = c.sum(axis=0, keepdims=True)
    return c / totals

# ---------------------------------------------------------------------------
# Continuum removal (upper convex hull)
# ---------------------------------------------------------------------------


def _upper_hull(wavelengths: np.ndarray,
                reflectances: np.ndarray) -> np.ndarray:
    """Compute the upper convex hull (continuum) for one spectrum.

    Returns an array of continuum values at each wavelength position.
    Uses the Graham-scan upper hull algorithm.
    """
    pts = list(zip(wavelengths.tolist(), reflectances.tolist()))
    # Graham scan: upper hull (points visible from above)
    hull: list[tuple] = []
    for p in pts:
        while len(hull) >= 2:
            o, a, b = hull[-2], hull[-1], p
            # Cross product z-component; negative = right turn (concave up) → remove
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hull_wl = np.array([h[0] for h in hull])
    hull_rf = np.array([h[1] for h in hull])
    return np.interp(wavelengths, hull_wl, hull_rf)


def continuum_remove(spec: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Apply convex-hull continuum removal to a 1-D spectrum."""
    continuum = _upper_hull(wavelengths, spec)
    with np.errstate(invalid='ignore', divide='ignore'):
        cr = np.where(continuum > 0, spec / continuum, 1.0)
    return np.clip(cr, 0.0, 1.0)


def continuum_remove_cube(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Apply continuum removal to all pixels in a cube (n_bands, rows, cols).

    Implemented as a vectorized batch loop for performance.
    """
    n_bands, rows, cols = cube.shape
    n_pix = rows * cols
    flat = cube.reshape(n_bands, n_pix).T  # (n_pix, n_bands)
    result = np.empty_like(flat)

    for i in range(n_pix):
        spec = flat[i]
        continuum = _upper_hull(wavelengths, spec)
        with np.errstate(invalid='ignore', divide='ignore'):
            cr = np.where(continuum > 0, spec / continuum, 1.0)
        result[i] = np.clip(cr, 0.0, 1.0)

    return result.T.reshape(n_bands, rows, cols)

# ---------------------------------------------------------------------------
# Spectral similarity / distance methods (all return similarity in [0, 1])
# ---------------------------------------------------------------------------

EPS = 1e-12


def _safe_cube(cube: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in cube with 0."""
    return np.where(np.isfinite(cube), cube, 0.0)


# ── 1. SAM — Spectral Angle Mapper ────────────────────────────────────────

def match_sam(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Spectral Angle Mapper: similarity = 1 – 2θ/π, θ ∈ [0, π/2]."""
    cube = _safe_cube(cube)
    dot = np.einsum('i,ijk->jk', ref, cube)
    norm_ref = np.linalg.norm(ref) + EPS
    norm_cube = np.sqrt(np.einsum('ijk,ijk->jk', cube, cube)) + EPS
    cos_a = np.clip(dot / (norm_ref * norm_cube), -1.0, 1.0)
    theta = np.arccos(cos_a)
    return np.clip(1.0 - 2.0 * theta / np.pi, 0.0, 1.0)


# ── 2. SID — Spectral Information Divergence ──────────────────────────────

def match_sid(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """SID = D(p‖q) + D(q‖p), similarity = exp(−SID)."""
    cube = _safe_cube(cube)
    p = ref / (ref.sum() + EPS) + EPS           # (n,)
    q = cube / (cube.sum(axis=0, keepdims=True) + EPS) + EPS  # (n, r, c)
    p3 = p[:, np.newaxis, np.newaxis]
    kld_pq = np.sum(p3 * np.log(p3 / q), axis=0)
    kld_qp = np.sum(q * np.log(q / p3), axis=0)
    sid = kld_pq + kld_qp
    return np.clip(np.exp(-np.abs(sid)), 0.0, 1.0)


# ── 3. SID-SAM — Hybrid ───────────────────────────────────────────────────

def match_sid_sam(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """SID × tan(SAM); similarity = exp(−SID·tan(θ))."""
    cube = _safe_cube(cube)
    # SAM angle
    dot = np.einsum('i,ijk->jk', ref, cube)
    norm_ref = np.linalg.norm(ref) + EPS
    norm_cube = np.sqrt(np.einsum('ijk,ijk->jk', cube, cube)) + EPS
    theta = np.arccos(np.clip(dot / (norm_ref * norm_cube), -1.0, 1.0))
    tan_theta = np.tan(np.clip(theta, 0.0, np.pi / 2 - 0.001))
    # SID
    p = ref / (ref.sum() + EPS) + EPS
    q = cube / (cube.sum(axis=0, keepdims=True) + EPS) + EPS
    p3 = p[:, np.newaxis, np.newaxis]
    sid = np.sum(p3 * np.log(p3 / q), axis=0) + np.sum(q * np.log(q / p3), axis=0)
    score = np.abs(sid) * tan_theta
    return np.clip(np.exp(-score), 0.0, 1.0)


# ── 4. ED — Euclidean Distance ────────────────────────────────────────────

def match_ed(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """L2 distance; similarity = 1/(1 + d/√n)."""
    cube = _safe_cube(cube)
    diff = cube - ref[:, np.newaxis, np.newaxis]
    n = cube.shape[0]
    dist = np.sqrt(np.sum(diff ** 2, axis=0)) / np.sqrt(n)
    return 1.0 / (1.0 + dist)


# ── 5. SAD — Spectral Absolute Difference ─────────────────────────────────

def match_sad(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """L1 distance; similarity = 1/(1 + mean|r_i − p_i|)."""
    cube = _safe_cube(cube)
    dist = np.mean(np.abs(cube - ref[:, np.newaxis, np.newaxis]), axis=0)
    return 1.0 / (1.0 + dist)


# ── 6. SCA — Spectral Correlation Angle (Pearson r) ──────────────────────

def match_sca(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pearson r between reference and pixel spectrum; similarity = (r+1)/2."""
    cube = _safe_cube(cube)
    mu_ref = ref.mean()
    mu_cube = cube.mean(axis=0)            # (r, c)
    ref_c = ref - mu_ref                   # (n,)
    cube_c = cube - mu_cube[np.newaxis]    # (n, r, c)
    n = cube.shape[0]
    cov = np.sum(ref_c[:, np.newaxis, np.newaxis] * cube_c, axis=0)
    std_ref = (ref_c ** 2).sum() ** 0.5 + EPS
    std_cube = np.sqrt((cube_c ** 2).sum(axis=0)) + EPS
    r = cov / (std_ref * std_cube)
    return np.clip((r + 1.0) / 2.0, 0.0, 1.0)


# ── 7 & 8. Continuum-Removed SAM / ED ────────────────────────────────────

def match_cr_sam(cube: np.ndarray, ref: np.ndarray,
                 wavelengths: np.ndarray) -> np.ndarray:
    """SAM applied to continuum-removed spectra."""
    ref_cr = continuum_remove(ref, wavelengths)
    cube_cr = continuum_remove_cube(cube, wavelengths)
    return match_sam(cube_cr, ref_cr)


def match_cr_ed(cube: np.ndarray, ref: np.ndarray,
                wavelengths: np.ndarray) -> np.ndarray:
    """Euclidean distance applied to continuum-removed spectra."""
    ref_cr = continuum_remove(ref, wavelengths)
    cube_cr = continuum_remove_cube(cube, wavelengths)
    return match_ed(cube_cr, ref_cr)


# ── 9 & 10. Derivative Matching ───────────────────────────────────────────

def _spectral_derivative(spec: np.ndarray, wavelengths: np.ndarray,
                         order: int = 1) -> np.ndarray:
    """Compute finite-difference derivative normalized by wavelength spacing."""
    dx = np.diff(wavelengths)
    dy = np.diff(spec, axis=0)
    d = dy / np.where(dx > 0, dx, 1.0)[:, np.newaxis, np.newaxis] \
        if spec.ndim == 3 else dy / np.where(dx > 0, dx, 1.0)
    if order == 2:
        dx2 = (dx[:-1] + dx[1:]) / 2.0
        dd = np.diff(d, axis=0)
        d = dd / np.where(dx2 > 0, dx2, 1.0)[:, np.newaxis, np.newaxis] \
            if d.ndim == 3 else dd / np.where(dx2 > 0, dx2, 1.0)
    return d


def match_gd1(cube: np.ndarray, ref: np.ndarray,
              wavelengths: np.ndarray) -> np.ndarray:
    """SAM applied to first-derivative spectra."""
    dx = np.diff(wavelengths)
    ref_d = np.diff(ref) / np.where(dx > 0, dx, 1.0)
    cube_d = np.diff(cube, axis=0) / np.where(dx > 0, dx, 1.0)[:, np.newaxis, np.newaxis]
    return match_sam(cube_d, ref_d)


def match_gd2(cube: np.ndarray, ref: np.ndarray,
              wavelengths: np.ndarray) -> np.ndarray:
    """SAM applied to second-derivative spectra."""
    dx = np.diff(wavelengths)
    ref_d1 = np.diff(ref) / np.where(dx > 0, dx, 1.0)
    cube_d1 = np.diff(cube, axis=0) / np.where(dx > 0, dx, 1.0)[:, np.newaxis, np.newaxis]
    dx2 = (dx[:-1] + dx[1:]) / 2.0
    ref_d2 = np.diff(ref_d1) / np.where(dx2 > 0, dx2, 1.0)
    cube_d2 = np.diff(cube_d1, axis=0) / np.where(dx2 > 0, dx2, 1.0)[:, np.newaxis, np.newaxis]
    return match_sam(cube_d2, ref_d2)


# ── 11. XCORR — Normalized Cross-Correlation ──────────────────────────────

def match_xcorr(cube: np.ndarray, ref: np.ndarray,
                max_lag: int = 3) -> np.ndarray:
    """Normalized cross-correlation at lags [−max_lag, +max_lag]; return max NCC.

    NCC at lag k = Σ (r(i) − μ_r)(p(i+k) − μ_p) / (n·σ_r·σ_p)
    """
    cube = _safe_cube(cube)
    n, rows, cols = cube.shape
    mu_ref = ref.mean()
    mu_cube = cube.mean(axis=0)       # (r, c)
    ref_c = ref - mu_ref
    cube_c = cube - mu_cube[np.newaxis]
    std_ref = np.sqrt((ref_c ** 2).mean()) + EPS
    std_cube = np.sqrt((cube_c ** 2).mean(axis=0)) + EPS

    best_ncc = np.full((rows, cols), -1.0)
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            i_ref = slice(None)
            i_pix = slice(None)
        elif lag > 0:
            i_ref = slice(None, n - lag)
            i_pix = slice(lag, None)
        else:
            i_ref = slice(-lag, None)
            i_pix = slice(None, n + lag)
        n_eff = len(range(n)[i_ref])
        if n_eff < 2:
            continue
        ncc = np.sum(ref_c[i_ref, np.newaxis, np.newaxis] * cube_c[i_pix],
                     axis=0) / (n_eff * std_ref * std_cube)
        best_ncc = np.maximum(best_ncc, ncc)

    return np.clip((best_ncc + 1.0) / 2.0, 0.0, 1.0)


# ── 12. DTW — Dynamic Time Warping ────────────────────────────────────────

def match_dtw(cube: np.ndarray, ref: np.ndarray,
              window: int = 3) -> np.ndarray:
    """DTW distance with Sakoe-Chiba band; similarity = exp(−dtw_norm).

    Computed in batches to manage memory.
    """
    cube = _safe_cube(cube)
    n_bands, rows, cols = cube.shape
    n_pix = rows * cols
    flat = cube.reshape(n_bands, n_pix).T.astype(np.float32)  # (n_pix, n_bands)
    m = len(ref)
    ref_f = ref.astype(np.float32)

    CHUNK = 512
    distances = np.empty(n_pix, dtype=np.float32)

    for start in range(0, n_pix, CHUNK):
        end = min(start + CHUNK, n_pix)
        chunk = flat[start:end]   # (k, n)
        k = end - start

        # Rolling two-row DTW: (k, m+1)
        INF = np.float32(1e30)
        prev = np.full((k, m + 1), INF, dtype=np.float32)
        prev[:, 0] = 0.0

        for i in range(1, n_bands + 1):
            curr = np.full((k, m + 1), INF, dtype=np.float32)
            j_lo = max(1, i - window)
            j_hi = min(m, i + window)
            for j in range(j_lo, j_hi + 1):
                cost = (chunk[:, i - 1] - ref_f[j - 1]) ** 2  # (k,)
                best = np.minimum(
                    np.minimum(prev[:, j], curr[:, j - 1]),
                    prev[:, j - 1]
                )
                curr[:, j] = cost + best
            prev = curr

        distances[start:end] = np.sqrt(prev[:, m] / max(n_bands, m))

    dist_map = distances.reshape(rows, cols)
    # Normalize: exp(−d) maps [0, ∞) → (0, 1]
    return np.clip(np.exp(-dist_map.astype(np.float64)), 0.0, 1.0)


# ── 13. SSIM — Spectral Structural Similarity Index ──────────────────────

def match_ssim(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Spectral SSIM (adapted from Wang et al. 2004 to 1-D spectra).

    SSIM = luminance × contrast × structure
    """
    cube = _safe_cube(cube)
    n = cube.shape[0]
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    C3 = C2 / 2.0

    mu_r = ref.mean()
    mu_p = cube.mean(axis=0)
    sigma_r = ref.std() + EPS
    sigma_p = cube.std(axis=0) + EPS
    sigma_rp = np.mean(
        (ref - mu_r)[:, np.newaxis, np.newaxis] * (cube - mu_p[np.newaxis]),
        axis=0
    )

    luminance = (2 * mu_r * mu_p + C1) / (mu_r ** 2 + mu_p ** 2 + C1)
    contrast = (2 * sigma_r * sigma_p + C2) / (sigma_r ** 2 + sigma_p ** 2 + C2)
    structure = (sigma_rp + C3) / (sigma_r * sigma_p + C3)

    ssim = luminance * contrast * structure
    return np.clip((ssim + 1.0) / 2.0, 0.0, 1.0)


# ── 14. JSD — Jensen-Shannon Divergence ───────────────────────────────────

def match_jsd(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """JSD = (KLD(p‖m) + KLD(q‖m))/2 where m=(p+q)/2; similarity = exp(−JSD)."""
    cube = _safe_cube(cube)
    p = (ref / (ref.sum() + EPS) + EPS)[:, np.newaxis, np.newaxis]  # (n,1,1)
    q = cube / (cube.sum(axis=0, keepdims=True) + EPS) + EPS
    m = (p + q) / 2.0
    kld_pm = np.sum(p * np.log(p / m), axis=0)
    kld_qm = np.sum(q * np.log(q / m), axis=0)
    jsd = (kld_pm + kld_qm) / 2.0
    return np.clip(np.exp(-jsd), 0.0, 1.0)


# ── 15. Bhattacharyya Coefficient ─────────────────────────────────────────

def match_bhatt(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """BC = Σ√(p_i · q_i) where p, q are probability distributions."""
    cube = _safe_cube(cube)
    p = np.maximum(ref / (ref.sum() + EPS), 0.0)
    q = np.maximum(
        cube / (cube.sum(axis=0, keepdims=True) + EPS), 0.0
    )
    bc = np.sum(
        np.sqrt(p[:, np.newaxis, np.newaxis] * q), axis=0
    )
    return np.clip(bc, 0.0, 1.0)


# ── 16. MTF — Matched / Tuned Filter ─────────────────────────────────────

def match_mtf(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Matched Tuned Filter (Reed & Yu 1990).

    Assumes background is the image mean; filter maximizes SNR for target.
    w = Σ⁻¹(d − μ) / (d − μ)ᵀ Σ⁻¹(d − μ)
    score = wᵀ(x − μ)
    """
    cube = _safe_cube(cube)
    n, rows, cols = cube.shape
    flat = cube.reshape(n, rows * cols)    # (n, n_pix)

    mu = flat.mean(axis=1)                 # (n,)
    centered = flat - mu[:, np.newaxis]    # (n, n_pix)

    # Covariance  (n × n)
    cov = (centered @ centered.T) / (flat.shape[1] - 1)
    cov += np.eye(n) * 1e-6 * np.trace(cov) / n  # regularise

    d = ref - mu
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        gs.warning("MTF: covariance matrix is singular; falling back to SAM")
        return match_sam(cube, ref)

    w = cov_inv @ d                        # (n,)
    denom = d @ w
    if abs(denom) < EPS:
        return np.zeros((rows, cols), dtype=np.float32)
    w /= denom

    scores = (w @ centered).reshape(rows, cols)
    lo, hi = np.percentile(scores, [1, 99])
    rng = hi - lo if hi > lo else 1.0
    return np.clip((scores - lo) / rng, 0.0, 1.0)


# ── 17. CEM — Constrained Energy Minimization ─────────────────────────────

def match_cem(cube: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """CEM (Chang & Heinz 2000): minimise output energy, constrain target=1.

    w = Σ⁻¹ d / dᵀ Σ⁻¹ d  (no background subtraction)
    score = wᵀ x
    """
    cube = _safe_cube(cube)
    n, rows, cols = cube.shape
    flat = cube.reshape(n, rows * cols)

    cov = (flat @ flat.T) / flat.shape[1]
    cov += np.eye(n) * 1e-6 * np.trace(cov) / n

    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        gs.warning("CEM: covariance matrix is singular; falling back to SAM")
        return match_sam(cube, ref)

    d = ref.astype(np.float64)
    w = cov_inv @ d
    denom = d @ w
    if abs(denom) < EPS:
        return np.zeros((rows, cols), dtype=np.float32)
    w /= denom

    scores = (w @ flat).reshape(rows, cols)
    lo, hi = np.percentile(scores, [1, 99])
    rng = hi - lo if hi > lo else 1.0
    return np.clip((scores - lo) / rng, 0.0, 1.0)


# ── 18. Ensemble — Rank-based Fusion ──────────────────────────────────────

def match_ensemble(score_maps: dict[str, np.ndarray]) -> np.ndarray:
    """Borda-count rank aggregation across all non-ensemble score maps.

    For each method, ranks pixels from worst (rank 0) to best (rank n_pix).
    Sum ranks, normalize to [0, 1].
    """
    maps = {k: v for k, v in score_maps.items() if k != 'ensemble'}
    if not maps:
        gs.fatal("Ensemble requires at least one other method to be computed")
    rows, cols = next(iter(maps.values())).shape
    n_pix = rows * cols
    rank_sum = np.zeros(n_pix, dtype=np.float64)
    for name, smap in maps.items():
        flat = smap.ravel().astype(np.float64)
        order = np.argsort(flat)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_pix)
        rank_sum += ranks
    rank_sum /= (len(maps) * n_pix)
    return rank_sum.reshape(rows, cols)

# ---------------------------------------------------------------------------
# Consensus hotspot analysis — multi-method calibrated probability fusion
# ---------------------------------------------------------------------------

# ── Scipy fallbacks (used when scipy is not available) ────────────────────

def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Normal percent-point function (inverse CDF) via erf approximation.

    Abramowitz & Stegun 26.2.17, max error ≈ 4.5 × 10⁻⁴.
    Accepts array input; safe for p ∈ (0, 1).
    """
    p = np.clip(p, EPS, 1.0 - EPS)
    sign = np.where(p < 0.5, -1.0, 1.0)
    t = np.sqrt(-2.0 * np.log(np.where(p < 0.5, p, 1.0 - p)))
    num = 2.515517 + 0.802853 * t + 0.010328 * t ** 2
    den = 1.0 + 1.432788 * t + 0.189269 * t ** 2 + 0.001308 * t ** 3
    return sign * (t - num / den)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Normal CDF via numpy erf — exact."""
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))


def _chi2_sf(x: np.ndarray, df: int) -> np.ndarray:
    """Chi-squared survival function P(χ²(df) ≥ x).

    Uses scipy.special.chdtrc when available; falls back to the
    Wilson-Hilferty normal approximation otherwise.
    """
    try:
        from scipy.special import chdtrc
        return np.asarray(chdtrc(df, x), dtype=np.float64)
    except ImportError:
        z = (x - df) / np.sqrt(2.0 * df)
        return 1.0 - _norm_cdf(z)


# ── Step 1: Empirical CDF calibration ────────────────────────────────────

def calibrate_scores(
        score_maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert each score map to a calibrated probability via empirical CDF.

    For method *m* and pixel *i*:

        p_m(i) = rank(score_m(i)) / n_pixels  ∈ (0, 1]

    This rank transform is the probability integral transform applied to the
    empirical distribution: it removes scale bias (an ED map in [0, 0.3]
    would otherwise be dominated by a SAM map in [0.7, 1.0]) and ensures
    every method has a uniform marginal distribution before fusion.

    High calibrated value → this pixel is among the top-scoring pixels for
    this method, regardless of the method's own output range.

    Skips 'ensemble' and 'consensus' which are already meta-maps.
    """
    calibrated: dict[str, np.ndarray] = {}
    for name, smap in score_maps.items():
        if name in ('ensemble', 'consensus'):
            continue
        flat = smap.ravel().astype(np.float64)
        n = len(flat)
        order = np.argsort(flat)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        calibrated[name] = (ranks / n).reshape(smap.shape)
    return calibrated


# ── Step 2: Diversity weights ─────────────────────────────────────────────

def compute_diversity_weights(
        calibrated: dict[str, np.ndarray]) -> dict[str, float]:
    """Assign each method a weight inversely proportional to its mean
    absolute Pearson correlation with all other methods.

    Rationale: correlated methods (e.g. SAM and SCA both measure spectral
    shape) carry redundant information.  Treating them as independent would
    over-count their shared evidence.  By weighting each method by its
    *diversity* relative to the rest of the pool, the consensus is not
    dominated by clusters of similar methods.

    Weight formula:

        w_i = 1 / (mean_j≠i |r_ij| + floor)

    where r_ij is the Pearson correlation between the rank-calibrated maps
    of methods i and j, and *floor* = 0.15 prevents complete silencing of
    any method.  Weights are then rescaled so their mean equals 1 (no
    overall gain/loss relative to uniform weighting).

    Returns a dict {method_name → float weight}.
    """
    names = list(calibrated.keys())
    k = len(names)
    if k == 1:
        return {names[0]: 1.0}

    # (k, n_pixels) matrix of rank-calibrated probabilities
    flat = np.array([calibrated[nm].ravel() for nm in names], dtype=np.float64)

    # Pearson correlation matrix via centered + normalized dot product
    centered = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + EPS
    normed = centered / norms
    corr = normed @ normed.T   # (k, k), diagonal = 1

    # Mean absolute off-diagonal correlation per method
    np.fill_diagonal(corr, 0.0)
    mean_abs = np.abs(corr).sum(axis=1) / max(k - 1, 1)

    # Diversity weight: less correlated → higher weight
    raw = 1.0 / (mean_abs + 0.15)
    raw *= k / raw.sum()   # rescale so mean weight = 1

    return {nm: float(w) for nm, w in zip(names, raw)}


# ── Step 3: Probability fusion ────────────────────────────────────────────

def fuse_probabilities(
        calibrated: dict[str, np.ndarray],
        weights: dict[str, float],
        mode: str) -> np.ndarray:
    """Fuse calibrated per-method probability maps into one hotspot probability.

    All modes operate on rank-calibrated probabilities p_m ∈ (0, 1] produced
    by :func:`calibrate_scores`.

    Parameters
    ----------
    calibrated : dict[method → (rows, cols) array in (0, 1]]
    weights    : dict[method → float] from :func:`compute_diversity_weights`
    mode       : one of FUSION_MODES

    Returns
    -------
    (rows, cols) hotspot probability in [0, 1]

    Mode descriptions
    -----------------
    rank_product
        Weighted geometric mean of rank fractions.  A pixel scores high only
        when EVERY method (weighted by diversity) simultaneously ranks it near
        the top.  No distributional assumptions; always well-defined.

        P = exp(Σ w_i · ln p_i)

    fisher
        Fisher (1932) combined probability test.  The "p-value" for each
        method is q_i = 1 − p_i (high similarity → low q → strong evidence).
        The weighted test statistic T = −2 Σ w_i ln(q_i) is compared to a
        chi-squared distribution.  Returns 1 − combined_p_value so that the
        hotspot map is high where the null hypothesis (pixel is background)
        is rejected.

    stouffer
        Stouffer (1949) weighted Z-score method.  Each calibrated probability
        p_i is mapped to its standard-normal equivalent z_i = Φ⁻¹(p_i).
        The combined Z = Σ w_i z_i / √(Σ w_i²) is then mapped back to [0,1]
        via the normal CDF.  The diversity weights enter naturally into the
        denominator and are handled exactly.

    group_product
        Geometric mean (AND) within each method group (geometric and distance
        and information etc.), then arithmetic mean (OR) across groups.
        Respects the independence structure of the method taxonomy: within a
        group methods are correlated so one consensus is formed; across groups
        independent evidence is combined more leniently.

    harmonic
        Weighted harmonic mean: P = 1 / Σ(w_i / p_i).  A single low-scoring
        method strongly drags down the result.  Use when high confidence
        requires every metric to simultaneously agree.

    min
        P = min_m p_m.  Absolute strictest: the combined probability cannot
        exceed the worst-performing method for any pixel.
    """
    names = [n for n in calibrated if n not in ('ensemble', 'consensus')]
    k = len(names)
    if k == 0:
        gs.fatal("consensus: no base method score maps available for fusion")

    rows, cols = next(c for n, c in calibrated.items()
                      if n not in ('ensemble', 'consensus')).shape
    n_pix = rows * cols

    # (k, n_pix) probability matrix
    P = np.array([calibrated[nm].ravel() for nm in names], dtype=np.float64)
    P = np.clip(P, EPS, 1.0 - EPS)

    # (k,) diversity weights, normalised to sum=1
    W = np.array([weights.get(nm, 1.0) for nm in names], dtype=np.float64)
    W /= W.sum()

    # ── rank_product ──────────────────────────────────────────────────────
    if mode == 'rank_product':
        log_P = np.log(P)
        result = np.exp((W[:, np.newaxis] * log_P).sum(axis=0))

    # ── fisher ────────────────────────────────────────────────────────────
    elif mode == 'fisher':
        # q_i = 1 - p_i  is the "p-value" under H0 (pixel is background)
        Q = 1.0 - P
        Q = np.clip(Q, EPS, 1.0 - EPS)
        # Weighted test statistic: T = -2 Σ w_i ln(q_i)
        # Under H0 with uniform weights this follows χ²(2k);
        # with non-uniform weights we use an effective df = 2 * (Σw)²/Σw² ≈ 2k
        T = -2.0 * (W[:, np.newaxis] * np.log(Q)).sum(axis=0)
        eff_df = int(round(2.0 * W.sum() ** 2 / (W ** 2).sum()))
        eff_df = max(eff_df, 2)
        combined_pval = _chi2_sf(T, eff_df)
        result = 1.0 - np.clip(combined_pval, 0.0, 1.0)

    # ── stouffer ──────────────────────────────────────────────────────────
    elif mode == 'stouffer':
        # z-score equivalent of each calibrated probability
        try:
            from scipy.special import ndtri, ndtr
            Z = ndtri(P)
            Z_combined = (W[:, np.newaxis] * Z).sum(axis=0) / np.sqrt((W ** 2).sum())
            result = ndtr(Z_combined)
        except ImportError:
            Z = _norm_ppf(P)
            Z_combined = (W[:, np.newaxis] * Z).sum(axis=0) / np.sqrt((W ** 2).sum())
            result = _norm_cdf(Z_combined)

    # ── group_product ─────────────────────────────────────────────────────
    elif mode == 'group_product':
        # Map each active method to its group; ungrouped methods go to '_other'
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        active_groups: dict[str, list[int]] = {}
        for nm in names:
            placed = False
            for gname, gmembers in METHOD_GROUPS.items():
                if nm in gmembers:
                    active_groups.setdefault(gname, []).append(name_to_idx[nm])
                    placed = True
                    break
            if not placed:
                active_groups.setdefault('_other', []).append(name_to_idx[nm])

        group_results: list[np.ndarray] = []
        group_wts: list[float] = []

        for gname, idxs in active_groups.items():
            g_P = P[idxs]   # (|g|, n_pix)
            g_W = W[idxs]
            g_W = g_W / g_W.sum()
            # Geometric mean within group (AND: all within-group must agree)
            log_g = np.log(np.clip(g_P, EPS, 1.0))
            g_score = np.exp((g_W[:, np.newaxis] * log_g).sum(axis=0))
            group_results.append(g_score)
            group_wts.append(float(len(idxs)))   # weight group by member count

        gw = np.array(group_wts)
        gw /= gw.sum()
        # Arithmetic mean across groups (OR: any independent evidence source suffices)
        result = sum(w * s for w, s in zip(gw, group_results))

    # ── harmonic ──────────────────────────────────────────────────────────
    elif mode == 'harmonic':
        # Weighted harmonic mean: 1 / Σ(w_i / p_i)
        result = 1.0 / (W[:, np.newaxis] / P).sum(axis=0)

    # ── min ───────────────────────────────────────────────────────────────
    elif mode == 'min':
        result = P.min(axis=0)

    else:
        gs.fatal(f"Unknown fusion mode '{mode}'. Valid: {', '.join(FUSION_MODES)}")

    return np.clip(result.reshape(rows, cols), 0.0, 1.0)


# ── Step 4: Per-pixel agreement statistics ────────────────────────────────

def compute_consensus_stats(
        calibrated: dict[str, np.ndarray],
        probability: np.ndarray,
        agreement_threshold: float = 0.80) -> dict[str, np.ndarray]:
    """Compute per-pixel statistics on the ensemble of calibrated score maps.

    Returns
    -------
    dict with keys:

    agreement
        Fraction of methods whose calibrated probability exceeds
        *agreement_threshold* at each pixel.  Values near 1 mean near-unanimous
        vote for a hotspot; near 0 means near-unanimous rejection.

    entropy
        Normalized Shannon entropy of the per-pixel score vector:

            H(i) = −Σ_m p̂_m(i) ln p̂_m(i) / ln k   ∈ [0, 1]

        where p̂ is the row-normalized score vector.  **0 = full agreement**
        (all methods give the same relative assessment); **1 = maximum
        disagreement** (scores are as spread as possible).

    conflict
        Float mask marking pixels where the *combined probability is high*
        (top-25%) but *entropy is also high* (top-25%).  These are candidates
        that some methods strongly support and others strongly reject — the
        most scientifically ambiguous hotspots, warranting manual inspection.

    spread
        Standard deviation of the calibrated scores per pixel [0, 0.5].
        Complements entropy: high spread with high combined probability
        indicates one method is a strong outlier.
    """
    names = [n for n in calibrated if n not in ('ensemble', 'consensus')]
    k = len(names)
    if k == 0:
        return {}

    rows, cols = probability.shape
    P = np.array([calibrated[nm].ravel() for nm in names], dtype=np.float64)  # (k, n_pix)

    # ── agreement ─────────────────────────────────────────────────────────
    above = (P > agreement_threshold).sum(axis=0) / k   # (n_pix,)

    # ── entropy ───────────────────────────────────────────────────────────
    # Normalize per pixel so the score vector sums to 1
    col_sum = P.sum(axis=0, keepdims=True) + EPS
    P_norm = P / col_sum
    log_P = np.where(P_norm > EPS, np.log(P_norm + EPS), 0.0)
    raw_h = -(P_norm * log_P).sum(axis=0)
    max_h = np.log(k) if k > 1 else 1.0
    entropy = raw_h / max_h   # (n_pix,)

    # ── spread ────────────────────────────────────────────────────────────
    spread = P.std(axis=0)   # (n_pix,)

    # ── conflict ──────────────────────────────────────────────────────────
    prob_flat = probability.ravel()
    p75_prob = np.percentile(prob_flat, 75)
    p75_ent  = np.percentile(entropy, 75)
    conflict = ((prob_flat > p75_prob) & (entropy > p75_ent)).astype(np.float32)

    return {
        'agreement': above.reshape(rows, cols).astype(np.float32),
        'entropy':   entropy.reshape(rows, cols).astype(np.float32),
        'conflict':  conflict.reshape(rows, cols),
        'spread':    spread.reshape(rows, cols).astype(np.float32),
    }


# ── Master pipeline ───────────────────────────────────────────────────────

def run_consensus_analysis(
        cube: np.ndarray,
        ref_proc: np.ndarray,
        wls: np.ndarray,
        shift_win: int,
        existing_score_maps: dict[str, np.ndarray],
        fusion_mode: str = 'rank_product',
        agreement_threshold: float = 0.80,
        skip_slow: bool = False,
        verbose: bool = False,
) -> dict:
    """Run all base methods, calibrate scores, fuse into hotspot probability.

    This is the master consensus function.  It drives a four-step pipeline:

    1. **Run base methods** — any BASE_METHOD not already in
       *existing_score_maps* is computed against the cube.  The expensive
       continuum-removal and DTW methods can be skipped with *skip_slow=True*.

    2. **Calibrate** — each score map is rank-transformed to a uniform
       probability in (0, 1] via :func:`calibrate_scores`.  This removes
       scale and range bias between heterogeneous metrics.

    3. **Diversity weights** — inter-method Pearson correlation is computed
       and each method is down-weighted in proportion to its average
       correlation with others, so correlated method clusters (e.g. all
       geometric methods) do not dominate.

    4. **Fuse** — :func:`fuse_probabilities` combines the calibrated
       probabilities according to *fusion_mode* (see its docstring for the
       mathematical definition of each mode).

    5. **Statistics** — :func:`compute_consensus_stats` produces per-pixel
       agreement count, entropy, conflict mask, and spread.

    Parameters
    ----------
    cube : (n_bands, rows, cols) preprocessed float64 pixel cube
    ref_proc : (n_bands,) preprocessed reference spectrum
    wls : (n_bands,) sensor wavelengths in nm
    shift_win : Sakoe-Chiba / xcorr window in bands
    existing_score_maps : already-computed method score maps (not rerun)
    fusion_mode : one of FUSION_MODES
    agreement_threshold : calibrated-probability threshold for agreement count
    skip_slow : skip cr_sam, cr_ed, dtw (per-pixel convex-hull / DTW loop)
    verbose : print per-step progress to gs.verbose

    Returns
    -------
    dict with keys:

    probability  (rows, cols) float64
        Fused hotspot probability [0, 1] — the primary output.
    agreement    (rows, cols) float32
        Fraction of methods voting above *agreement_threshold* [0, 1].
    entropy      (rows, cols) float32
        Method agreement entropy [0, 1] — 0 = unanimous, 1 = maximal conflict.
    conflict     (rows, cols) float32
        Binary mask of pixels with high probability AND high entropy.
    spread       (rows, cols) float32
        Std dev of calibrated scores per pixel.
    weights      dict[str → float]
        Diversity weights used in fusion (useful for audit).
    score_maps   dict[str → (rows,cols)]
        All base-method score maps (input + newly computed).
    calibrated   dict[str → (rows,cols)]
        Rank-calibrated probability maps for each method.
    """
    SLOW = {'cr_sam', 'cr_ed', 'dtw'}

    # ── 1. Compute missing base methods ───────────────────────────────────
    score_maps = dict(existing_score_maps)
    to_run = [m for m in BASE_METHODS
              if m not in score_maps and (not skip_slow or m not in SLOW)]

    if verbose:
        gs.verbose(f"consensus: {len(to_run)} method(s) to compute: "
                   f"{', '.join(to_run) if to_run else '(none — all cached)'}")

    for idx, mth in enumerate(to_run):
        if verbose:
            gs.verbose(f"  [{idx + 1}/{len(to_run)}] {METHOD_LABELS.get(mth, mth)}")
        score_maps[mth] = compute_method(mth, cube, ref_proc, wls,
                                         shift_win, score_maps)

    # ── 2. Calibrate ──────────────────────────────────────────────────────
    if verbose:
        gs.verbose("consensus: calibrating scores (empirical CDF rank transform)")
    calibrated = calibrate_scores(score_maps)

    if len(calibrated) == 0:
        gs.fatal("consensus: no base-method score maps available for fusion")

    # ── 3. Diversity weights ──────────────────────────────────────────────
    if verbose:
        gs.verbose(f"consensus: computing diversity weights "
                   f"({len(calibrated)} methods, correlation matrix "
                   f"{len(calibrated)}×{len(calibrated)})")
    weights = compute_diversity_weights(calibrated)

    if verbose:
        for nm, w in sorted(weights.items(), key=lambda x: -x[1]):
            corr_note = " (down-weighted: correlated)" if w < 0.75 else ""
            gs.verbose(f"    {nm:<12s} w={w:.3f}{corr_note}")

    # ── 4. Fuse ───────────────────────────────────────────────────────────
    if verbose:
        gs.verbose(f"consensus: fusing with mode='{fusion_mode}'")
    probability = fuse_probabilities(calibrated, weights, fusion_mode)

    # ── 5. Statistics ─────────────────────────────────────────────────────
    stats = compute_consensus_stats(calibrated, probability, agreement_threshold)

    # ── Summary (always shown) ────────────────────────────────────────────
    p_flat = probability.ravel()
    n_pix = p_flat.size
    for thr in (0.9, 0.8, 0.7, 0.5):
        n_hot = int((p_flat >= thr).sum())
        if n_hot > 0:
            gs.verbose(f"consensus: {n_hot} pixels ({100.*n_hot/n_pix:.2f}%) ≥ {thr:.1f} "
                       f"(mode={fusion_mode})")
            break

    return {
        'probability': probability,
        'agreement':   stats.get('agreement'),
        'entropy':     stats.get('entropy'),
        'conflict':    stats.get('conflict'),
        'spread':      stats.get('spread'),
        'weights':     weights,
        'score_maps':  score_maps,
        'calibrated':  calibrated,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def compute_method(method: str, cube: np.ndarray, ref: np.ndarray,
                   wavelengths: np.ndarray,
                   shift_window: int,
                   score_maps: dict) -> np.ndarray:
    """Run a single matching method and return the similarity map."""
    if method == 'sam':
        return match_sam(cube, ref)
    elif method == 'sid':
        return match_sid(cube, ref)
    elif method == 'sid_sam':
        return match_sid_sam(cube, ref)
    elif method == 'ed':
        return match_ed(cube, ref)
    elif method == 'sad':
        return match_sad(cube, ref)
    elif method == 'sca':
        return match_sca(cube, ref)
    elif method == 'cr_sam':
        return match_cr_sam(cube, ref, wavelengths)
    elif method == 'cr_ed':
        return match_cr_ed(cube, ref, wavelengths)
    elif method == 'gd1':
        return match_gd1(cube, ref, wavelengths)
    elif method == 'gd2':
        return match_gd2(cube, ref, wavelengths)
    elif method == 'xcorr':
        return match_xcorr(cube, ref, max_lag=shift_window)
    elif method == 'dtw':
        return match_dtw(cube, ref, window=shift_window)
    elif method == 'ssim':
        return match_ssim(cube, ref)
    elif method == 'jsd':
        return match_jsd(cube, ref)
    elif method == 'bhatt':
        return match_bhatt(cube, ref)
    elif method == 'mtf':
        return match_mtf(cube, ref)
    elif method == 'cem':
        return match_cem(cube, ref)
    elif method == 'ensemble':
        return match_ensemble(score_maps)
    elif method == 'consensus':
        # consensus needs the full pipeline; it must be handled by the caller
        # (main / point_analysis) rather than dispatched here.
        gs.fatal("compute_method: 'consensus' must be handled by the calling "
                 "pipeline, not dispatched here.  This is a bug.")
    else:
        gs.fatal(f"Unknown method: {method}")

# ---------------------------------------------------------------------------
# Point mode: single-pixel analysis
# ---------------------------------------------------------------------------


def point_analysis(spec_pixel: np.ndarray, ref: np.ndarray,
                   wavelengths: np.ndarray,
                   methods: list[str],
                   shift_window: int,
                   flag_c: bool,
                   flag_z: bool) -> dict[str, float]:
    """Compute all requested similarity scores for one pixel spectrum."""

    def _1d_to_cube(s):
        return s[:, np.newaxis, np.newaxis]

    if flag_z:
        ref_proc = to_prob_simplex(ref)
        pix_proc = to_prob_simplex(spec_pixel)
    else:
        ref_proc = ref.copy()
        pix_proc = spec_pixel.copy()

    if flag_c:
        ref_proc = continuum_remove(ref_proc, wavelengths)
        pix_proc = continuum_remove(pix_proc, wavelengths)

    scores: dict[str, float] = {}
    score_maps: dict[str, np.ndarray] = {}
    do_consensus_pt = 'consensus' in methods

    for mth in methods:
        if mth in ('ensemble', 'consensus'):
            continue
        cube_1 = _1d_to_cube(pix_proc)
        ref_use = ref_proc
        if mth in ('cr_sam', 'cr_ed') and flag_c:
            mth_run = 'sam' if mth == 'cr_sam' else 'ed'
            result = compute_method(mth_run, cube_1, ref_use, wavelengths,
                                    shift_window, score_maps)
        else:
            result = compute_method(mth, cube_1, ref_use, wavelengths,
                                    shift_window, score_maps)
        val = float(result[0, 0])
        scores[mth] = val
        score_maps[mth] = result

    if 'ensemble' in methods and len(score_maps) > 0:
        ens = match_ensemble(score_maps)
        scores['ensemble'] = float(ens[0, 0])

    if do_consensus_pt and len(score_maps) > 0:
        # Calibrate the single-pixel score maps (trivially rank-1, so
        # calibrated = 1.0 for the one pixel; use raw scores for display
        # but run the full pipeline to show weights and fusion logic).
        cal = {nm: np.array([[v]], dtype=np.float64)
               for nm, v in scores.items() if nm not in ('ensemble', 'consensus')}
        if cal:
            weights = compute_diversity_weights(cal)
            # For a single pixel there is no rank variation, so we just show
            # the weighted geometric mean of the raw scores as consensus.
            names = list(cal.keys())
            W = np.array([weights.get(nm, 1.0) for nm in names])
            W /= W.sum()
            raw_scores = np.array([scores[nm] for nm in names])
            consensus_val = float(np.exp((W * np.log(np.clip(raw_scores, EPS, 1.0))).sum()))
            scores['consensus'] = consensus_val
            scores['_weights'] = weights   # expose to caller for display

    return scores

# ---------------------------------------------------------------------------
# Output: color table and metadata
# ---------------------------------------------------------------------------


def set_similarity_colors(output_name: str) -> None:
    """Apply a perceptually uniform blue–yellow–red color ramp for similarity."""
    rules = """\
0.00 0:0:128
0.10 0:0:255
0.25 0:128:255
0.40 0:220:220
0.55 128:255:0
0.70 255:220:0
0.85 255:128:0
1.00 200:0:0
"""
    gs.write_command('r.colors', map=output_name, rules='-',
                     stdin=rules, quiet=True)


def set_raster_metadata(output_name: str, raster3d: str,
                        method: str, ref_desc: str) -> None:
    title = f"Spectral similarity ({METHOD_LABELS.get(method, method)})"
    description = (f"Similarity to reference spectrum [{ref_desc}] "
                   f"using {METHOD_LABELS.get(method, method)} "
                   f"from i.hyper.sleuth on {raster3d}. Range: 0=no match, 1=perfect match.")
    try:
        gs.run_command('r.support', map=output_name,
                       title=title, description=description, quiet=True)
    except Exception as e:
        gs.warning(f"Could not set metadata for {output_name}: {e}")

# ---------------------------------------------------------------------------
# Info mode
# ---------------------------------------------------------------------------


def print_info(bands: list[dict], ref_wls: np.ndarray, ref_vals: np.ndarray,
               resampled: np.ndarray, methods: list[str],
               lut: Optional['WavelengthLUT'] = None) -> None:
    sep = "=" * 68
    wls = [b['wavelength'] for b in bands]

    gs.message(sep)
    gs.message("i.hyper.sleuth — Spectral Target Detection")
    gs.message(sep)
    gs.message(f"Sensor  : {len(bands)} bands, "
               f"{min(wls):.1f} – {max(wls):.1f} nm")
    gs.message(f"Reference: {len(ref_wls)} points, "
               f"{ref_wls[0]:.1f} – {ref_wls[-1]:.1f} nm, "
               f"reflectance [{ref_vals.min():.4f}, {ref_vals.max():.4f}]")
    gs.message(f"Resampled: {len(resampled)} sensor bands, "
               f"range [{resampled.min():.4f}, {resampled.max():.4f}]")
    if lut is not None:
        gs.message(f"LUT      : {lut.coverage_report()}")
        n_oor = int((~lut.valid_dst).sum())
        if n_oor:
            gs.message(f"           {n_oor} sensor band(s) outside reference range "
                       f"→ edge-fill applied; restrict with min/max_wavelength= "
                       f"to [{lut.overlap_lo:.0f}, {lut.overlap_hi:.0f}] nm "
                       "to avoid edge-fill bias")
        n_hidden = int((~lut.valid_src).sum())
        if n_hidden:
            gs.message(f"           {n_hidden} reference point(s) outside sensor range "
                       "→ those features are not observable in this dataset")
    gs.message(" ")
    gs.message("Requested methods:")
    for m in methods:
        need_global = "  [needs image covariance]" if m in GLOBAL_STATS_METHODS else ""
        need_cr = "  [continuum removal per pixel — can be slow]" \
                  if m in ('cr_sam', 'cr_ed') else ""
        gs.message(f"  {m:10s}  {METHOD_LABELS.get(m,'')}{need_global}{need_cr}")
    gs.message(sep)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(options: dict, flags: dict) -> int:
    raster3d    = options['input']
    output      = options['output']
    ref_inline  = options.get('reference') or ''
    ref_file    = options.get('reference_file') or ''
    methods_str = options.get('method', 'sam')
    out_prefix  = options.get('output_prefix') or ''
    resample_m  = options.get('resample', 'linear')
    normalize_m = options.get('normalize', 'none')
    shift_win   = int(options.get('shift_window', 3))
    fusion_mode = options.get('fusion_mode', 'rank_product')
    agr_thresh  = float(options.get('agreement_threshold', 0.80))
    min_wl      = float(options['min_wavelength']) if options.get('min_wavelength') else None
    max_wl      = float(options['max_wavelength']) if options.get('max_wavelength') else None
    coords_str  = options.get('coordinates') or ''

    flag_n = flags.get('n', False)
    flag_i = flags.get('i', False)
    flag_v = flags.get('v', False)
    flag_c = flags.get('c', False)
    flag_p = flags.get('p', False)
    flag_z = flags.get('z', False)

    methods = [m.strip() for m in methods_str.split(',') if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        gs.fatal(f"Unknown method(s): {', '.join(unknown)}. "
                 f"Valid: {', '.join(ALL_METHODS)}")
    if fusion_mode not in FUSION_MODES:
        gs.fatal(f"Unknown fusion_mode '{fusion_mode}'. "
                 f"Valid: {', '.join(FUSION_MODES)}")

    do_consensus = 'consensus' in methods
    # consensus is mutually exclusive with ensemble in the same run
    # (consensus subsumes ensemble and much more)
    if do_consensus and 'ensemble' in methods:
        gs.warning("Both 'consensus' and 'ensemble' requested; "
                   "'ensemble' will be skipped — consensus subsumes it.")
        methods = [m for m in methods if m != 'ensemble']

    if flag_p and not coords_str:
        gs.fatal("Point mode (-p) requires coordinates= to be set")
    if 'ensemble' in methods and len(methods) < 2:
        gs.fatal("ensemble requires at least one additional method")

    # ------------------------------------------------------------------
    # Step 1: Reference spectrum
    # ------------------------------------------------------------------
    if ref_inline:
        ref_wls, ref_vals = parse_reference_inline(ref_inline)
        ref_desc = f"inline ({len(ref_wls)} points)"
    else:
        ref_wls, ref_vals = parse_reference_file(ref_file)
        ref_desc = os.path.basename(ref_file)

    if np.any(ref_vals < 0):
        gs.warning("Reference contains negative reflectance values — check units")
    if np.any(ref_vals > 2):
        gs.warning("Reference contains reflectance > 2 — values may be in % not [0,1]; "
                   "ensure pixel data and reference are on the same scale")

    # ------------------------------------------------------------------
    # Step 2: Band metadata
    # ------------------------------------------------------------------
    gs.message(f"Scanning hyperspectral bands in: {raster3d}")
    bands = get_band_info(raster3d, only_valid=flag_n, min_wl=min_wl, max_wl=max_wl)
    wls = np.array([b['wavelength'] for b in bands], dtype=np.float64)
    gs.message(f"Found {len(bands)} usable bands: {wls[0]:.1f} – {wls[-1]:.1f} nm")

    # ------------------------------------------------------------------
    # Step 3: Build WavelengthLUT (ref → sensor) — computed once, reused
    #         everywhere: resampling, overlap restriction, diagnostics.
    # ------------------------------------------------------------------
    lut = WavelengthLUT(ref_wls, wls, fill='edge')

    if not lut.has_overlap:
        gs.fatal(
            f"No wavelength overlap between reference "
            f"({ref_wls[0]:.1f}–{ref_wls[-1]:.1f} nm) and sensor "
            f"({wls[0]:.1f}–{wls[-1]:.1f} nm).  Check units or wavelength range."
        )

    # Diagnostic: sensor bands outside reference range get edge-fill values.
    # Those bands contribute noise to the similarity score; advise restriction.
    n_oor = int((~lut.valid_dst).sum())
    if n_oor:
        gs.warning(
            f"{n_oor} of {len(bands)} sensor bands fall outside the reference "
            f"wavelength range [{ref_wls[0]:.1f}–{ref_wls[-1]:.1f} nm] and "
            "will receive edge-fill reflectance values.  "
            f"Use min_wavelength={lut.overlap_lo:.0f} max_wavelength="
            f"{lut.overlap_hi:.0f} to restrict matching to the overlap region "
            "and avoid similarity bias."
        )
    n_hidden = int((~lut.valid_src).sum())
    if n_hidden:
        gs.warning(
            f"{n_hidden} reference point(s) lie outside the sensor range "
            f"[{wls[0]:.1f}–{wls[-1]:.1f} nm] and cannot be matched."
        )
    if flag_v:
        gs.verbose(f"WavelengthLUT: {lut.coverage_report()}")
        gs.verbose(f"  Precomputed {len(lut.left_idx)} index pairs "
                   f"(left_idx, right_idx, alpha) for O(n) apply()")

    # ------------------------------------------------------------------
    # Step 4: Resample reference to sensor wavelengths
    #         Uses LUT.apply() — O(n_sensor), no binary search.
    # ------------------------------------------------------------------
    ref_resampled = resample_reference(ref_wls, ref_vals, wls,
                                       method=resample_m, lut=lut)

    if flag_i:
        print_info(bands, ref_wls, ref_vals, ref_resampled, methods, lut=lut)
        return 0

    # ------------------------------------------------------------------
    # Step 5: Apply preprocessing to reference
    # ------------------------------------------------------------------
    ref_proc = ref_resampled.copy()
    if flag_z:
        ref_proc = to_prob_simplex(ref_proc)
    if normalize_m != 'none':
        ref_proc = normalize_spectrum(ref_proc, normalize_m)
    if flag_c and 'cr_sam' not in methods and 'cr_ed' not in methods:
        # Global -c flag: pre-remove continuum from reference
        ref_proc = continuum_remove(ref_proc, wls)

    # ------------------------------------------------------------------
    # Step 5: Point mode
    # ------------------------------------------------------------------
    if flag_p:
        east, north = (float(v) for v in coords_str.split(','))
        gs.message(f"Point mode: extracting spectrum at E={east}, N={north}")
        spec_pixel = read_pixel_spectrum(bands, raster3d, east, north)
        if np.all(np.isnan(spec_pixel)):
            gs.fatal("All band values are null at the given coordinates")

        if flag_z:
            spec_pixel = to_prob_simplex(spec_pixel)
        if normalize_m != 'none':
            spec_pixel = normalize_spectrum(spec_pixel, normalize_m)

        scores = point_analysis(spec_pixel, ref_proc, wls, methods,
                                shift_win, flag_c, flag_z=False)

        sep = "-" * 64
        gs.message(sep)
        gs.message(f"i.hyper.sleuth — Point analysis  E={east}  N={north}")
        gs.message(sep)
        gs.message(f"  {'Method':<12}  {'Score':>6}  {'Label'}")
        gs.message(sep)
        for mth in methods:
            if mth == '_weights':
                continue
            score = scores.get(mth, float('nan'))
            marker = ' ◀' if mth == 'consensus' else ''
            gs.message(f"  {mth:<12}  {score:6.4f}  "
                       f"{METHOD_LABELS.get(mth,'')}{marker}")
        # Print diversity weights if consensus was requested
        pt_weights = scores.get('_weights')
        if pt_weights:
            gs.message(sep)
            gs.message("  Diversity weights used in consensus:")
            for nm, wt in sorted(pt_weights.items(), key=lambda x: -x[1]):
                gs.message(f"    {nm:<12}  w={wt:.3f}")
        gs.message(sep)

        if flag_v:
            gs.message("Pixel spectrum (resampled to sensor bands):")
            for b, v in zip(bands, spec_pixel):
                gs.verbose(f"  {b['wavelength']:8.2f} nm : {v:.6f}")
            gs.message("Reference spectrum (resampled):")
            for wl, v in zip(wls, ref_proc):
                gs.verbose(f"  {wl:8.2f} nm : {v:.6f}")

        return 0

    # ------------------------------------------------------------------
    # Step 6: Load full cube
    # ------------------------------------------------------------------
    gs.message("Loading hyperspectral cube into memory...")
    cube = load_cube(bands, raster3d, verbose=flag_v)

    # Apply preprocessing to cube
    if flag_z:
        cube = to_prob_simplex_cube(cube)
    if normalize_m != 'none':
        cube = normalize_cube(cube, normalize_m)
    if flag_c and 'cr_sam' not in methods and 'cr_ed' not in methods:
        gs.message("Applying continuum removal to all pixels...")
        cube = continuum_remove_cube(cube, wls)

    cube = cube.astype(np.float64)

    # ------------------------------------------------------------------
    # Step 7a: Consensus path (runs all base methods, fuses internally)
    # ------------------------------------------------------------------
    consensus_result: dict = {}

    if do_consensus:
        gs.message(f"Running full consensus analysis "
                   f"(fusion_mode={fusion_mode}, "
                   f"agreement_threshold={agr_thresh})...")

        # Any explicitly requested non-consensus, non-ensemble methods
        # are pre-computed and passed in so they are not run twice.
        seed_maps: dict[str, np.ndarray] = {}
        seed_methods = [m for m in methods
                        if m not in ('consensus', 'ensemble')]
        for idx, mth in enumerate(seed_methods):
            gs.message(f"  (seed) [{idx+1}/{len(seed_methods)}] "
                       f"{METHOD_LABELS.get(mth, mth)}")
            seed_maps[mth] = compute_method(mth, cube, ref_proc, wls,
                                            shift_win, seed_maps)

        consensus_result = run_consensus_analysis(
            cube, ref_proc, wls, shift_win,
            existing_score_maps=seed_maps,
            fusion_mode=fusion_mode,
            agreement_threshold=agr_thresh,
            skip_slow=False,
            verbose=flag_v,
        )

        primary_method = 'consensus'
        primary_map    = consensus_result['probability']
        score_maps     = consensus_result['score_maps']

    # ------------------------------------------------------------------
    # Step 7b: Normal path (explicit method list, no consensus)
    # ------------------------------------------------------------------
    else:
        gs.message(f"Computing {len(methods)} method(s): {', '.join(methods)}")
        score_maps: dict[str, np.ndarray] = {}

        non_ensemble = [m for m in methods if m != 'ensemble']
        do_ensemble  = 'ensemble' in methods

        for idx, mth in enumerate(non_ensemble):
            gs.message(f"  [{idx+1}/{len(non_ensemble)}] "
                       f"{METHOD_LABELS.get(mth, mth)} ({mth})")
            gs.percent(idx, len(non_ensemble), 2)
            score_maps[mth] = compute_method(mth, cube, ref_proc, wls,
                                             shift_win, score_maps)
            if flag_v:
                s = score_maps[mth]
                valid = s[np.isfinite(s)]
                if valid.size > 0:
                    gs.verbose(f"    range [{valid.min():.4f}, {valid.max():.4f}] "
                               f"mean={valid.mean():.4f}")

        if do_ensemble:
            gs.message(f"  Ensemble fusion from {len(score_maps)} methods...")
            score_maps['ensemble'] = match_ensemble(score_maps)

        gs.percent(len(non_ensemble), len(non_ensemble), 2)

        primary_method = 'ensemble' if do_ensemble else non_ensemble[0]
        primary_map    = score_maps[primary_method]

    # ------------------------------------------------------------------
    # Step 8: Write primary output
    # ------------------------------------------------------------------
    gs.message(f"Writing primary output: {output} (method: {primary_method})")
    write_raster(primary_map, output)
    set_similarity_colors(output)
    set_raster_metadata(output, raster3d, primary_method, ref_desc)

    # ------------------------------------------------------------------
    # Step 9: Per-method prefix outputs
    # ------------------------------------------------------------------
    if out_prefix:
        gs.message(f"Writing per-method maps with prefix: {out_prefix}")

        # Base score maps
        for mth, smap in score_maps.items():
            map_name = f"{out_prefix}_{mth}"
            if map_name == output:
                continue
            gs.message(f"  {map_name}")
            write_raster(smap, map_name)
            set_similarity_colors(map_name)
            set_raster_metadata(map_name, raster3d, mth, ref_desc)

        # Consensus diagnostic maps (agreement, entropy, conflict, spread)
        if do_consensus:
            diag_labels = {
                'agreement': 'Fraction of methods voting hotspot',
                'entropy':   'Method agreement entropy (0=agree, 1=conflict)',
                'conflict':  'Ambiguous hotspot mask (high prob + high entropy)',
                'spread':    'Std dev of calibrated scores per pixel',
            }
            for key, label in diag_labels.items():
                arr = consensus_result.get(key)
                if arr is None:
                    continue
                map_name = f"{out_prefix}_consensus_{key}"
                gs.message(f"  {map_name}  [{label}]")
                write_raster(arr.astype(np.float64), map_name)
                # Tailored color tables
                if key == 'agreement':
                    set_similarity_colors(map_name)
                elif key == 'entropy':
                    # Reversed: 0=green (good agreement), 1=red (conflict)
                    rules = "0.0 0:200:0\n0.5 255:220:0\n1.0 200:0:0\n"
                    gs.write_command('r.colors', map=map_name,
                                     rules='-', stdin=rules, quiet=True)
                elif key == 'conflict':
                    gs.run_command('r.colors', map=map_name,
                                   color='grey', quiet=True)
                elif key == 'spread':
                    gs.run_command('r.colors', map=map_name,
                                   color='plasma', quiet=True)
                try:
                    gs.run_command('r.support', map=map_name,
                                   title=f"Consensus {key}",
                                   description=label, quiet=True)
                except Exception:
                    pass

            # Per-method calibrated probability maps
            calibrated = consensus_result.get('calibrated', {})
            for mth, cmap in calibrated.items():
                map_name = f"{out_prefix}_cal_{mth}"
                gs.message(f"  {map_name}  [calibrated p for {mth}]")
                write_raster(cmap.astype(np.float64), map_name)
                set_similarity_colors(map_name)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    s = primary_map[np.isfinite(primary_map)]
    sep = "=" * 60
    gs.message(" ")
    gs.message(sep)
    gs.message("i.hyper.sleuth completed successfully.")
    gs.message(f"  Primary method : {primary_method}")
    gs.message(f"  Output map     : {output}")
    if s.size > 0:
        gs.message(f"  Score range    : [{s.min():.4f}, {s.max():.4f}]")
        gs.message(f"  Score mean     : {s.mean():.4f}")
        for thr in (0.9, 0.8, 0.7, 0.5):
            n_above = int((s >= thr).sum())
            pct = 100.0 * n_above / s.size
            if n_above > 0:
                gs.message(f"  Pixels ≥ {thr:.1f}   : {n_above} ({pct:.2f}%)")
                break

    if do_consensus:
        w = consensus_result['weights']
        gs.message("  Diversity weights:")
        for nm, wt in sorted(w.items(), key=lambda x: -x[1])[:6]:
            gs.message(f"    {nm:<12s} {wt:.3f}")
        if len(w) > 6:
            gs.message(f"    ... ({len(w)} methods total)")
    if out_prefix:
        gs.message(f"  Per-method maps: {out_prefix}_{{method}}")
    gs.message(sep)

    return 0


if __name__ == "__main__":
    options, flags = gs.parser()
    sys.exit(main(options, flags))
