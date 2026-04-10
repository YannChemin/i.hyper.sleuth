#!/usr/bin/env python3
"""
Generate synthetic hyperspectral test data for i.hyper.sleuth.

Author:    Yann Chemin <yann.chemin@gmail.com>
Copyright: (C) 2026 by Yann Chemin and the GRASS Development Team
License:   GPL-2.0-or-later

Creates a synthetic 20×20 pixel hyperspectral 3D raster with wavelength
metadata.  Each scene is a uniform reflectance field with one 3×3 pixel
target patch that has a distinctive spectral signature.

Scenes
------
kaolinite
    Featureless background (flat 0.20) with a 3×3 kaolinite-like target:
    Al-OH double absorption at 2165/2205 nm, high SWIR reflectance otherwise.

chlorophyll
    Soil-like background (0.15 flat) with a 3×3 green-vegetation target:
    red-edge at 700 nm, strong NIR plateau, red well (670 nm absorption).

Usage (inside a GRASS session)
-------------------------------
    python3 generate_test_data.py [--scene all|kaolinite|chlorophyll] [--cleanup]
"""

import argparse
import os
import sys
import tempfile
import json

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required.")
    sys.exit(1)

try:
    import grass.script as gs
except ImportError:
    print("Error: must be run inside a GRASS GIS session.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Band definitions — 30 bands spanning 400–2500 nm
# ---------------------------------------------------------------------------

BAND_WAVELENGTHS = [
    400, 450, 500, 550, 600, 650, 670, 700, 730, 800,
    850, 900, 1000, 1100, 1300, 1400, 1500, 1600, 1700, 1750,
    1800, 1900, 2000, 2100, 2165, 2200, 2205, 2250, 2350, 2500,
]
N_BANDS = len(BAND_WAVELENGTHS)
FWHM_NM = 10.0
ROWS, COLS = 20, 20

# 3×3 target patch centred at (10, 10): rows 9-11, cols 9-11
TARGET_ROWS = slice(9, 12)
TARGET_COLS = slice(9, 12)


# ---------------------------------------------------------------------------
# Spectral end-member functions
# ---------------------------------------------------------------------------

def _kaolinite_bg(wl):
    """Featureless background reflectance (flat ~0.20)."""
    return 0.20


def _kaolinite_target(wl):
    """Kaolinite-like spectrum: Al-OH doublet at 2165/2205 nm."""
    r = 0.35
    if 650 < wl < 700:
        r = 0.25
    if 2100 < wl < 2180:
        r = 0.35 - 0.18 * max(0, 1.0 - abs(wl - 2165) / 40)
    if 2185 < wl < 2250:
        r = 0.35 - 0.22 * max(0, 1.0 - abs(wl - 2205) / 30)
    return max(0.0, r)


def _chlorophyll_bg(wl):
    """Dry-soil-like background (flat ~0.15)."""
    return 0.15


def _chlorophyll_target(wl):
    """Green-vegetation spectrum: red-edge at 700 nm, NIR plateau."""
    if wl < 500:
        return 0.04
    if wl < 670:
        r = 0.04 + 0.04 * (wl - 500) / 170
        if 620 < wl < 680:
            r -= 0.04 * max(0, 1 - abs(wl - 670) / 30)   # red absorption
        return max(0, r)
    if wl < 700:
        return 0.04 + 0.36 * (wl - 670) / 30   # steep red edge
    if wl < 1300:
        return 0.40 + 0.05 * ((wl - 700) / 600)
    if wl < 1900:
        return 0.45 - 0.10 * ((wl - 1300) / 600)
    return 0.20


SCENE_FUNCTIONS = {
    "kaolinite":   (_kaolinite_bg,   _kaolinite_target),
    "chlorophyll": (_chlorophyll_bg, _chlorophyll_target),
}

# Expected similarity properties for each scene
SCENE_EXPECTATIONS = {
    "kaolinite": {
        "target_rows": (9, 12),
        "target_cols": (9, 12),
        "description": "kaolinite-like Al-OH doublet target on flat background",
    },
    "chlorophyll": {
        "target_rows": (9, 12),
        "target_cols": (9, 12),
        "description": "green-vegetation target on dry-soil background",
    },
}

# Reference spectrum as CSV content for each scene
REFERENCE_CSV = {
    "kaolinite": "\n".join(
        ["wavelength,reflectance"]
        + [f"{wl},{_kaolinite_target(wl):.6f}" for wl in BAND_WAVELENGTHS]
    ),
    "chlorophyll": "\n".join(
        ["wavelength,reflectance"]
        + [f"{wl},{_chlorophyll_target(wl):.6f}" for wl in BAND_WAVELENGTHS]
    ),
}

# Reference spectrum as JSON (parallel-arrays layout)
REFERENCE_JSON = {
    "kaolinite": json.dumps({
        "wavelengths":  BAND_WAVELENGTHS,
        "reflectances": [_kaolinite_target(w) for w in BAND_WAVELENGTHS],
    }),
    "chlorophyll": json.dumps({
        "wavelengths":  BAND_WAVELENGTHS,
        "reflectances": [_chlorophyll_target(w) for w in BAND_WAVELENGTHS],
    }),
}


# ---------------------------------------------------------------------------
# Scene creation helpers
# ---------------------------------------------------------------------------

def setup_test_region(rows=ROWS, cols=COLS):
    """Set GRASS computational region to rows×cols at 1 m resolution."""
    gs.run_command(
        "g.region", n=rows, s=0, e=cols, w=0,
        rows=rows, cols=cols, res=1, quiet=True
    )


def inject_band_metadata(raster3d_name, band_wavelengths=None, fwhm=FWHM_NM):
    """Write wavelength metadata into each 2D band-slice raster.

    i.hyper.sleuth reads wavelength via ``r.info -h`` on ``{map}#{band}``.

    :param str raster3d_name: base name of the 3D raster (bands must exist as
                               ``{raster3d_name}#{1..N}``)
    :param list band_wavelengths: list of wavelength values in nm
    :param float fwhm: FWHM in nm applied to all bands
    """
    if band_wavelengths is None:
        band_wavelengths = BAND_WAVELENGTHS
    for i, wl in enumerate(band_wavelengths, start=1):
        band_map = f"{raster3d_name}#{i}"
        gs.run_command(
            "r.support", map=band_map,
            history=f"wavelength={wl:.2f}\nFWHM={fwhm:.2f}\nvalid=1\nunit=nm",
            quiet=True,
        )


def create_scene(name, rng=None):
    """Create a synthetic hyperspectral scene as band-slice 2D rasters + 3D raster.

    Band-slice rasters are named ``{name}#1`` … ``{name}#{N_BANDS}``.

    :param str name: scene key (one of :data:`SCENE_FUNCTIONS`)
    :param rng: optional numpy random generator for reproducible noise
    :type rng: np.random.Generator or None
    :raises KeyError: if *name* is not in :data:`SCENE_FUNCTIONS`
    """
    if name not in SCENE_FUNCTIONS:
        raise KeyError(f"Unknown scene '{name}'. Available: {list(SCENE_FUNCTIONS)}")
    if rng is None:
        rng = np.random.default_rng(42)

    bg_fn, tgt_fn = SCENE_FUNCTIONS[name]

    import grass.script.array as garray

    band_maps = []
    for i, wl in enumerate(BAND_WAVELENGTHS, start=1):
        band_name = f"{name}#{i}"
        data = np.full((ROWS, COLS), bg_fn(wl), dtype=np.float32)
        data[TARGET_ROWS, TARGET_COLS] = tgt_fn(wl)
        # Add ±1% Gaussian noise
        data += rng.normal(0, 0.005, data.shape).astype(np.float32)
        data = np.clip(data, 0.0, 1.0)

        arr = garray.array()
        arr[:] = data
        arr.write(band_name, overwrite=True)
        band_maps.append(band_name)

    inject_band_metadata(name)

    # Build 3D raster from band slices so raster3d_info() works
    expr_parts = [f"depth{i}={bm}" for i, bm in enumerate(band_maps, start=1)]
    try:
        for i, bm in enumerate(band_maps, start=1):
            # r3.mapcalc accepts depth slices
            gs.run_command(
                "r3.mapcalc",
                expression=f"{name} = depth() == {i} ? {bm} : 0",
                overwrite=True,
                quiet=True,
            )
    except Exception:
        # Fallback: write 3D raster by concatenating slices via r3.from.2d
        gs.run_command(
            "r3.from.2d",
            input=",".join(band_maps),
            output=name,
            overwrite=True,
            quiet=True,
        )

    gs.message(f"Created scene '{name}': {N_BANDS} bands × {ROWS}×{COLS} pixels")
    return band_maps


def cleanup_scene(name):
    """Remove all rasters associated with scene *name*.

    :param str name: scene name (removes the 3D raster and all ``{name}#N`` slices)
    """
    slice_names = [f"{name}#{i}" for i in range(1, N_BANDS + 1)]
    existing = [n for n in slice_names
                if gs.find_file(n, element="cell").get("name")]
    if existing:
        gs.run_command("g.remove", type="raster",
                       name=",".join(existing), flags="f", quiet=True)
    if gs.find_file(name, element="raster_3d").get("name"):
        gs.run_command("g.remove", type="raster_3d",
                       name=name, flags="f", quiet=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic i.hyper.sleuth test scenes."
    )
    parser.add_argument(
        "--scene", default="all",
        help="Scene to create: all, kaolinite, chlorophyll (default: all)"
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Remove previously created test scenes instead of creating"
    )
    args = parser.parse_args()

    scenes = list(SCENE_FUNCTIONS) if args.scene == "all" else [args.scene]
    setup_test_region()

    for sc in scenes:
        if args.cleanup:
            cleanup_scene(sc)
            gs.message(f"Cleaned up scene '{sc}'")
        else:
            create_scene(sc)


if __name__ == "__main__":
    main()
