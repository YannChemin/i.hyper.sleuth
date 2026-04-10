# i.hyper.sleuth

**GRASS GIS module — Spectral target detection and multi-method consensus
hotspot mapping from hyperspectral imagery**

Part of the [i.hyper](../README.md) module family for VNIR-SWIR hyperspectral
data processing in GRASS GIS.

---

## Overview

`i.hyper.sleuth` finds the pixels in a hyperspectral 3D raster that most
closely match a user-supplied reference spectrum.  For each pixel a
similarity score in **[0, 1]** (0 = no match, 1 = perfect match) is computed
using one or more of **19 similarity methods** drawn from remote sensing,
signal analysis, information theory, morphological mathematics, and
subpixel-detection theory.

The module also exposes a full **multi-method consensus pipeline**
(`method=consensus`) that runs all 17 base methods simultaneously, calibrates
their scores to true per-pixel probabilities via the empirical CDF rank
transform, down-weights correlated methods using an inter-method diversity
analysis, and fuses the result into a single hotspot probability map — plus
four diagnostic maps (agreement, entropy, conflict, spread).

Input 3D rasters are produced by
[`i.hyper.import`](../i.hyper.import) or
[`i.hyper.atcorr`](../i.hyper.atcorr).
Reference spectra can be supplied inline, as CSV, or as JSON.

---

## Similarity methods

| Key | Name | Category |
|-----|------|----------|
| `sam` | Spectral Angle Mapper | Geometric |
| `sid` | Spectral Information Divergence | Information theory |
| `sid_sam` | SID × tan(SAM) hybrid | Combined |
| `ed` | Euclidean Distance (L2) | Distance |
| `sad` | Spectral Absolute Difference (L1) | Distance |
| `sca` | Spectral Correlation Angle (Pearson *r*) | Statistical |
| `cr_sam` | Continuum-Removed SAM | Morphological |
| `cr_ed` | Continuum-Removed Euclidean Distance | Morphological |
| `gd1` | 1st-Derivative Shape Matching | Signal analysis |
| `gd2` | 2nd-Derivative Shape Matching | Signal analysis |
| `xcorr` | Normalized Cross-Correlation | Signal analysis |
| `dtw` | Dynamic Time Warping | Signal analysis |
| `ssim` | Spectral Structural Similarity Index | Signal analysis |
| `jsd` | Jensen-Shannon Divergence | Information theory |
| `bhatt` | Bhattacharyya Coefficient | Statistical |
| `mtf` | Matched Tuned Filter | Subpixel detection |
| `cem` | Constrained Energy Minimization | Subpixel detection |
| `ensemble` | Rank-based Borda-count fusion | Meta |
| `consensus` | Multi-method calibrated probability fusion | Meta |

---

## Consensus analysis (`method=consensus`)

When `consensus` is requested the module executes a four-step pipeline:

1. **Compute all base methods** — all 17 methods except `ensemble`/`consensus`
   are run against the cube (already-computed maps are reused).

2. **Empirical CDF calibration** — each score map is rank-transformed to a
   uniform probability in (0, 1].  This removes the scale bias that would
   otherwise allow high-range methods (SAM in [0.8–1.0]) to dominate
   over low-range methods (ED in [0.0–0.2]).

3. **Diversity weighting** — the full *k × k* Pearson correlation matrix is
   built across calibrated maps.  Each method receives weight
   `w ∝ 1 / mean |r_ij|`, so that correlated clusters (e.g. SAM + SCA)
   do not over-count their shared evidence.

4. **Fusion** — six modes are available (controlled by `fusion_mode=`):

| Mode | Description |
|------|-------------|
| `rank_product` | Weighted geometric mean of rank fractions *(default)* |
| `fisher` | Fisher χ² combined probability test — proper statistical p-value |
| `stouffer` | Stouffer weighted Z-score — diversity weights enter exactly |
| `group_product` | AND within method-type groups, OR across groups |
| `harmonic` | Harmonic mean — strictest, all methods must agree |
| `min` | Minimum across methods — absolute strictest |

### Consensus output maps

When `output_prefix=` is set, the following additional maps are written:

| Map | Content |
|-----|---------|
| `{prefix}_consensus_agreement` | Fraction of methods voting above threshold [0, 1] |
| `{prefix}_consensus_entropy` | Agreement entropy: 0 = unanimous, 1 = maximal conflict |
| `{prefix}_consensus_conflict` | High-probability pixels where methods disagree (review these) |
| `{prefix}_consensus_spread` | Std dev of calibrated scores per pixel |
| `{prefix}_cal_{method}` | Rank-calibrated probability map for each base method |
| `{prefix}_{method}` | Raw similarity score for each base method |

---

## Wavelength LUT

A `WavelengthLUT` is built once from the reference and sensor wavelength
grids.  It precomputes `searchsorted` indices and linear blend weights so
that all subsequent resampling is O(*n*) with no repeated binary search.
It also reports:

- which sensor bands fall outside the reference range (edge-fill bias risk)
- which reference points fall outside the sensor range (unobservable features)
- the exact overlap interval for use with `min_wavelength=` / `max_wavelength=`

---

## Reference spectrum formats

### Inline (`reference=`)

Comma-separated `wavelength:reflectance` pairs passed directly on the
command line.  Colons and semicolons are both accepted as pair separators;
pairs may also be whitespace-delimited.  Scientific notation is supported.
The list is sorted by wavelength before use.

```bash
# Colon separator, comma list (most common)
reference="450:0.04,550:0.11,670:0.05,750:0.40,900:0.45,1650:0.22,2200:0.18"

# Semicolon separator
reference="450;0.04,550;0.11,670;0.05"

# Whitespace list
reference="450:0.04 550:0.11 670:0.05"

# Scientific notation
reference="4.5e2:4.0e-2,5.5e2:1.1e-1,8.0e2:4.0e-1"
```

At least 2 pairs are required.  Wavelength units must match the sensor
metadata (typically nanometres).  Reflectance values should be in [0, 1]
(surface reflectance); values > 2 trigger a warning that the data may be in
percent reflectance.

### CSV file (`reference_file=`)

Two columns: `wavelength` and `reflectance`.  Any row whose first field is
non-numeric (header line, comment, blank line) is silently skipped.
Pairs are sorted by wavelength before use.  **Minimum 2 data rows required.**

```csv
wavelength,reflectance
450,0.04
550,0.11
670,0.05
750,0.40
900,0.45
1650,0.22
2200,0.18
```

Header-less CSV is equally valid:

```csv
450,0.04
550,0.11
670,0.05
```

Lines beginning with `#` or other non-numeric text are treated as comments:

```csv
# kaolinite — USGS Spectral Library 7
# wavelength (nm), reflectance [0-1]
wavelength,reflectance
2165,0.22
2195,0.18
2205,0.09
2240,0.21
2320,0.31
```

### JSON file (`reference_file=`)

Four distinct JSON layouts are accepted:

**1. Array of pairs** (most compact):

```json
[[450, 0.04], [550, 0.11], [670, 0.05], [800, 0.42], [2200, 0.18]]
```

**2. Parallel arrays** (ENVI / spectral library style):

```json
{
    "wavelengths":  [450, 550, 670, 800, 2200],
    "reflectances": [0.04, 0.11, 0.05, 0.42, 0.18]
}
```

Key aliases accepted: `"wavelength"` or `"wl"` for the wavelength array;
`"reflectance"` or `"r"` for the reflectance array.

**3. Named list** (`"data"`, `"spectrum"`, or `"pairs"` key):

```json
{"data": [[450, 0.04], [550, 0.11], [670, 0.05]]}
```

```json
{"spectrum": [[450, 0.04], [550, 0.11], [670, 0.05]]}
```

**4. Nested object with mixed keys** (custom export formats):

```json
{
    "wl": [450, 550, 670, 800],
    "r":  [0.04, 0.11, 0.05, 0.42]
}
```

All JSON variants are sorted by wavelength before use.

### Wavelength units

The reference wavelengths must be in the same units as the sensor band
metadata (usually **nanometres**).  The following unit strings in the band
metadata are converted automatically to nm: `um` / `µm` / `micrometer`,
`m` / `meter`.  If your reference file is in µm (e.g., USGS `.asc` exports),
convert before passing:

```bash
# Convert USGS µm → nm in a quick Python one-liner
python3 -c "
import csv, sys
for row in csv.reader(open('kaolinite.csv')):
    try: print(f'{float(row[0])*1000:.2f},{row[1]}')
    except: print(','.join(row))
" > kaolinite_nm.csv
```

| Format | Description |
|--------|-------------|
| `reference=` | Inline `wl:r,wl:r,...` pairs on the command line |
| CSV file | Two columns `wavelength,reflectance`; header row skipped |
| JSON file | `[[wl,r],...]` or `{"wavelengths":[...],"reflectances":[...]}` |

---

## Output

| Output | GRASS type | Range | Content |
|--------|-----------|-------|---------|
| `output=` | float FCELL | 0 – 1 | Similarity / hotspot probability |
| `{prefix}_{method}` | float FCELL | 0 – 1 | Per-method similarity map |
| `{prefix}_consensus_*` | float FCELL | 0 – 1 | Consensus diagnostic maps |
| `{prefix}_cal_{method}` | float FCELL | 0 – 1 | Calibrated probability per method |

All output maps share a blue → yellow → red colour ramp (0 = blue, 1 = red).

---

## Quick examples

```bash
# SAM match against a kaolinite CSV library entry
i.hyper.sleuth input=scene_atcorr output=kaolinite_sam \
  reference_file=kaolinite_usgs.csv method=sam

# Full consensus analysis with all methods + diagnostic maps
i.hyper.sleuth input=scene_atcorr output=hotspot \
  reference_file=target.csv \
  method=consensus fusion_mode=group_product \
  output_prefix=tgt

# Point inspection: all methods at one pixel
i.hyper.sleuth input=scene_atcorr output=_ \
  reference="450:0.04,670:0.05,800:0.42,2200:0.18" \
  method=sam,sid,bhatt,dtw,mtf,consensus \
  coordinates="452300,4325100" -p -v

# Six methods + ensemble, per-method maps
i.hyper.sleuth input=scene_atcorr output=best \
  reference_file=chlorophyll_a.json \
  method=sam,cr_sam,gd1,jsd,bhatt,ensemble \
  output_prefix=chl normalize=minmax -c
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input=` | — | Input hyperspectral 3D raster (from `i.hyper.import` / `i.hyper.atcorr`) |
| `output=` | — | Output similarity raster map (FCELL, [0, 1]) |
| `reference=` | — | Inline `wl:r,...` spectrum (mutually exclusive with `reference_file=`) |
| `reference_file=` | — | CSV or JSON spectrum file (mutually exclusive with `reference=`) |
| `method=` | `sam` | Comma-separated list of methods (see table above) |
| `fusion_mode=` | `rank_product` | Fusion strategy for `method=consensus` |
| `agreement_threshold=` | `0.80` | Calibrated-probability threshold for per-pixel agreement count (consensus only) |
| `output_prefix=` | — | Prefix for per-method output maps; enables map caching in consensus |
| `resample=` | `linear` | Interpolation for resampling reference to sensor grid: `linear`, `cubic`, `pchip` |
| `normalize=` | `none` | Spectrum normalisation before matching: `none`, `area`, `max`, `minmax`, `vector` |
| `shift_window=` | `3` | Maximum band-shift for shift-tolerant methods (`xcorr`, `dtw`); 0 disables |
| `min_wavelength=` | — | Lower wavelength limit (nm); restricts matching to overlap region |
| `max_wavelength=` | — | Upper wavelength limit (nm); restricts matching to overlap region |
| `coordinates=` | — | `east,north` for point-mode analysis (requires `-p`) |

## Flags

| Flag | Effect |
|------|--------|
| `-n` | Only use bands marked `valid=1` in metadata |
| `-i` | Info mode: print band coverage and LUT summary, then exit |
| `-v` | Verbose: show per-method scores and diversity weights |
| `-c` | Apply convex-hull continuum removal before matching |
| `-p` | Point mode: print score table for one pixel at `coordinates=` |
| `-z` | Normalize spectra to probability simplex (sum-to-one) |

---

## Performance notes

| Method class | Speed | Note |
|---|---|---|
| `sam`, `sid`, `ed`, `sad`, `sca`, `jsd`, `bhatt`, `ssim`, `xcorr`, `gd1`, `gd2` | Fast | Fully vectorized over all pixels |
| `mtf`, `cem` | Fast | One k×k covariance inversion, then linear |
| `dtw` | Moderate | Chunked Sakoe-Chiba rolling-window; controlled by `shift_window=` |
| `cr_sam`, `cr_ed` | Slow | Per-pixel Graham-scan convex hull |
| `consensus` | Slow (first run) | Runs all 17 base methods; subsequent runs with `output_prefix=` reuse cached maps |

---

## References

- Kruse *et al.* (1993) — SAM. *Remote Sens. Environ.* 44, 145–163.
- Chang C.I. (2000) — SID and SID-SAM. *IEEE Trans. Inf. Theory* 46(5).
- Clark *et al.* (1987) — Continuum removal. *J. Geophys. Res.* 92(B12).
- Reed & Yu (1990) — Matched Tuned Filter. *IEEE Trans. ASSP* 38(10).
- Chang & Heinz (2000) — CEM. *IEEE Trans. GRSS* 38(3).
- Wang *et al.* (2004) — SSIM. *IEEE Trans. Image Process.* 13(4).
- Sakoe & Chiba (1978) — DTW. *IEEE Trans. ASSP* 26(1).
- Fisher R.A. (1932) — Combined probability test. Oliver & Boyd.
- Stouffer S.A. *et al.* (1949) — Measurement and Prediction. Princeton UP.

---

## See also

[`i.hyper.import`](../i.hyper.import) ·
[`i.hyper.atcorr`](../i.hyper.atcorr) ·
[`i.hyper.continuum`](../i.hyper.continuum) ·
[`i.hyper.spectroscopy`](../i.hyper.spectroscopy) ·
[`i.hyper.geology`](../i.hyper.geology)
