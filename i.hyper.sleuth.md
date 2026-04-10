## DESCRIPTION

*i.hyper.sleuth* finds the pixels in a hyperspectral 3D raster that most
closely match a user-supplied reference spectrum.  For each pixel the
module computes a similarity score in **[0, 1]** (0 = no match, 1 = perfect
match) using one or more of **18 spectral-similarity methods** drawn from
remote sensing, signal analysis, information theory, morphological
mathematics, and subpixel-detection theory.

The reference spectrum is automatically resampled onto the sensor wavelength
grid via a precomputed **WavelengthLUT** (linear interpolation; O(n) per
pixel, no repeated binary search).  Optional preprocessing — continuum
removal, normalization, probability-simplex projection — is applied
identically to the reference and to every pixel before matching.

The module also exposes a full **multi-method consensus pipeline**
(`method=consensus`) that runs all 16 base methods simultaneously,
calibrates each score map to a true per-pixel probability via an empirical
CDF rank transform, down-weights correlated methods using an inter-method
diversity analysis, and fuses the result into a single hotspot probability
map plus four diagnostic maps (agreement, entropy, conflict, spread).

*i.hyper.sleuth* is part of the **i.hyper** module family for VNIR-SWIR
hyperspectral data import, processing, and analysis in GRASS GIS.

### Reference spectrum formats

#### Inline (`reference=`)

Comma-separated `wavelength:reflectance` pairs on the command line:

::: code

    reference="450:0.04,550:0.11,670:0.05,750:0.40,900:0.45,1650:0.22,2200:0.18"

:::

Semicolons are also accepted as the pair separator.

#### CSV file (`reference_file=`)

Two columns `wavelength,reflectance`, one pair per line.  An optional
header row (non-numeric first field) is skipped automatically.  Wavelength
units must match the sensor metadata (nanometres by default).

#### JSON file (`reference_file=`)

Any of the following layouts are accepted:

- `[[wl, r], [wl, r], ...]`
- `{"wavelengths": [...], "reflectances": [...]}`
- `{"data": [[wl, r], ...]}`

### Similarity methods

| Key | Name | Category | Core formula |
|-----|------|----------|-------------|
| `sam` | Spectral Angle Mapper | Geometric | θ = arccos(r·p / \|r\|\|p\|); sim = 1 − 2θ/π |
| `sid` | Spectral Information Divergence | Information theory | SID = D(r‖p) + D(p‖r); sim = exp(−SID) |
| `sid_sam` | SID × tan(SAM) | Combined | sim = exp(−SID·tan θ) |
| `ed` | Euclidean Distance | Distance | sim = 1 / (1 + \|r−p\|₂ / √n) |
| `sad` | Spectral Absolute Difference | Distance | sim = 1 / (1 + mean\|rᵢ−pᵢ\|) |
| `sca` | Spectral Correlation Angle | Statistical | Pearson r mapped to [0, 1] |
| `cr_sam` | Continuum-Removed SAM | Morphological | upper-hull CR then SAM |
| `cr_ed` | Continuum-Removed ED | Morphological | upper-hull CR then L2 distance |
| `gd1` | 1st-Derivative Shape Matching | Signal analysis | finite-diff derivative, then SAM |
| `gd2` | 2nd-Derivative Shape Matching | Signal analysis | 2nd finite-diff, then SAM |
| `xcorr` | Normalized Cross-Correlation | Signal analysis | max NCC over lags ±W; sim = (max+1)/2 |
| `dtw` | Dynamic Time Warping | Signal analysis | Sakoe-Chiba DTW; sim = exp(−DTW/n) |
| `ssim` | Spectral SSIM | Signal analysis | luminance × contrast × structure |
| `jsd` | Jensen-Shannon Divergence | Information theory | JSD = (KLD(r‖m)+KLD(p‖m))/2; sim = exp(−JSD) |
| `bhatt` | Bhattacharyya Coefficient | Statistical | BC = Σ√(rᵢpᵢ) ∈ [0, 1] directly |
| `mtf` | Matched Tuned Filter | Subpixel | w = Σ⁻¹(d−μ); score percentile-normalised |
| `cem` | Constrained Energy Minimization | Subpixel | w = Σ⁻¹d/(dᵀΣ⁻¹d); score percentile-normalised |
| `ensemble` | Borda-count rank fusion | Meta | sum of per-method pixel ranks |
| `consensus` | Calibrated probability fusion | Meta | CDF calibration + diversity weights + fusion |

Multiple methods can be specified as a comma-separated list
(`method=sam,sid,bhatt,dtw`).  When `output_prefix=` is set, each method
produces its own similarity map `{prefix}_{method}`.

### Method details

#### SAM — Spectral Angle Mapper

The angle between the reference vector **r** and the pixel vector **p** in
*n*-dimensional reflectance space:

    θ = arccos( r · p / (|r| · |p|) )     θ ∈ [0, π/2]
    similarity = 1 − 2θ/π

Scale-invariant: insensitive to illumination magnitude.  Recommended as a
first-pass method; threshold 0.97 (≈5°) for tight match, 0.90 (≈26°)
for loose.

#### SID — Spectral Information Divergence

Each spectrum is treated as a discrete probability distribution (values are
normalised to sum to one before computing).  The symmetric KL divergence is:

    SID = D(p‖q) + D(q‖p) = Σᵢ pᵢ ln(pᵢ/qᵢ) + qᵢ ln(qᵢ/pᵢ)
    similarity = exp(−SID)

Sensitive to subtle redistribution of spectral energy between bands.
Recommended with the `-z` flag.

#### SID-SAM hybrid

    similarity = exp(−SID · tan θ_SAM)

Combines probabilistic and angular discrimination.  Particularly effective
when both spectral shape and energy distribution differ between target and
background.

#### ED / SAD — Distance methods

    ED  similarity = 1 / (1 + ‖r−p‖₂ / √n)
    SAD similarity = 1 / (1 + mean|rᵢ−pᵢ|)

Simple and fast.  Sensitive to overall magnitude offset; use
`normalize=vector` or `normalize=minmax` to remove illumination effects.
SAD (L1) is more robust to individual outlier bands than ED (L2).

#### SCA — Spectral Correlation Angle

Pearson correlation coefficient between the two spectra:

    SCA = (Σ (r̄ᵢ·p̄ᵢ)) / (‖r̄‖ · ‖p̄‖)    where ̄ denotes mean-centring

Mapped to [0, 1] as `(SCA + 1) / 2`.  Equivalent to SAM applied to
mean-centered spectra.  Captures spectral shape independently of mean level.

#### CR-SAM / CR-ED — Continuum-Removed methods

The upper convex hull of the spectrum (wavelength as x-axis, reflectance as
y-axis) is computed using a Graham-scan variant.  The continuum-removed
spectrum at each band is `CR(λ) = ρ(λ) / hull(λ)`.  SAM or L2 distance is
then applied to the CR spectra.

Isolates absorption-feature shape, removing broadband albedo and slope
contributions.  Best for mineral identification where absorption position and
depth matter more than overall brightness.

**Note:** this computation cannot be vectorized over pixels and involves a
per-pixel loop.  Expect roughly 10× slower throughput than pure-linear
methods.

#### GD1 / GD2 — Derivative Shape Matching

First and second finite differences of the resampled spectra:

    gd1ᵢ = ρᵢ₊₁ − ρᵢ
    gd2ᵢ = ρᵢ₊₁ − 2ρᵢ + ρᵢ₋₁

SAM is then applied to the derivative vector.  GD1 is highly sensitive to
slope transitions and absorption-edge positions; insensitive to overall level
and broadband curvature — analogous to high-pass filtering before matching.
GD2 responds only to local curvature (absorption-band centres and inflection
points).  GD2 requires well-calibrated data with low noise.

#### XCORR — Normalized Cross-Correlation

NCC is computed at integer lags k ∈ [−W, +W] (W = `shift_window=`):

    NCC(k) = (r̄ · p̄_shift(k)) / (‖r̄‖ · ‖p̄_shift(k)‖)
    similarity = (max_k NCC(k) + 1) / 2

Tolerates small wavelength-calibration offsets between sensor and reference.

#### DTW — Dynamic Time Warping

Sakoe-Chiba band DTW with a constraint of W = `shift_window=` bands,
implemented as a rolling two-row window (constant memory):

    DTW_norm = min-cost alignment / n_bands
    similarity = exp(−DTW_norm)

Captures non-linear stretching of absorption features across sensors.
Chunked over CHUNK=512 pixels to bound peak memory usage.

#### SSIM — Spectral Structural Similarity Index

Wang et al. (2004) adapted to 1-D spectra:

    L = (2μ_r μ_p + C₁) / (μ_r² + μ_p² + C₁)     [luminance]
    C = (2σ_r σ_p + C₂) / (σ_r² + σ_p² + C₂)     [contrast]
    S = (σ_rp + C₃)    / (σ_r σ_p + C₃)           [structure]
    SSIM = L · C · S ∈ [−1, 1];   similarity = (SSIM + 1) / 2

Captures perceptually meaningful deviations across all three components
(mean, variance, covariance) simultaneously.

#### JSD — Jensen-Shannon Divergence

    m = (p + q) / 2
    JSD = (KLD(p‖m) + KLD(q‖m)) / 2 ∈ [0, ln 2]
    similarity = exp(−JSD)

Always well-defined (avoids log(0) even for exact zeros via the mixture m).
Recommended with the `-z` flag.

#### Bhattacharyya Coefficient

    BC = Σᵢ √(pᵢ qᵢ) ∈ [0, 1]

Measures overlap between two non-negative spectra treated as probability
distributions.  Already a similarity: no further transform needed.
Robust to noise in individual bands.  Recommended with the `-z` flag.

#### MTF — Matched Tuned Filter

Reed & Yu (1990) adaptive matched filter.  Builds a full image covariance
and constructs a linear detector:

    w = Σ⁻¹(d − μ) / (d − μ)ᵀ Σ⁻¹(d − μ)

where **μ** is the image mean spectrum, **Σ** is the image covariance, and
**d** is the target signature.  Maximises the signal-to-background ratio for
the target.  Scores are percentile-normalised to [0, 1].  Requires loading
the full cube to compute **Σ** (Tikhonov regularisation
`Σ ← Σ + ε·tr(Σ)/n·I` prevents singularity).

#### CEM — Constrained Energy Minimization

Chang & Heinz (2000).  Minimises mean output energy subject to the
constraint that the target response equals 1:

    w = Σ⁻¹d / (dᵀ Σ⁻¹d)

Uses only the correlation matrix (no explicit background model).  Effective
for detecting targets occupying a small fraction of pixels.  Same
normalisation and regularisation as MTF.

#### Ensemble — Borda-count rank fusion

Each pixel is ranked 1–N within each requested base method.  Ranks are
summed across methods and normalised to [0, 1].  Robust to scale differences
between methods; reduces sensitivity to any single method's failure mode.
Requires at least one other method in the method list.

### Consensus analysis (`method=consensus`)

When `consensus` is requested the module executes a four-step pipeline:

**Step 1 — Compute all base methods.**  All 16 methods except
`ensemble` / `consensus` are run against the cube.  Already-computed maps
(identified by `output_prefix=`) are reused to avoid redundant computation.

**Step 2 — Empirical CDF calibration.**  Each score map is rank-transformed
to a uniform probability in (0, 1]:

    p_m(i) = rank(score_m(i)) / N_pixels

This removes scale bias that would otherwise allow high-range methods (SAM ∈
[0.8, 1.0]) to dominate over low-range methods (ED ∈ [0.0, 0.2]).  The
calibrated maps are written as `{prefix}_cal_{method}` when
`output_prefix=` is set.

**Step 3 — Diversity weighting.**  The full *k × k* Pearson correlation
matrix is built across calibrated maps.  Each method receives weight:

    w_i ∝ 1 / (mean_{j≠i} |r_{ij}| + 0.15)

so that correlated clusters (e.g., SAM + SCA which both measure spectral
angle) do not over-count shared evidence.  Weights are rescaled so that
`mean(w) = 1`.  The weights and correlation matrix are printed when
`-v` is set.

**Step 4 — Fusion.**  Six modes are available via `fusion_mode=`:

| Mode | Formula | Characteristics |
|------|---------|-----------------|
| `rank_product` | exp(Σᵢ wᵢ ln pᵢ) | Weighted geometric mean; AND-like; robust (default) |
| `fisher` | 1 − χ²_sf(−2 Σᵢ wᵢ ln(1−pᵢ), df_eff) | Proper p-value from combined probability test (Fisher 1932) |
| `stouffer` | Φ(Σᵢ wᵢ Φ⁻¹(pᵢ) / √Σᵢ wᵢ²) | Diversity weights enter exactly via Z-score combination |
| `group_product` | geometric mean within groups; arithmetic mean across | AND within SAM/distance/info groups; OR across groups |
| `harmonic` | 1 / Σᵢ wᵢ/pᵢ | Strictest: all methods must simultaneously agree |
| `min` | min_i pᵢ | Absolute strictest; dominated by the single worst method |

For the Fisher mode, the effective degrees of freedom are
`df_eff = 2 · (Σᵢ wᵢ)² / Σᵢ wᵢ²`.

### Consensus output maps

When `output_prefix=` is set, the following maps are written in addition to
the primary `output=` hotspot probability:

| Map | Content |
|-----|---------|
| `{prefix}_consensus_agreement` | Fraction of methods voting above `agreement_threshold` [0, 1] |
| `{prefix}_consensus_entropy` | Normalised Shannon entropy of calibrated scores: 0 = unanimous, 1 = maximal conflict |
| `{prefix}_consensus_conflict` | Pixels that are simultaneously high-probability AND high-entropy (review these) |
| `{prefix}_consensus_spread` | Standard deviation of calibrated scores per pixel |
| `{prefix}_cal_{method}` | Rank-calibrated probability for each of the 16 base methods |
| `{prefix}_{method}` | Raw similarity score for each base method |

### Preprocessing flags

| Flag | Effect | Recommended with |
|------|--------|-----------------|
| `-c` | Apply upper-convex-hull continuum removal to every pixel and to the reference before matching. | `sam`, `ed`, `sca` — removes illumination / slope bias |
| `-z` | Normalise spectra to the probability simplex (sum-to-one, all values ≥ 0) before matching. | `sid`, `jsd`, `bhatt` — these methods assume probability distributions |

### Normalization options (`normalize=`)

| Option | Transform |
|--------|-----------|
| `none` | Raw reflectance (default) |
| `area` | Divide by spectral sum; total power = 1 |
| `max` | Divide by maximum band value |
| `minmax` | Map range [min, max] → [0, 1] |
| `vector` | Divide by L2 norm; unit vector in spectral space |

### Point mode (`-p`)

When `-p` is set together with `coordinates=east,north`,
*i.hyper.sleuth* extracts the spectrum at the given geographic location and
prints a ranked similarity table for every requested method.  No raster
output is produced.  Useful for validation, threshold selection, and
method-behaviour inspection before running full-image analysis.

## NOTES

### WavelengthLUT

The reference spectrum and the sensor wavelength grid will almost always
differ in sampling and range.  A `WavelengthLUT` is built once before any
pixel processing:

- `np.searchsorted` indices `left_idx`, `right_idx` and linear blend
  weights `alpha` are precomputed for every destination band.
- `apply(src_vals)` is O(n_bands) with no binary search: a single
  gather-and-blend over the two precomputed index arrays.
- `apply_cube(cube)` extends the same operation vectorized over all pixels
  via NumPy broadcasting.

The LUT also tracks:

- `valid_dst` — which sensor bands fall inside the reference wavelength range
  (bands outside are edge-filled and flagged as at risk of bias).
- `valid_src` — which reference points fall inside the sensor range
  (features outside are unobservable and trigger a warning).
- `overlap_lo / overlap_hi` — the exact wavelength interval common to both
  grids, used by `min_wavelength=` / `max_wavelength=`.

Use `-i` (info mode) to inspect the coverage report before processing.

### Input requirements

The module expects a 3D raster map with wavelength metadata stored in
band-level metadata following the *i.hyper* standard:
**wavelength**, **FWHM**, **valid**, and **unit**.  Such maps are produced
by *i.hyper.import* or *i.hyper.atcorr*.

All wavelengths must be in the same unit as the sensor metadata (typically
nanometres).  The module automatically converts µm and m to nm when reading
band metadata.

Spectra must be in surface reflectance units (0–1 range).  Top-of-atmosphere
radiance or raw DN values will produce invalid results.

### Output map types

All output maps are **floating-point FCELL** rasters in [0, 1].  The primary
`output=` map contains the similarity / hotspot probability.  All maps share
a blue → yellow → red colour ramp applied automatically via **r.colors**
(`0 = blue, 0.5 = yellow, 1 = red`).

### Performance

| Method class | Speed | Note |
|---|---|---|
| `sam`, `sid`, `ed`, `sad`, `sca`, `jsd`, `bhatt`, `ssim`, `xcorr`, `gd1`, `gd2` | Fast | Fully vectorized over all pixels |
| `mtf`, `cem` | Fast | One k×k covariance inversion, then linear |
| `dtw` | Moderate | Chunked Sakoe-Chiba rolling-window O(n·W); controlled by `shift_window=` |
| `cr_sam`, `cr_ed` | Slow | Per-pixel Graham-scan upper convex hull; ~10× slower than linear methods |
| `consensus` | Slow (first run) | Runs all 16 base methods; subsequent runs with `output_prefix=` reuse cached maps |

For a 1000 × 1000 image with 200 bands, fast methods complete in seconds;
`cr_sam`/`cr_ed` may take several minutes; `dtw` with default
`shift_window=3` is intermediate.

### Band tolerance

When the sensor and reference wavelength grids overlap only partially:

- Sensor bands outside the reference range are edge-filled (nearest
  boundary value) and printed as warnings.
- Reference features outside the sensor range are unobservable; the module
  warns which spectral region is missing.
- Use `min_wavelength=` / `max_wavelength=` to restrict matching to the
  overlap interval reported by `-i` mode.

### Threshold selection

Method-specific similarity thresholds vary significantly.  As a rough guide:

- **sam**: 0.97 (≈5°) tight target match, 0.90 (≈26°) loose
- **sid / jsd / bhatt**: > 0.8 indicates strong similarity
- **consensus probability**: > 0.7 conservative hotspot, > 0.5 candidate

Use `-p` point mode on known target and background pixels to calibrate
thresholds for your specific sensor and target.

### Flags

The **-n** flag restricts processing to only bands marked as valid
(`valid=1`) in metadata.  This excludes atmospheric water vapour absorption
bands (around 1400 nm and 1900 nm) flagged during import.

The **-i** flag prints band coverage and WavelengthLUT diagnostics (including
the LUT coverage report, edge-fill warnings, and unobservable reference
features) without processing any raster data.  Run this first.

The **-v** flag prints per-method similarity scores in point mode, and the
inter-method correlation matrix and diversity weights in consensus mode.

The **-c** flag applies upper-convex-hull continuum removal before matching.
Note: `cr_sam` and `cr_ed` always apply continuum removal internally; `-c`
additionally applies it to all other requested methods.

The **-p** flag activates point mode: extracts the spectrum at
`coordinates=east,north` and prints a ranked similarity table.  No raster
output is written.

The **-z** flag normalises both reference and pixel spectra to the
probability simplex (sum-to-one, all values ≥ 0) before matching.
Recommended for `sid`, `jsd`, and `bhatt`.

## EXAMPLES

::: code

    # SAM match against a kaolinite USGS library entry
    i.hyper.sleuth input=scene_atcorr output=kaolinite_sam \
      reference_file=kaolinite_usgs.csv method=sam

:::

::: code

    # Full consensus analysis with all methods + all diagnostic maps
    i.hyper.sleuth input=scene_atcorr output=hotspot \
      reference_file=target.csv \
      method=consensus fusion_mode=group_product \
      output_prefix=tgt

    # Produces:
    #   tgt_consensus          — hotspot probability map
    #   tgt_consensus_agreement, _entropy, _conflict, _spread  — diagnostics
    #   tgt_cal_sam, tgt_cal_sid, ... (16 calibrated probability maps)
    #   tgt_sam, tgt_sid, ...         (16 raw similarity maps)

:::

::: code

    # Point inspection: all methods at one pixel, verbose output
    i.hyper.sleuth input=scene_atcorr output=_ \
      reference="450:0.04,670:0.05,800:0.42,2200:0.18" \
      method=sam,sid,bhatt,dtw,mtf,consensus \
      coordinates="452300,4325100" -p -v

:::

::: code

    # Six methods + ensemble, per-method maps, PCHIP resampling
    i.hyper.sleuth input=scene_atcorr output=best \
      reference_file=chlorophyll_a.json \
      method=sam,cr_sam,gd1,jsd,bhatt,ensemble \
      output_prefix=chl normalize=minmax resample=pchip -c

:::

::: code

    # Subpixel mineral detection with CEM and MTF
    i.hyper.sleuth input=scene_atcorr output=kaolinite_sub \
      reference_file=kaolinite_usgs.csv \
      method=cem,mtf,ensemble output_prefix=kaolinite

:::

::: code

    # Check band coverage and WavelengthLUT diagnostics before processing
    i.hyper.sleuth input=scene_atcorr output=_ \
      reference_file=target.csv method=sam -i

    # Console output (example):
    #   src [400–2500 nm, n=200] → dst [350–2500 nm, n=242]
    #   overlap [400–2500 nm]
    #   edge-fill risk: 42 dst bands below src range (350–399 nm)
    #   unobservable: 0 src features outside dst range

:::

::: code

    # Consensus with Stouffer fusion, strict agreement threshold
    i.hyper.sleuth input=scene_atcorr output=hotspot_strict \
      reference_file=chlorophyll_a.csv \
      method=consensus fusion_mode=stouffer \
      agreement_threshold=0.90 \
      output_prefix=chl_strict

:::

::: code

    # Restrict matching to SWIR for mineral identification
    i.hyper.sleuth input=scene_atcorr output=min_match \
      reference_file=kaolinite_usgs.csv method=sam,cr_sam,gd2 \
      min_wavelength=1000 max_wavelength=2500 output_prefix=min

:::

## SEE ALSO

[i.hyper.import](i.hyper.import.html),
[i.hyper.atcorr](i.hyper.atcorr.html),
[i.hyper.continuum](i.hyper.continuum.html),
[i.hyper.spectroscopy](i.hyper.spectroscopy.html),
[i.hyper.geology](i.hyper.geology.html),
[r.mapcalc](https://grass.osgeo.org/grass-stable/manuals/r.mapcalc.html),
[r.colors](https://grass.osgeo.org/grass-stable/manuals/r.colors.html),
[r.univar](https://grass.osgeo.org/grass-stable/manuals/r.univar.html),
[r3.info](https://grass.osgeo.org/grass-stable/manuals/r3.info.html)

## REFERENCES

- Kruse F.A., Lefkoff A.B., Boardman J.W., Heidebrecht K.B., Shapiro A.T.,
  Barloon P.J., Goetz A.F.H. (1993): The spectral image processing system
  (SIPS) — Interactive visualization and analysis of imaging spectrometer
  data. *Remote Sensing of Environment*, 44, 145–163.
- Chang C.I. (2000): An information-theoretic approach to spectral
  variability, similarity, and discrimination for hyperspectral image
  analysis. *IEEE Trans. Inf. Theory*, 46(5), 1927–1932.
- Clark R.N., King T.V.V., Klejwa M., Swayze G.A. (1987): High spectral
  resolution reflectance spectroscopy of minerals. *J. Geophys. Res.*,
  92(B12), 12653–12680.
- Reed I.S., Yu X. (1990): Adaptive multiple-band CFAR detection of an
  optical pattern with unknown spectral distribution. *IEEE Trans. Acoust.
  Speech Signal Process.*, 38(10), 1760–1770.
- Chang C.I., Heinz D.C. (2000): Constrained subpixel target detection for
  remotely sensed imagery. *IEEE Trans. Geosci. Remote Sens.*, 38(3),
  1144–1159.
- Wang Z., Bovik A.C., Sheikh H.R., Simoncelli E.P. (2004): Image quality
  assessment: from error visibility to structural similarity. *IEEE Trans.
  Image Process.*, 13(4), 600–612.
- Sakoe H., Chiba S. (1978): Dynamic programming algorithm optimization for
  spoken word recognition. *IEEE Trans. Acoust. Speech Signal Process.*,
  26(1), 43–49.
- Fisher R.A. (1932): *Statistical Methods for Research Workers*, 4th ed.
  Oliver and Boyd, Edinburgh.
- Stouffer S.A., Suchman E.A., DeVinney L.C., Star S.A., Williams R.M.
  (1949): *The American Soldier: Adjustment During Army Life*, vol. 1.
  Princeton University Press.

## AUTHORS

Created for the i.hyper module family.

Based on spectral similarity methods from Kruse et al. (1993),
Chang (2000), Reed & Yu (1990), Chang & Heinz (2000), Wang et al. (2004),
and classical information theory.
