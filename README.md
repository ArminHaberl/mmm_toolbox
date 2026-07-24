# HORNSIM

A Python port of the **Mode Matching Method (MMM) Toolbox** for simulating the
acoustic performance of horn loudspeakers. Calculates throat impedance, internal
sound fields, far-field radiated pressure, polar response, and directivity index
for axisymmetric horns.

---

## Credits

This project is a direct port of the **[MMM Toolbox](https://github.com/bkolbrek/MMM_toolbox)**
created by **Bjørn Kolbrek** (2012–2025). All algorithms, computational
methods, and the underlying acoustic theory are his work.

- [kolbrek.hornspeakersystems.info](https://kolbrek.hornspeakersystems.info/)
- [github.com/bkolbrek/MMM_toolbox](https://github.com/bkolbrek/MMM_toolbox)

The Python translation was done by **Armin Haberl**.

---

## Installation

Requires Python ≥ 3.12. The project uses [uv](https://github.com/astral-sh/uv)
for package management.

```bash
git clone <repo-url> hornsim
cd hornsim
uv pip install -e .
```

Or with plain pip:

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pytest`.

### Data files

The library requires two precomputed data files:

| File | Purpose |
|------|---------|
| `MMM_besselzeros.mat` | Zeros of Bessel function J₁ (axisymmetric eigenvalues) |
| `ZradAS32.mat` | Precomputed modal radiation impedance lookup table (32 modes) |

These are included in the original [MMM Toolbox](https://github.com/bkolbrek/MMM_toolbox).
Place them in a `matlab/` directory at the project root, or pass explicit paths
to `init_horn_data` and `baffled_rad_zmatrix_axi`.

You can also generate `ZradAS{N}.mat` from scratch:

```bash
uv run python precompute.py          # generates ZradAS32.mat (~5-15 min)
```

Or programmatically: `precompute_rad_zmatrix(max_modes=N)`. This performs direct numerical
integration and may take several minutes for large N.

---

## Usage

A typical axisymmetric simulation follows six steps, equivalent to the original
`AxiHorndemo1.m`:

1. **Generate the horn contour** with `horn_coord_1d` — choose from 9 horn types
   (`conical`, `exponential`, `hypex`, `oswg`, `bessel`, `spherical`,
   `tractrix`, `radius`, `flared conical`)
2. **Initialize the data dictionary** with `init_horn_data` — discretizes the
   contour, assembles coupling matrices, sets up the simulation
3. **Compute radiation impedance** with `baffled_rad_zmatrix_axi` — interpolates
   a precomputed lookup table
4. **Run the core simulation** with `calculate_matrices` — propagates modal
   impedances backward from mouth to throat, producing throat impedance `Z00`
5. **Calculate radiated pressure** with `radiated_pressure_axi` — fast
   far-field modal summation at arbitrary field points
6. **Compute directivity index** with `get_di_axi` — numerical angular
   integration (Beranek / Gerzon)

The `data` dictionary carries all state between calls. See the API reference
below for key fields. For a complete runnable example, see
`tests/test_integration.py`.

---

## API reference

| Python function | MATLAB original | Description |
|-----------------|----------------|-------------|
| `horn_coord_1d(htype, yth, ym, L, Tn, dz, ...)` | `MMM_1Dhorncoord` | Generate horn contour (z, radius) |
| `make_steps(coords)` | `MMM_makesteps` | Stepped duct approximation |
| `make_fmat_axi(N, c1, c2, bz)` | `MMM_ASmakefmat` | F coupling matrix at a discontinuity |
| `make_km_axi(k, coord, M, bz)` | `MMM_ASmakekm` | Modal wavenumbers |
| `get_eigenfunctions_axi(R, r, ev, ...)` | `MMM_ASgeteigenfunctions` | Bessel eigenfunctions J₀(αr) |
| `init_horn_data(fvec, N, coords, geom, ...)` | `MMM_init` | Build simulation data dictionary |
| `make_big_fmat(N, coords, mode_info, f)` | `MMM_makebigfmat` | Assemble F matrices for all junctions |
| `calculate_matrices(data, ...)` | `MMM_calculateMatrices` | Propagate impedances mouth→throat |
| `baffled_rad_zmatrix_axi(k, ρ, c, S, M, file)` | `MMM_ASbaffledradzmatrixIntp` | Modal radiation impedance (interpolated) |
| `baffled_rad_zmatrix_direct_axi(k, ρ, c, S, M, bz)` | `MMM_ASbaffledradzmatrix` | Modal radiation impedance (direct integration) |
| `precompute_rad_zmatrix(M, ...)` | `MMM_ASbaffledradzmatrixPrecompute` | Generate a `ZradAS{N}.mat` lookup table |
| `radiated_pressure_axi(data, pts, ...)` | `MMM_ASradiatedPressure` | Far-field modal radiated pressure |
| `get_di_axi(data, angles)` | `MMM_ASgetDI` | Directivity index |

The central object is the `data` dictionary returned by `init_horn_data` and
modified in-place by `calculate_matrices`, `radiated_pressure_axi`, and
`get_di_axi`. Key fields:

| Key | Shape | Description |
|-----|-------|-------------|
| `data["fvec"]` | `(nfreq,)` | Frequency vector [Hz] |
| `data["k"]` | `(nfreq,)` | Wavenumber [rad/m] |
| `data["stepped_coords"]` | `(n_steps, 2)` | Stepped (z, radius) profile |
| `data["S"]` | `(n_steps,)` | Cross-section area at each step |
| `data["big_f"]` | `(N, N, n_steps)` | Coupling matrices at discontinuities |
| `data["Zrad"]` | `(N, N, nfreq)` | Modal radiation impedance at mouth |
| `data["BigZ"]` | `(N, N, n_steps, nfreq)` | Modal impedance at every step |
| `data["Z00"]` | `(nfreq,)` | Throat impedance (fundamental mode) |
| `data["pRad"]` | `(n_angles, nfreq)` | Radiated pressure at field points |
| `data["DI"]` | `(nfreq,)` | Directivity index [dB] |

---

## Tests

```bash
uv run pytest tests/ -v
```

11 tests validate the entire axisymmetric pipeline against MATLAB reference
outputs saved in `test_data/`. Tolerance: `atol=1e-12` for direct math, `1e-10`
for interpolation, `1e-8` for far-field pressure.

---

## Status

### Ported and validated

- All 9 axisymmetric horn contour types
- Stepped duct discretization
- Mode-matching F-matrix assembly
- Core impedance propagation (mouth → throat)
- Interpolation-based and direct-integration modal radiation impedance
- Lookup table precomputation via `precompute_rad_zmatrix`
- Far-field modal radiated pressure
- Directivity index (Beranek / Gerzon weighting)

### Not yet ported

- `MMM_ASpressureDistribution` — internal / near-field pressure field
- Rayleigh integral near-field branch of `radiated_pressure_axi`
- MATLAB plotting utilities (`MMM_PlotZth`, `MMM_ASplotHorn`, `MMM_ASpolarMap`)
- Rectangular horn geometry (requires sourcing missing `MMM_RE*` source files
  from the original MMM Toolbox)

---

## License

GPL-2.0-or-later — same license as the original MMM Toolbox.
See [LICENSE.txt](LICENSE.txt).
