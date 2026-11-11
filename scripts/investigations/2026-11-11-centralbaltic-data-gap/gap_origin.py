"""Where does the CentralBaltic NODATA-on-water gap come from?

Hypothesis: scripts/generate_central_baltic.py built every layer on its own synthetic
bathymetry, whose only land is a border band (distance to the grid edge < 2%, islands are made
shallow, never land); scripts/download_emodnet_tiles.py later replaced bathy.asc alone with
EMODnet depths. Then the gap must be exactly the EMODnet water cells inside that border band,
and the valid layers must carry the generator's signatures.

Works on the raw ASC files (row 0 = north, the generator's row order). Describes the data BEFORE
scripts/fill_centralbaltic_data_gap.py (as of a39cc08); check out those files to reproduce.
Result: confirmed cell for cell (gap == water & band; every layer holds land values on the band;
salinity residual sd 0.500 against the generator gradient).
Run from the repo root: python3 scripts/investigations/2026-11-11-centralbaltic-data-gap/gap_origin.py
"""

from pathlib import Path

import numpy as np

LAND = Path(__file__).resolve().parents[3] / "data" / "CentralBaltic"


def read(name):
    with open(LAND / name) as f:
        for _ in range(6):
            f.readline()
        return np.loadtxt(f)


bathy = read("bathy.asc")
sal = read("salinity01.asc")
prey = read("prey01.asc")
dist = read("disttocoast.asc")
sed = read("sediment.asc")
patches = read("patches.asc")
blocks = read("blocks.asc")
nrows, ncols = bathy.shape
water = bathy > 0
gap = water & (sal <= -9000)
print(
    f"grid {nrows}x{ncols}; EMODnet-style water {int(water.sum())} "
    f"({water.mean():.1%}); gap {int(gap.sum())}"
)

# Generator land mask: edge_distance < 0.02 on linspace(0, 1) coordinates
X, Y = np.meshgrid(np.linspace(0, 1, ncols), np.linspace(0, 1, nrows))
edge = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
band = edge < 0.02
print(f"generator border band (edge < 0.02): {int(band.sum())} cells")
print(
    f"gap == water & band (cell for cell): {np.array_equal(gap, water & band)}; "
    f"gap outside band {int((gap & ~band).sum())}; water in band without gap "
    f"{int((water & band & ~gap).sum())}"
)
nod_any = (sal <= -9000) | (prey <= -9000)
print(
    f"layer NODATA mask == band (all cells): {np.array_equal(sal <= -9000, band)} / "
    f"prey {np.array_equal(prey <= -9000, band)}"
)
for name, arr, landval in (
    ("disttocoast", dist, 0),
    ("sediment", sed, 0),
    ("patches", patches, 0),
    ("blocks", blocks, 0),
):
    print(
        f"  {name:11s} == {landval} exactly on band: {np.array_equal(arr == landval, band)}; "
        f"on gap cells: values {np.unique(arr[gap])[:5]}"
    )

# Generator signatures on valid water (interior)
valid = water & ~gap
season1 = 1.0 - 0.15 * np.sin((1 - 4) * np.pi / 6)
expected_sal = np.clip((8 + 10 * (1 - X) * (1 - Y)) * season1, 2, 25)
res = sal[valid] - expected_sal[valid]
print(
    f"salinity01 - generator gradient on valid water: mean {res.mean():+.3f} "
    f"sd {res.std():.3f} (generator noise sd 0.5)"
)
sed_expected = np.select([bathy > 50, bathy > 20, bathy > 0], [1, 2, 3], 0)
print(
    f"sediment matches generator rule on CURRENT bathy: {(sed[valid] == sed_expected[valid]).mean():.3f}"
)
print(
    f"disttocoast == 50 * current depth on valid water: "
    f"{np.isclose(dist[valid], 50 * bathy[valid]).mean():.3f}"
)
bs = 50
expected_blocks = (
    (np.indices(bathy.shape)[0] // bs) * (ncols // bs) + np.indices(bathy.shape)[1] // bs + 1
)
print(
    f"blocks match the generator's 50x50 numbering on valid water: "
    f"{(blocks[valid] == expected_blocks[valid]).mean():.3f}"
)
print(f"patches on valid water: range [{patches[valid].min():.3f}, {patches[valid].max():.3f}]")

# Where the gap sits
rows, cols = np.indices(bathy.shape)
for label, m in (
    ("north band", rows < 37),
    ("south band", rows > nrows - 38),
    ("west band", cols < 36),
    ("east band", cols > ncols - 37),
):
    print(f"  gap in {label:10s}: {int((gap & m).sum())}")
