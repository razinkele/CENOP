"""Create expanded Lithuania landscape with correct bathymetry."""
import numpy as np
from pyproj import Transformer
from pathlib import Path

# EXPANDED Grid parameters - 2x north, west, south
# Original: lat 55.4-56.6, lon 19.0-21.5
# Expanded: lat 54.2-57.2 (more south and north), lon 17.5-21.5 (more west)

lat_min, lat_max = 54.2, 57.2
lon_min, lon_max = 17.5, 21.5

print(f'Target area in WGS84:')
print(f'  Lat: {lat_min} to {lat_max}')
print(f'  Lon: {lon_min} to {lon_max}')

# Transform corners to EPSG:3035
transformer_to_3035 = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
transformer_to_wgs = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)

sw_x, sw_y = transformer_to_3035.transform(lon_min, lat_min)
ne_x, ne_y = transformer_to_3035.transform(lon_max, lat_max)

print(f'\nCorners in EPSG:3035:')
print(f'  SW: ({sw_x:.0f}, {sw_y:.0f})')
print(f'  NE: ({ne_x:.0f}, {ne_y:.0f})')

# Calculate grid size with 1km cells
CELLSIZE = 1000

XLLCORNER = int(np.floor(sw_x / 1000) * 1000)
YLLCORNER = int(np.floor(sw_y / 1000) * 1000)

NCOLS = int(np.ceil((ne_x - XLLCORNER) / CELLSIZE)) + 10
NROWS = int(np.ceil((ne_y - YLLCORNER) / CELLSIZE)) + 10

print(f'\nGrid parameters:')
print(f'  NCOLS: {NCOLS}')
print(f'  NROWS: {NROWS}')
print(f'  XLLCORNER: {XLLCORNER}')
print(f'  YLLCORNER: {YLLCORNER}')
print(f'  CELLSIZE: {CELLSIZE}')

data_dir = Path("data/Lithuania")
data_dir.mkdir(exist_ok=True)

def write_asc(filename, data, nodata=-9999):
    with open(filename, 'w') as f:
        f.write(f"NCOLS {NCOLS}\n")
        f.write(f"NROWS {NROWS}\n")
        f.write(f"XLLCORNER {XLLCORNER}\n")
        f.write(f"YLLCORNER {YLLCORNER}\n")
        f.write(f"CELLSIZE {CELLSIZE}\n")
        f.write(f"NODATA_value {nodata}\n")
        for row in np.flipud(data):
            f.write(" ".join(f"{v:.2f}" if v != nodata else str(int(nodata)) for v in row) + "\n")

# Create coordinate grids
x_coords = np.arange(XLLCORNER, XLLCORNER + NCOLS * CELLSIZE, CELLSIZE)
y_coords = np.arange(YLLCORNER, YLLCORNER + NROWS * CELLSIZE, CELLSIZE)
X, Y = np.meshgrid(x_coords, y_coords)

print('\nConverting coordinates...')
lons = np.zeros((NROWS, NCOLS))
lats = np.zeros((NROWS, NCOLS))
for i in range(NROWS):
    for j in range(NCOLS):
        lons[i,j], lats[i,j] = transformer_to_wgs.transform(X[i,j], Y[i,j])

print(f"Grid lat range: {lats.min():.2f} to {lats.max():.2f}")
print(f"Grid lon range: {lons.min():.2f} to {lons.max():.2f}")

# === BATHYMETRY with REALISTIC coastline ===
print('\nGenerating bathymetry...')
depth = np.full((NROWS, NCOLS), -9999.0)

for i in range(NROWS):
    for j in range(NCOLS):
        lon = lons[i, j]
        lat = lats[i, j]

        is_land = False

        # Lithuanian mainland - east of Klaipeda (~21.0)
        if lon > 21.05:
            is_land = True

        # Curonian Spit (narrow peninsula)
        if 55.25 < lat < 55.95 and 20.85 < lon < 21.05:
            is_land = True

        # Kaliningrad region coast
        if lat < 55.0 and lon > 20.3:
            is_land = True
        if lat < 54.6 and lon > 19.8:
            is_land = True

        # Polish coast (southern edge)
        if lat < 54.5 and lon > 18.5:
            is_land = True
        if lat < 54.4 and lon > 18.0:
            is_land = True

        if is_land:
            depth[i, j] = -9999
        else:
            # Calculate depth based on distance from coast
            dist_from_east_coast = (21.0 - lon) * 50
            dist_from_south_coast = (lat - 54.5) * 80
            min_dist = min(dist_from_east_coast, dist_from_south_coast)

            if min_dist < 0:
                depth[i, j] = -9999
            elif min_dist < 5:
                depth[i, j] = 5 + min_dist * 3 + np.random.uniform(-2, 2)
            elif min_dist < 20:
                depth[i, j] = 20 + (min_dist - 5) * 2 + np.random.uniform(-3, 3)
            elif min_dist < 50:
                depth[i, j] = 50 + (min_dist - 20) * 0.8 + np.random.uniform(-5, 5)
            else:
                depth[i, j] = 75 + np.random.uniform(-10, 10)

            if depth[i, j] != -9999:
                depth[i, j] = max(3, depth[i, j])

water_cells = np.sum(depth > 0)
land_cells = np.sum(depth == -9999)
print(f"Water cells: {water_cells} ({100*water_cells/(NROWS*NCOLS):.1f}%)")
print(f"Land cells: {land_cells} ({100*land_cells/(NROWS*NCOLS):.1f}%)")
if water_cells > 0:
    print(f"Depth range (water): {depth[depth > 0].min():.1f} to {depth[depth > 0].max():.1f} m")

write_asc(data_dir / "bathy.asc", depth)
print("Created bathy.asc")

# === Other files ===
disttocoast = np.where(depth > 0, np.maximum(1, (21.0 - lons) * 60), -9999)
write_asc(data_dir / "disttocoast.asc", disttocoast)
print("Created disttocoast.asc")

sediment = np.where(depth > 0, 1.0, -9999)
write_asc(data_dir / "sediment.asc", sediment)
print("Created sediment.asc")

patches = np.where(depth > 0, 0.4 + 0.4 * np.random.random((NROWS, NCOLS)), -9999)
write_asc(data_dir / "patches.asc", patches)
print("Created patches.asc")

blocks = np.where(depth > 0, 0, -9999)
write_asc(data_dir / "blocks.asc", blocks)
print("Created blocks.asc")

for month in range(1, 13):
    seasonal = 0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)
    prey = np.where(depth > 0, 0.3 + 0.4 * seasonal + 0.2 * np.random.random((NROWS, NCOLS)), -9999)
    write_asc(data_dir / f"prey{month:02d}.asc", prey)
print("Created prey01-12.asc")

for month in range(1, 13):
    base_sal = 7.5 + 0.5 * np.sin((month - 1) * np.pi / 6)
    salinity = np.where(depth > 0, base_sal + np.random.uniform(-0.5, 0.5, (NROWS, NCOLS)), -9999)
    write_asc(data_dir / f"salinity{month:02d}.asc", salinity)
print("Created salinity01-12.asc")

# Verify turbine positions
print("\nVerifying Curonian Nord turbine positions...")
test_coords = [(20.21, 55.99), (20.42, 56.03)]
for lon, lat in test_coords:
    x, y = transformer_to_3035.transform(lon, lat)
    col = int((x - XLLCORNER) / CELLSIZE)
    row = int((y - YLLCORNER) / CELLSIZE)
    if 0 <= row < NROWS and 0 <= col < NCOLS:
        d = depth[row, col]
        print(f"  ({lon}, {lat}) -> depth={d:.1f}m - {'WATER' if d > 0 else 'LAND!'}")

# Output new bounds
grid_x_max = XLLCORNER + NCOLS * CELLSIZE
grid_y_max = YLLCORNER + NROWS * CELLSIZE
sw_lon, sw_lat = transformer_to_wgs.transform(XLLCORNER, YLLCORNER)
ne_lon, ne_lat = transformer_to_wgs.transform(grid_x_max, grid_y_max)
print(f"\nUpdate LANDSCAPE_BOUNDS to: ({sw_lat:.2f}, {ne_lat:.2f}, {sw_lon:.2f}, {ne_lon:.2f})")
print(f"\nLithuania landscape created: {NCOLS}x{NROWS} grid")
