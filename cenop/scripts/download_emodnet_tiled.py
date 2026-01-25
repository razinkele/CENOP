"""
Download EMODnet Bathymetry for Central Baltic Sea - Tiled Approach

Downloads bathymetry in smaller tiles to avoid WCS size limits,
then merges them into the final grid.

EMODnet WCS has ~100MB limit per request, so we split into tiles.
"""

import numpy as np
from pathlib import Path
import requests
import tempfile
import time

# Central Baltic bounds (WGS84)
LAT_MIN = 53.9
LAT_MAX = 59.5
LON_MIN = 13.0
LON_MAX = 22.0

# Target grid parameters (EPSG:3035)
XLLCORNER = 4750000.0
YLLCORNER = 3140000.0
CELLSIZE = 1000.0
NCOLS = 450
NROWS = 460
NODATA_VALUE = -9999

# Tile configuration (smaller tiles to stay under limit)
TILE_SIZE_DEG = 1.5  # degrees per tile


def download_tile(lon_min, lat_min, lon_max, lat_max):
    """Download a single tile from EMODnet WCS."""
    wcs_url = "https://ows.emodnet-bathymetry.eu/wcs"
    
    params = {
        "service": "WCS",
        "version": "2.0.1", 
        "request": "GetCoverage",
        "CoverageId": "emodnet:mean",
        "format": "image/tiff",
        "subset": [f"Long({lon_min},{lon_max})", f"Lat({lat_min},{lat_max})"],
    }
    
    try:
        response = requests.get(wcs_url, params=params, timeout=120)
        if response.status_code == 200 and len(response.content) > 1000:
            return response.content
        else:
            print(f"  Tile failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  Tile error: {e}")
        return None


def parse_geotiff_simple(data):
    """
    Simple GeoTIFF parser for EMODnet tiles.
    Returns depth array and bounds.
    """
    try:
        import struct
        from io import BytesIO
        
        f = BytesIO(data)
        
        # Check TIFF magic number
        magic = f.read(2)
        if magic == b'II':  # Little endian
            byte_order = '<'
        elif magic == b'MM':  # Big endian
            byte_order = '>'
        else:
            return None, None
        
        # Skip to rasterio/GDAL processing
        # This is complex, use alternative approach
        return None, None
        
    except Exception as e:
        return None, None


def download_depth_points():
    """
    Download depth at sample points using REST API.
    This is slower but more reliable.
    """
    print("Downloading depth data using point sampling...")
    print("This may take a few minutes...")
    
    # Create grid of sample points at ~1km resolution
    # At ~57°N: 1km ≈ 0.009° lat, 0.018° lon
    lat_step = 1.0 / 111.0  # ~1km in degrees
    lon_step = 1.0 / 60.0   # ~1km at this latitude
    
    # Sample at lower resolution first, then interpolate
    sample_factor = 5  # Sample every 5km
    
    lats = np.linspace(LAT_MIN, LAT_MAX, NROWS // sample_factor + 1)
    lons = np.linspace(LON_MIN, LON_MAX, NCOLS // sample_factor + 1)
    
    print(f"Sampling {len(lats)} x {len(lons)} = {len(lats) * len(lons)} points")
    
    # Store sampled depths
    depth_samples = np.full((len(lats), len(lons)), np.nan)
    
    base_url = "https://rest.emodnet-bathymetry.eu"
    
    total = len(lats) * len(lons)
    count = 0
    errors = 0
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            count += 1
            
            try:
                # Use depth_sample endpoint
                url = f"{base_url}/depth_sample"
                params = {"geom": f"POINT({lon} {lat})"}
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and 'depth' in data and data['depth'] is not None:
                        # EMODnet returns negative depths for underwater
                        depth_samples[i, j] = abs(data['depth'])
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
            
            # Progress update
            if count % 50 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.1f}%) - Errors: {errors}")
            
            # Rate limiting
            time.sleep(0.05)
    
    print(f"Completed: {count} points, {errors} errors")
    
    return lats, lons, depth_samples


def interpolate_to_grid(lats, lons, samples):
    """Interpolate sampled points to full grid."""
    from scipy.interpolate import RegularGridInterpolator
    
    print("Interpolating to full grid...")
    
    # Create interpolator
    # Note: samples[i,j] corresponds to lats[i], lons[j]
    # Handle NaN values
    valid_mask = ~np.isnan(samples)
    
    if np.sum(valid_mask) < 10:
        print("ERROR: Too few valid samples")
        return None
    
    # Fill NaN with nearest neighbor first
    from scipy.ndimage import distance_transform_edt
    
    if np.any(np.isnan(samples)):
        # Create filled version for interpolation
        filled = samples.copy()
        mask = np.isnan(filled)
        
        # Get indices of valid points
        valid_indices = np.where(~mask)
        invalid_indices = np.where(mask)
        
        if len(valid_indices[0]) > 0 and len(invalid_indices[0]) > 0:
            # Simple nearest-neighbor fill for NaN values
            from scipy.spatial import cKDTree
            valid_points = np.column_stack(valid_indices)
            tree = cKDTree(valid_points)
            invalid_points = np.column_stack(invalid_indices)
            _, nearest_idx = tree.query(invalid_points)
            filled[mask] = samples[valid_indices[0][nearest_idx], valid_indices[1][nearest_idx]]
        
        samples = filled
    
    # Create full resolution grid
    full_lats = np.linspace(LAT_MIN, LAT_MAX, NROWS)
    full_lons = np.linspace(LON_MIN, LON_MAX, NCOLS)
    
    # Interpolate
    interp = RegularGridInterpolator((lats, lons), samples, method='linear', bounds_error=False, fill_value=NODATA_VALUE)
    
    full_grid = np.zeros((NROWS, NCOLS))
    for i, lat in enumerate(full_lats):
        for j, lon in enumerate(full_lons):
            full_grid[i, j] = interp((lat, lon))
    
    return full_grid


def add_land_mask(depth_grid):
    """
    Add land mask based on coastline.
    Cells with depth = 0 or very shallow are marked as land.
    """
    # Mark very shallow or zero as land
    land_mask = (depth_grid <= 0) | np.isnan(depth_grid)
    depth_grid[land_mask] = NODATA_VALUE
    
    # Also mask the southern edge (more land there)
    # This is approximate - real coastline would need shapefile
    return depth_grid


def write_asc_file(filepath, data):
    """Write data to ASC format."""
    print(f"Writing: {filepath}")
    
    with open(filepath, 'w') as f:
        f.write(f"NCOLS {NCOLS} \n")
        f.write(f"NROWS {NROWS} \n")
        f.write(f"XLLCORNER {XLLCORNER} \n")
        f.write(f"YLLCORNER {YLLCORNER} \n")
        f.write(f"CELLSIZE {CELLSIZE} \n")
        f.write(f"NODATA_value {NODATA_VALUE} \n")
        
        # ASC files are written from top (north) to bottom (south)
        for row in reversed(data):
            line = " ".join(f"{val:.2f}" if val != NODATA_VALUE else str(int(NODATA_VALUE)) for val in row)
            f.write(line + "\n")


def main():
    print("=" * 60)
    print("EMODnet Bathymetry Downloader - Central Baltic Sea")
    print("=" * 60)
    print(f"Area: {LON_MIN}°E to {LON_MAX}°E, {LAT_MIN}°N to {LAT_MAX}°N")
    print(f"Grid: {NCOLS} x {NROWS} cells at {CELLSIZE}m")
    print()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "DEPONS-master" / "data" / "CentralBaltic"
    cenop_dir = script_dir.parent / "data" / "CentralBaltic"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cenop_dir.mkdir(parents=True, exist_ok=True)
    
    # Download depth samples
    lats, lons, samples = download_depth_points()
    
    # Check if we got enough data
    valid_count = np.sum(~np.isnan(samples))
    print(f"\nValid depth samples: {valid_count} / {samples.size}")
    
    if valid_count < 100:
        print("ERROR: Not enough valid samples. EMODnet may be rate-limiting.")
        print("Try again later or download manually from:")
        print("  https://emodnet.ec.europa.eu/geoviewer/")
        return
    
    # Interpolate to full grid
    depth_grid = interpolate_to_grid(lats, lons, samples)
    
    if depth_grid is None:
        print("ERROR: Interpolation failed")
        return
    
    # Add land mask
    depth_grid = add_land_mask(depth_grid)
    
    # Statistics
    water_mask = depth_grid != NODATA_VALUE
    if np.any(water_mask):
        print(f"\nDepth statistics:")
        print(f"  Water cells: {np.sum(water_mask)}")
        print(f"  Land cells: {np.sum(~water_mask)}")
        print(f"  Min depth: {np.min(depth_grid[water_mask]):.1f}m")
        print(f"  Max depth: {np.max(depth_grid[water_mask]):.1f}m")
        print(f"  Mean depth: {np.mean(depth_grid[water_mask]):.1f}m")
    
    # Save
    write_asc_file(output_dir / "bathy.asc", depth_grid)
    write_asc_file(cenop_dir / "bathy.asc", depth_grid)
    
    print("\n" + "=" * 60)
    print("EMODnet bathymetry downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
