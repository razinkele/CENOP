"""
Download EMODnet Bathymetry in Tiles for Central Baltic Sea

This script downloads bathymetry data from EMODnet in smaller tiles
to avoid the WCS service size limits, then stitches them together.

Target: Central Baltic Sea
- 450 x 460 cells at 1km resolution
- EPSG:3035 coordinates
"""

import numpy as np
from pathlib import Path
import requests
import time
import sys

# Try to import optional dependencies
try:
    from scipy import ndimage
    from scipy.interpolate import griddata
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("NOTE: scipy not available - using simple interpolation")

# Target grid parameters
# 250m resolution for better coastal detail (lagoon spits)
NCOLS = 1800
NROWS = 1840
CELLSIZE = 250.0
XLLCORNER = 4750000.0
YLLCORNER = 3140000.0
NODATA_VALUE = -9999

# WGS84 bounding box
LON_MIN = 13.0
LON_MAX = 22.0
LAT_MIN = 53.9
LAT_MAX = 59.5

# Output directories
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "DEPONS-master" / "data" / "CentralBaltic"
CENOP_DATA_DIR = SCRIPT_DIR.parent / "data" / "CentralBaltic"


def download_emodnet_tile(lon_min, lat_min, lon_max, lat_max, max_retries=3):
    """
    Download a single tile from EMODnet WCS.
    Returns depth data as numpy array or None if failed.
    """
    wcs_url = "https://ows.emodnet-bathymetry.eu/wcs"
    
    # Request parameters for a small tile
    # Using ~100m resolution to stay under size limits
    url = (f"{wcs_url}?service=WCS&version=2.0.1&request=GetCoverage"
           f"&CoverageId=emodnet:mean"
           f"&format=image/tiff"
           f"&subset=Long({lon_min},{lon_max})"
           f"&subset=Lat({lat_min},{lat_max})")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Check if it's a valid TIFF
                if response.content[:4] in [b'II*\x00', b'MM\x00*']:
                    return response.content
                else:
                    # Might be an error message
                    if b'Exception' in response.content:
                        print(f"    WCS error: {response.content[:200]}")
                        return None
            else:
                print(f"    HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
        except Exception as e:
            print(f"    Error: {e}")
            
    return None


def extract_depth_from_tiff(tiff_data):
    """
    Extract depth values from GeoTIFF data.
    Requires rasterio or returns None.
    """
    try:
        import rasterio
        from io import BytesIO
        
        with rasterio.open(BytesIO(tiff_data)) as src:
            data = src.read(1)
            nodata = src.nodata
            
            # Convert to float and handle nodata
            data = data.astype(np.float32)
            if nodata is not None:
                data[data == nodata] = np.nan
                
            # EMODnet depths are negative - make positive
            data = -data
            
            return data, src.bounds
            
    except ImportError:
        print("rasterio not available for TIFF processing")
        return None, None
    except Exception as e:
        print(f"Error reading TIFF: {e}")
        return None, None


def download_with_tiles(n_tiles_x=3, n_tiles_y=3):
    """
    Download the full area in tiles and stitch together.
    """
    print(f"Downloading EMODnet bathymetry in {n_tiles_x}x{n_tiles_y} tiles...")
    
    tile_data = []
    tile_bounds = []
    
    lon_step = (LON_MAX - LON_MIN) / n_tiles_x
    lat_step = (LAT_MAX - LAT_MIN) / n_tiles_y
    
    total_tiles = n_tiles_x * n_tiles_y
    tile_num = 0
    
    for j in range(n_tiles_y):
        row_data = []
        for i in range(n_tiles_x):
            tile_num += 1
            
            t_lon_min = LON_MIN + i * lon_step
            t_lon_max = LON_MIN + (i + 1) * lon_step
            t_lat_min = LAT_MIN + j * lat_step
            t_lat_max = LAT_MIN + (j + 1) * lat_step
            
            print(f"  Tile {tile_num}/{total_tiles}: ({t_lon_min:.1f}-{t_lon_max:.1f}°E, {t_lat_min:.1f}-{t_lat_max:.1f}°N)...", end=" ")
            
            tiff_data = download_emodnet_tile(t_lon_min, t_lat_min, t_lon_max, t_lat_max)
            
            if tiff_data:
                depth, bounds = extract_depth_from_tiff(tiff_data)
                if depth is not None:
                    row_data.append(depth)
                    print(f"OK ({depth.shape})")
                else:
                    row_data.append(None)
                    print("Failed to parse")
            else:
                row_data.append(None)
                print("Download failed")
            
            time.sleep(0.5)  # Be nice to the server
            
        tile_data.append(row_data)
    
    # Stitch tiles together
    return stitch_tiles(tile_data, n_tiles_x, n_tiles_y)


def stitch_tiles(tile_data, n_tiles_x, n_tiles_y):
    """
    Stitch downloaded tiles into a single array.
    """
    print("\nStitching tiles...")
    
    # Find valid tiles to determine dimensions
    valid_tiles = []
    for row in tile_data:
        for tile in row:
            if tile is not None:
                valid_tiles.append(tile)
    
    if not valid_tiles:
        print("No valid tiles downloaded!")
        return None
    
    # Get typical tile size
    typical_shape = valid_tiles[0].shape
    
    # Create output array
    total_rows = typical_shape[0] * n_tiles_y
    total_cols = typical_shape[1] * n_tiles_x
    
    full_data = np.full((total_rows, total_cols), np.nan, dtype=np.float32)
    
    # Place tiles - note: tiles are ordered bottom-to-top in lat
    for j, row in enumerate(tile_data):
        for i, tile in enumerate(row):
            if tile is not None:
                # Calculate position (flip j because lat increases upward)
                row_start = (n_tiles_y - 1 - j) * typical_shape[0]
                row_end = row_start + tile.shape[0]
                col_start = i * typical_shape[1]
                col_end = col_start + tile.shape[1]
                
                # Handle size mismatches at edges
                tile_rows = min(tile.shape[0], row_end - row_start)
                tile_cols = min(tile.shape[1], col_end - col_start)
                
                full_data[row_start:row_start+tile_rows, col_start:col_start+tile_cols] = tile[:tile_rows, :tile_cols]
    
    print(f"  Stitched size: {full_data.shape}")
    
    # Resample to target grid
    return resample_to_target(full_data)


def resample_to_target(data):
    """
    Resample the downloaded data to target grid dimensions.
    """
    print(f"Resampling to {NROWS}x{NCOLS}...")
    
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_y = NROWS / data.shape[0]
    zoom_x = NCOLS / data.shape[1]
    
    # Resample
    resampled = zoom(data, (zoom_y, zoom_x), order=1)
    
    # Handle NaN values
    resampled[np.isnan(resampled)] = NODATA_VALUE
    
    # Set land (depth < MIN_DEPTH for porpoises) to NODATA
    # DEPONS uses min_depth = 1.0m as minimum water depth for porpoises
    MIN_DEPTH = 1.0  # meters - porpoises can't swim in water shallower than this
    resampled[resampled < MIN_DEPTH] = NODATA_VALUE
    
    print(f"  Resampled size: {resampled.shape}")
    
    valid = resampled[resampled != NODATA_VALUE]
    if len(valid) > 0:
        print(f"  Depth range: {valid.min():.1f}m to {valid.max():.1f}m")
    
    return resampled


def use_depth_api_sampling():
    """
    Alternative: Use EMODnet REST API to sample depth at grid points.
    This is slower but more reliable.
    """
    print("\nUsing EMODnet REST API for depth sampling...")
    print("(This method samples individual points - may take several minutes)")
    
    api_url = "https://rest.emodnet-bathymetry.eu/depth_sample"
    
    # Sample at lower resolution first, then interpolate
    sample_step = 5  # Sample every 5th cell
    
    lat_step = (LAT_MAX - LAT_MIN) / NROWS
    lon_step = (LON_MAX - LON_MIN) / NCOLS
    
    sample_rows = range(0, NROWS, sample_step)
    sample_cols = range(0, NCOLS, sample_step)
    total_samples = len(list(sample_rows)) * len(list(sample_cols))
    
    print(f"  Sampling {total_samples} points...")
    
    sample_points = []
    sample_values = []
    count = 0
    failed = 0
    
    for row in range(0, NROWS, sample_step):
        lat = LAT_MAX - (row + 0.5) * lat_step
        for col in range(0, NCOLS, sample_step):
            lon = LON_MIN + (col + 0.5) * lon_step
            
            try:
                response = requests.get(
                    api_url,
                    params={"lat": lat, "lon": lon},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data and 'depth' in data and data['depth'] is not None:
                        depth = abs(float(data['depth']))
                        if depth > 0:
                            sample_points.append((row, col))
                            sample_values.append(depth)
                            
            except Exception as e:
                failed += 1
                
            count += 1
            if count % 100 == 0:
                print(f"    Sampled {count}/{total_samples} points ({len(sample_values)} valid, {failed} failed)...")
            
            # Small delay to avoid rate limiting
            if count % 50 == 0:
                time.sleep(0.1)
    
    print(f"  Got {len(sample_values)} valid depth samples")
    
    if len(sample_values) < 50:
        print("  Not enough samples - API may be unavailable")
        return None
    
    # Interpolate to full grid
    return interpolate_to_grid(sample_points, sample_values)


def interpolate_to_grid(points, values):
    """
    Interpolate sampled points to full grid.
    """
    if not HAS_SCIPY:
        print("scipy required for interpolation")
        return None
    
    print("  Interpolating to full grid...")
    
    points = np.array(points)
    values = np.array(values)
    
    # Create full grid
    grid_rows, grid_cols = np.mgrid[0:NROWS, 0:NCOLS]
    
    # Interpolate
    depths = griddata(points, values, (grid_rows, grid_cols), method='linear')
    
    # Fill remaining NaN with nearest neighbor
    mask = np.isnan(depths)
    if mask.any():
        depths_nearest = griddata(points, values, (grid_rows, grid_cols), method='nearest')
        depths[mask] = depths_nearest[mask]
    
    # Set remaining NaN to NODATA
    depths[np.isnan(depths)] = NODATA_VALUE
    
    # Stats
    valid = depths[depths != NODATA_VALUE]
    print(f"  Interpolated: {len(valid)} water cells")
    print(f"  Depth range: {valid.min():.1f}m to {valid.max():.1f}m")
    
    return depths.astype(np.float32)


def write_asc_file(filepath, data):
    """Write data to ASC file format."""
    print(f"Writing: {filepath}")
    
    with open(filepath, 'w') as f:
        f.write(f"NCOLS {NCOLS} \n")
        f.write(f"NROWS {NROWS} \n")
        f.write(f"XLLCORNER {XLLCORNER} \n")
        f.write(f"YLLCORNER {YLLCORNER} \n")
        f.write(f"CELLSIZE {CELLSIZE} \n")
        f.write(f"NODATA_value {NODATA_VALUE} \n")
        
        for row in data:
            line = ' '.join(
                f'{val:.6f}' if val != NODATA_VALUE else str(int(NODATA_VALUE))
                for val in row
            )
            f.write(line + '\n')


def main():
    print("=" * 60)
    print("EMODnet Bathymetry Download - Central Baltic")
    print("=" * 60)
    print(f"Target: {NCOLS}x{NROWS} grid at {CELLSIZE}m resolution")
    print(f"Area: {LON_MIN}°-{LON_MAX}°E, {LAT_MIN}°-{LAT_MAX}°N")
    
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CENOP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    depths = None
    
    # Method 1: Try downloading in tiles
    try:
        import rasterio
        print("\nMethod 1: Tile-based download (WCS)...")
        depths = download_with_tiles(n_tiles_x=4, n_tiles_y=3)
    except ImportError:
        print("\nrasterio not available - skipping tile method")
    except Exception as e:
        print(f"\nTile download failed: {e}")
    
    # Method 2: REST API sampling
    if depths is None:
        print("\nMethod 2: REST API sampling...")
        depths = use_depth_api_sampling()
    
    # Save results
    if depths is not None:
        # Save to both locations
        write_asc_file(OUTPUT_DIR / "bathy.asc", depths)
        write_asc_file(CENOP_DATA_DIR / "bathy.asc", depths)
        
        print("\n" + "=" * 60)
        print("SUCCESS! EMODnet bathymetry downloaded.")
        print(f"Saved to: {OUTPUT_DIR / 'bathy.asc'}")
        print(f"Copy at: {CENOP_DATA_DIR / 'bathy.asc'}")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED: Could not download bathymetry")
        print("Please download manually from:")
        print("  https://emodnet.ec.europa.eu/geoviewer/")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
