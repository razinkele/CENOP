"""
Download EMODnet Bathymetry for Central Baltic Sea

This script downloads real bathymetry data from EMODnet (European Marine 
Observation and Data Network) and converts it to ASC format for CENOP.

EMODnet Bathymetry WCS endpoint:
https://ows.emodnet-bathymetry.eu/wcs

Coverage: Central Baltic Sea
- Latitude: 53.9°N to 59.5°N (covers Oder mouth to Gulf of Finland)
- Longitude: 13°E to 22°E (Denmark to Estonia)
- Grid: 450x460 cells at 1km resolution
"""

import numpy as np
from pathlib import Path
import requests
from io import BytesIO

try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    print("WARNING: GDAL not available. Will try alternative method.")

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


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


def download_emodnet_wcs(bbox, output_path):
    """
    Download bathymetry from EMODnet WCS.
    
    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max)
        output_path: Path to save the downloaded GeoTIFF
    """
    # EMODnet Bathymetry WCS endpoint
    wcs_url = "https://ows.emodnet-bathymetry.eu/wcs"
    
    # WCS GetCoverage request parameters
    # Using EPSG:4326 for the request, will reproject later
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "CoverageId": "emodnet:mean",  # Mean depth layer
        "format": "image/tiff",
        "subset": f"Long({bbox[0]},{bbox[2]})",
        "subsettingcrs": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }
    
    # Add latitude subset
    params["subset"] = [
        f"Long({bbox[0]},{bbox[2]})",
        f"Lat({bbox[1]},{bbox[3]})"
    ]
    
    print(f"Downloading EMODnet bathymetry for bbox: {bbox}")
    print(f"URL: {wcs_url}")
    
    try:
        response = requests.get(wcs_url, params={
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "CoverageId": "emodnet:mean",
            "format": "image/tiff",
            "subset": [f"Long({bbox[0]},{bbox[2]})", f"Lat({bbox[1]},{bbox[3]})"],
        }, timeout=300)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded to: {output_path}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text[:500])
            return False
    except Exception as e:
        print(f"Download error: {e}")
        return False


def download_emodnet_alternative(bbox, output_path):
    """
    Alternative download using direct tile access or different WCS format.
    """
    # Try the REST API approach
    base_url = "https://rest.emodnet-bathymetry.eu/depth_sample"
    
    # For larger areas, we need to sample at grid points
    print("Using REST API for bathymetry sampling...")
    
    lon_min, lat_min, lon_max, lat_max = bbox
    
    # Create sample grid (1km ~ 0.009° lat, ~0.016° lon at this latitude)
    lons = np.linspace(lon_min, lon_max, NCOLS)
    lats = np.linspace(lat_min, lat_max, NROWS)
    
    depth_grid = np.full((NROWS, NCOLS), NODATA_VALUE, dtype=np.float32)
    
    # Sample points (this might take a while for full grid)
    # For efficiency, sample a subset and interpolate
    sample_step = 10  # Sample every 10th point
    
    print(f"Sampling {len(lats[::sample_step])} x {len(lons[::sample_step])} points...")
    
    sampled_depths = {}
    total_points = len(lats[::sample_step]) * len(lons[::sample_step])
    count = 0
    
    for i, lat in enumerate(lats[::sample_step]):
        for j, lon in enumerate(lons[::sample_step]):
            try:
                url = f"{base_url}?geom=POINT({lon} {lat})"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data and 'depth' in data:
                        depth = data['depth']
                        if depth is not None:
                            sampled_depths[(i * sample_step, j * sample_step)] = abs(depth)
                count += 1
                if count % 100 == 0:
                    print(f"  Sampled {count}/{total_points} points...")
            except:
                pass
    
    print(f"Sampled {len(sampled_depths)} valid depth points")
    
    # Interpolate to full grid
    if len(sampled_depths) > 0:
        from scipy.interpolate import griddata
        
        points = np.array(list(sampled_depths.keys()))
        values = np.array(list(sampled_depths.values()))
        
        grid_y, grid_x = np.mgrid[0:NROWS, 0:NCOLS]
        depth_grid = griddata(points, values, (grid_y, grid_x), method='linear', fill_value=NODATA_VALUE)
        
        return depth_grid
    
    return None


def process_geotiff_to_asc(geotiff_path, asc_path):
    """
    Process downloaded GeoTIFF to ASC format.
    Reprojects to EPSG:3035 and resamples to target grid.
    """
    if HAS_RASTERIO:
        return process_with_rasterio(geotiff_path, asc_path)
    elif HAS_GDAL:
        return process_with_gdal(geotiff_path, asc_path)
    else:
        print("ERROR: Neither rasterio nor GDAL available for processing")
        return False


def process_with_rasterio(geotiff_path, asc_path):
    """Process GeoTIFF using rasterio."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    
    dst_crs = CRS.from_epsg(3035)
    
    with rasterio.open(geotiff_path) as src:
        # Calculate transform for target CRS and resolution
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs,
            src.width, src.height,
            *src.bounds,
            resolution=CELLSIZE
        )
        
        # But we want specific dimensions
        transform = rasterio.transform.from_bounds(
            XLLCORNER, YLLCORNER,
            XLLCORNER + NCOLS * CELLSIZE,
            YLLCORNER + NROWS * CELLSIZE,
            NCOLS, NROWS
        )
        
        depth = np.empty((NROWS, NCOLS), dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=depth,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
    
    # EMODnet uses negative depths, convert to positive
    depth = np.abs(depth)
    
    # Set land (originally 0 or positive) to NODATA
    depth[depth <= 0] = NODATA_VALUE
    
    # Flip vertically (ASC format is top-to-bottom)
    depth = np.flipud(depth)
    
    # Write ASC file
    write_asc_file(asc_path, depth)
    return True


def process_with_gdal(geotiff_path, asc_path):
    """Process GeoTIFF using GDAL."""
    from osgeo import gdal, osr
    
    # Open source
    src_ds = gdal.Open(str(geotiff_path))
    if src_ds is None:
        print("Failed to open GeoTIFF")
        return False
    
    # Create target SRS
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(3035)
    
    # Warp to target grid
    dst_bounds = (
        XLLCORNER,
        YLLCORNER,
        XLLCORNER + NCOLS * CELLSIZE,
        YLLCORNER + NROWS * CELLSIZE
    )
    
    # Create in-memory output
    mem_drv = gdal.GetDriverByName('MEM')
    dst_ds = mem_drv.Create('', NCOLS, NROWS, 1, gdal.GDT_Float32)
    
    dst_gt = (XLLCORNER, CELLSIZE, 0, YLLCORNER + NROWS * CELLSIZE, 0, -CELLSIZE)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.SetProjection(dst_srs.ExportToWkt())
    
    # Reproject
    gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_Bilinear)
    
    # Read data
    depth = dst_ds.GetRasterBand(1).ReadAsArray()
    
    # EMODnet uses negative depths
    depth = np.abs(depth)
    depth[depth <= 0] = NODATA_VALUE
    
    # Write ASC
    write_asc_file(asc_path, depth)
    
    return True


def write_asc_file(filepath, data):
    """Write data to ASC format."""
    with open(filepath, 'w') as f:
        f.write(f"NCOLS {NCOLS} \n")
        f.write(f"NROWS {NROWS} \n")
        f.write(f"XLLCORNER {XLLCORNER} \n")
        f.write(f"YLLCORNER {YLLCORNER} \n")
        f.write(f"CELLSIZE {CELLSIZE} \n")
        f.write(f"NODATA_value {NODATA_VALUE} \n")
        
        for row in data:
            line = " ".join(f"{val:.6f}" if val != NODATA_VALUE else str(int(NODATA_VALUE)) for val in row)
            f.write(line + "\n")
    
    print(f"Written: {filepath}")


def download_gebco_alternative():
    """
    Alternative: Download from GEBCO if EMODnet fails.
    GEBCO provides global bathymetry.
    """
    print("\nTrying GEBCO as alternative source...")
    # GEBCO WCS is available at https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/
    # But requires different setup
    return None


def create_from_existing_data():
    """
    Create bathymetry from locally available data or manual download.
    
    Instructions for manual EMODnet download:
    1. Go to https://emodnet.ec.europa.eu/en/bathymetry
    2. Click "Data Access" -> "Download"
    3. Select area: 13°E-22°E, 53.9°N-59.5°N
    4. Download as GeoTIFF
    5. Place in this folder and run processing
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD INSTRUCTIONS                                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  The automated download may not work due to API limitations.      ║
║  Please download manually:                                        ║
║                                                                   ║
║  1. Go to: https://emodnet.ec.europa.eu/geoviewer/                ║
║                                                                   ║
║  2. In the map, navigate to Central Baltic Sea                    ║
║     (between Germany/Poland and Sweden/Finland)                   ║
║                                                                   ║
║  3. Use the "Download" tool to select the area:                   ║
║     - West: 13°E                                                  ║
║     - East: 22°E                                                  ║
║     - South: 53.9°N                                               ║
║     - North: 59.5°N                                               ║
║                                                                   ║
║  4. Download as GeoTIFF format                                    ║
║                                                                   ║
║  5. Save the file as:                                             ║
║     emodnet_central_baltic.tif                                    ║
║     in the scripts folder                                         ║
║                                                                   ║
║  6. Run this script again with --process flag                     ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download EMODnet bathymetry for Central Baltic")
    parser.add_argument("--process", action="store_true", help="Process existing GeoTIFF file")
    parser.add_argument("--input", type=str, default="emodnet_central_baltic.tif", help="Input GeoTIFF file")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "DEPONS-master" / "data" / "CentralBaltic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    geotiff_path = script_dir / args.input
    asc_path = output_dir / "bathy.asc"
    
    if args.process:
        # Process existing file
        if geotiff_path.exists():
            print(f"Processing {geotiff_path}...")
            if process_geotiff_to_asc(geotiff_path, asc_path):
                print(f"\nBathymetry saved to: {asc_path}")
                
                # Copy to cenop/data as well
                cenop_data = script_dir.parent / "data" / "CentralBaltic"
                cenop_data.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(asc_path, cenop_data / "bathy.asc")
                print(f"Also copied to: {cenop_data / 'bathy.asc'}")
        else:
            print(f"ERROR: File not found: {geotiff_path}")
            create_from_existing_data()
        return
    
    # Try automatic download
    bbox = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    
    print("Attempting automated EMODnet download...")
    if download_emodnet_wcs(bbox, geotiff_path):
        process_geotiff_to_asc(geotiff_path, asc_path)
    else:
        print("\nAutomated download failed. Showing manual instructions...")
        create_from_existing_data()


if __name__ == "__main__":
    main()
