"""
Generate Central Baltic Landscape Files for DEPONS/CENOP

This script creates the necessary landscape data files for the Central Baltic region.
The Central Baltic covers approximately:
- Latitude: 54°N to 60°N (Lithuania to Sweden/Finland)
- Longitude: 13°E to 23°E (Denmark to Estonia)

EPSG:3035 (LAEA Europe) coordinates for this region:
- Approximate bounds (calculated from lat/lon):
  - XLLCORNER: ~4750000 (western edge)
  - YLLCORNER: ~3200000 (southern edge)
  - Extent: 400x400 cells at 1000m = 400km x 400km

Data sources for real implementation:
- EMODnet Bathymetry (https://emodnet.ec.europa.eu/en/bathymetry)
- HELCOM for Baltic Sea environmental data
- Copernicus Marine Service for salinity data

This generates PLACEHOLDER data with realistic distributions for testing.
For production use, replace with actual EMODnet/HELCOM data.
"""

import numpy as np
from pathlib import Path


# Central Baltic bounds in EPSG:3035 (LAEA Europe)
# These are approximate values covering the central Baltic Sea
# The grid covers from roughly Denmark/Germany to Lithuania/Latvia
XLLCORNER = 4750000.0  # Western edge (approx 13°E)
YLLCORNER = 3140000.0  # Southern edge - extended to cover Oder mouth (approx 53.4°N)
CELLSIZE = 250.0       # 250m cells for better coastal detail (lagoon spits)
NCOLS = 1800           # 450km width at 250m resolution
NROWS = 1840           # 460km height at 250m resolution
NODATA_VALUE = -9999


def write_asc_header(f, ncols, nrows, xllcorner, yllcorner, cellsize, nodata):
    """Write ASC file header."""
    f.write(f"NCOLS {ncols} \n")
    f.write(f"NROWS {nrows} \n")
    f.write(f"XLLCORNER {xllcorner} \n")
    f.write(f"YLLCORNER {yllcorner} \n")
    f.write(f"CELLSIZE {cellsize} \n")
    f.write(f"NODATA_value {nodata} \n")


def generate_baltic_bathymetry(ncols, nrows):
    """
    Generate realistic Baltic Sea bathymetry pattern.
    
    The Baltic Sea has:
    - Shallow areas near coasts (0-30m)
    - Moderate depths in central basin (30-100m)
    - Deep trenches/basins (100-459m max in Landsort Deep)
    - Several distinct basins
    """
    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)
    X, Y = np.meshgrid(x, y)
    
    # Base depth - central area deeper
    # Central Baltic has average depth ~55m
    base_depth = 30 + 50 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.15)
    
    # Add some basin structures
    # Gotland Basin (central-eastern)
    gotland_basin = 60 * np.exp(-((X - 0.6)**2 + (Y - 0.55)**2) / 0.02)
    
    # Bornholm Basin (southwestern) 
    bornholm_basin = 40 * np.exp(-((X - 0.25)**2 + (Y - 0.3)**2) / 0.015)
    
    # Gulf of Gdansk (southern)
    gdansk_shallow = -20 * np.exp(-((X - 0.4)**2 + (Y - 0.1)**2) / 0.02)
    
    # Combine basins
    depth = base_depth + gotland_basin + bornholm_basin + gdansk_shallow
    
    # Add noise for natural variation
    noise = np.random.normal(0, 5, (nrows, ncols))
    depth += noise
    
    # Coastal areas - shallow near edges
    edge_distance = np.minimum(X, 1-X)
    edge_distance = np.minimum(edge_distance, Y)
    edge_distance = np.minimum(edge_distance, 1-Y)
    
    # Make edges shallower (land/coast mask)
    coastal_mask = edge_distance < 0.05
    depth[coastal_mask] = depth[coastal_mask] * 0.3
    
    # Land mask (set negative for land)
    land_mask = edge_distance < 0.02
    depth[land_mask] = NODATA_VALUE
    
    # Add some islands/shallow areas (random)
    for _ in range(15):
        ix, iy = np.random.randint(50, ncols-50), np.random.randint(50, nrows-50)
        island_size = np.random.uniform(0.01, 0.03)
        island_mask = ((X - ix/ncols)**2 + (Y - iy/nrows)**2) < island_size**2
        depth[island_mask] = np.minimum(depth[island_mask], 5 + np.random.uniform(0, 10))
    
    # Ensure positive depths where water exists
    water_mask = depth != NODATA_VALUE
    depth[water_mask] = np.maximum(depth[water_mask], 1.0)
    
    # Cap maximum depth (Baltic max is ~459m, but most areas are shallower)
    depth[water_mask] = np.minimum(depth[water_mask], 150)
    
    return depth


def generate_dist_to_coast(depth):
    """Generate distance to coast based on depth (simplified)."""
    # Approximate: distance proportional to depth
    dist = np.where(depth == NODATA_VALUE, 0, depth * 50)  # Rough approximation
    return dist


def generate_sediment(ncols, nrows, depth):
    """
    Generate sediment type map.
    
    Sediment types (following DEPONS conventions):
    0 = Land
    1 = Mud (deep areas)
    2 = Sand (moderate depth)
    3 = Gravel (shallow)
    """
    sediment = np.zeros((nrows, ncols), dtype=int)
    
    water_mask = depth != NODATA_VALUE
    
    # Assign based on depth
    sediment[depth == NODATA_VALUE] = 0  # Land
    sediment[(depth > 0) & (depth <= 20)] = 3  # Gravel in shallows
    sediment[(depth > 20) & (depth <= 50)] = 2  # Sand
    sediment[(depth > 50)] = 1  # Mud in deep areas
    
    return sediment


def generate_food_patches(ncols, nrows, depth):
    """
    Generate food probability patches.
    
    Food availability varies spatially - higher in productive areas.
    Baltic Sea has higher productivity in coastal areas and around banks.
    """
    patches = np.zeros((nrows, ncols))
    
    water_mask = depth != NODATA_VALUE
    
    # Base food probability in water (0.3-0.8)
    patches[water_mask] = 0.3 + np.random.uniform(0, 0.5, size=np.sum(water_mask))
    
    # Higher food near coasts (upwelling, nutrients)
    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)
    X, Y = np.meshgrid(x, y)
    edge_dist = np.minimum(np.minimum(X, 1-X), np.minimum(Y, 1-Y))
    
    coastal_boost = (1 - edge_dist / 0.15) * 0.3
    coastal_boost = np.maximum(coastal_boost, 0)
    
    patches[water_mask] += coastal_boost[water_mask]
    patches = np.minimum(patches, 1.0)  # Cap at 1.0
    
    return patches


def generate_blocks(ncols, nrows, depth):
    """
    Generate block identifiers.
    
    Blocks divide the area into management zones.
    Use simple grid-based blocks.
    """
    block_size = 50  # 50x50 cells per block
    blocks = np.zeros((nrows, ncols), dtype=int)
    
    for i in range(nrows):
        for j in range(ncols):
            block_row = i // block_size
            block_col = j // block_size
            blocks[i, j] = block_row * (ncols // block_size) + block_col + 1
    
    # Set land to block 0
    blocks[depth == NODATA_VALUE] = 0
    
    return blocks


def generate_monthly_salinity(ncols, nrows, depth):
    """
    Generate monthly salinity files.
    
    Baltic Sea salinity:
    - Surface: 6-8 PSU in central Baltic
    - Increases towards Danish Straits (up to 25 PSU)
    - Decreases towards Gulf of Bothnia (2-3 PSU)
    - Seasonal variation (lower in summer due to river runoff)
    """
    salinity_data = []
    
    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)
    X, Y = np.meshgrid(x, y)
    
    # Base gradient: higher in SW (Danish Straits), lower in NE (Gulf of Bothnia)
    base_salinity = 8 + 10 * (1 - X) * (1 - Y)  # SW to NE gradient
    
    for month in range(1, 13):
        # Seasonal variation: lower in spring/summer (river runoff)
        seasonal_factor = 1.0 - 0.15 * np.sin((month - 4) * np.pi / 6)  # Min in April
        
        monthly_sal = base_salinity * seasonal_factor
        
        # Add some noise
        noise = np.random.normal(0, 0.5, (nrows, ncols))
        monthly_sal += noise
        
        # Ensure valid range
        monthly_sal = np.maximum(monthly_sal, 2)  # Min 2 PSU
        monthly_sal = np.minimum(monthly_sal, 25)  # Max 25 PSU
        
        # Set land to nodata
        monthly_sal[depth == NODATA_VALUE] = NODATA_VALUE
        
        salinity_data.append(monthly_sal)
    
    return salinity_data


def generate_monthly_prey(ncols, nrows, depth, patches):
    """
    Generate monthly prey (MaxEnt) files.
    
    Prey availability varies seasonally:
    - Higher in summer (more prey fish)
    - Lower in winter
    """
    prey_data = []
    
    for month in range(1, 13):
        # Seasonal variation: higher in summer
        seasonal_factor = 0.7 + 0.3 * np.sin((month - 3) * np.pi / 6)  # Max in June
        
        monthly_prey = patches * seasonal_factor
        
        # Add noise
        noise = np.random.normal(0, 0.1, (nrows, ncols))
        monthly_prey += noise
        
        monthly_prey = np.maximum(monthly_prey, 0)
        monthly_prey = np.minimum(monthly_prey, 1)
        
        # Set land to nodata
        monthly_prey[depth == NODATA_VALUE] = NODATA_VALUE
        
        prey_data.append(monthly_prey)
    
    return prey_data


def write_asc_file(filepath, data, ncols, nrows, xllcorner, yllcorner, cellsize, nodata):
    """Write data to ASC file."""
    with open(filepath, 'w') as f:
        write_asc_header(f, ncols, nrows, xllcorner, yllcorner, cellsize, nodata)
        
        for row in data:
            # Write row with space-separated values
            row_str = ' '.join(f'{v:.6f}' if v != nodata else str(int(nodata)) for v in row)
            f.write(row_str + '\n')
    
    print(f"  Written: {filepath}")


def main():
    """Generate all Central Baltic landscape files."""
    output_dir = Path(__file__).parent.parent.parent / "DEPONS-master" / "data" / "CentralBaltic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Central Baltic landscape files in: {output_dir}")
    print(f"Grid: {NCOLS}x{NROWS} cells, {CELLSIZE}m resolution")
    print(f"EPSG:3035 bounds: X={XLLCORNER}, Y={YLLCORNER}")
    print()
    
    # Generate base bathymetry
    print("Generating bathymetry...")
    depth = generate_baltic_bathymetry(NCOLS, NROWS)
    write_asc_file(output_dir / "bathy.asc", depth, NCOLS, NROWS, 
                   XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Generate distance to coast
    print("Generating distance to coast...")
    dist_coast = generate_dist_to_coast(depth)
    write_asc_file(output_dir / "disttocoast.asc", dist_coast, NCOLS, NROWS,
                   XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Generate sediment
    print("Generating sediment types...")
    sediment = generate_sediment(NCOLS, NROWS, depth)
    write_asc_file(output_dir / "sediment.asc", sediment.astype(float), NCOLS, NROWS,
                   XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Generate food patches
    print("Generating food patches...")
    patches = generate_food_patches(NCOLS, NROWS, depth)
    write_asc_file(output_dir / "patches.asc", patches, NCOLS, NROWS,
                   XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Generate blocks
    print("Generating blocks...")
    blocks = generate_blocks(NCOLS, NROWS, depth)
    write_asc_file(output_dir / "blocks.asc", blocks.astype(float), NCOLS, NROWS,
                   XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Generate monthly salinity
    print("Generating monthly salinity (12 months)...")
    salinity = generate_monthly_salinity(NCOLS, NROWS, depth)
    for month, sal_data in enumerate(salinity, 1):
        write_asc_file(output_dir / f"salinity{month:02d}.asc", sal_data, NCOLS, NROWS,
                       XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Generate monthly prey/entropy
    print("Generating monthly prey data (12 months)...")
    prey = generate_monthly_prey(NCOLS, NROWS, depth, patches)
    for month, prey_data in enumerate(prey, 1):
        write_asc_file(output_dir / f"prey{month:02d}.asc", prey_data, NCOLS, NROWS,
                       XLLCORNER, YLLCORNER, CELLSIZE, NODATA_VALUE)
    
    # Create empty ships.json
    ships_path = output_dir / "ships.json"
    with open(ships_path, 'w') as f:
        f.write('[]')
    print(f"  Written: {ships_path}")
    
    print()
    print("=" * 50)
    print("Central Baltic landscape generated successfully!")
    print()
    print("NOTE: This is SIMULATED data for testing purposes.")
    print("For production use, replace with real data from:")
    print("  - EMODnet Bathymetry")
    print("  - HELCOM environmental data")
    print("  - Copernicus Marine Service")
    print("=" * 50)


if __name__ == "__main__":
    main()
