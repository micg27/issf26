import cdsapi

# Initialize the Copernicus Climate Data Store client
c = cdsapi.Client()

# Define common timeframes and variables for Phase 1
years = [str(year) for year in range(2020, 2025)]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]
times = [f"{h:02d}:00" for h in range(24)]

# Variables available in standard ERA5 (includes Boundary Layer Height)
era5_variables = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'total_precipitation',
    'boundary_layer_height'
]

# 1. DOWNLOAD SINGAPORE DATASET
print("Initiating download for Singapore (2020-2024)...")
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': era5_variables,
        'year': years,
        'month': months,
        'day': days,
        'time': times,
        'area': [1.55, 103.55, 1.15, 104.10], # North, West, South, East
    },
    'singapore_met_data_2020_2024.nc'
)

# 2. DOWNLOAD MELBOURNE DATASET
print("Initiating download for Melbourne (2020-2024)...")
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': era5_variables,
        'year': years,
        'month': months,
        'day': days,
        'time': times,
        'area': [-37.38, 144.40, -38.51, 146.30], # North, West, South, East
    },
    'melbourne_met_data_2020_2024.nc'
)

print("Download requests successfully submitted.")
