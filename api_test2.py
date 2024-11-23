import openeo
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Connect to Copernicus openEO backend
connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
# Authenticate using refresh token or OIDC (whichever method you are using)
connection.authenticate_oidc()

# Define spatial and temporal extent
spatial_extent = {"west": 5.05, "south": 51.21, "east": 5.1, "north": 51.23}
temporal_extent = ["2022-05-01", "2022-05-30"]

# Load Sentinel-2 data with the bands needed for NDVI and False Color (B04, B08, B03, B02)
cube = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=["B04", "B08", "B02"],  # Red (B04), Near-Infrared (B08), Blue (B02)
    max_cloud_cover=80
)

# Compute NDVI: (B08 - B04) / (B08 + B04)
ndvi_cube = cube.ndvi()

# Apply transformations (take the maximum over time for simplicity)
ndvi_max = ndvi_cube.max_time()

# Download the processed NDVI data as a GeoTIFF file
ndvi_max.download("sentinel_ndvi_max.tiff")

# Read the NDVI image data
with rasterio.open("sentinel_ndvi_max.tiff") as src:
    ndvi_data = src.read(1)

# Clip NDVI values for visualization (between -1 and 1)
ndvi_data = np.clip(ndvi_data, -1, 1)

# Create custom NDVI colormap based on the provided ranges
ndvi_cmap = mcolors.ListedColormap([
    "#0c0c0c",  # NDVI < -0.5
    "#eaeaea",  # -0.5 < NDVI ≤ 0
    "#ccc682",  # 0 < NDVI ≤ 0.1
    "#91bf51",  # 0.1 < NDVI ≤ 0.2
    "#70a33f",  # 0.2 < NDVI ≤ 0.3
    "#4f892d",  # 0.3 < NDVI ≤ 0.4
    "#306d1c",  # 0.4 < NDVI ≤ 0.5
    "#0f540a",  # 0.5 < NDVI ≤ 0.6
    "#004400"   # 0.6 < NDVI ≤ 1.0
])

# Save the NDVI image with the custom colormap
plt.imsave("sentinel_ndvi_max.png", ndvi_data, cmap=ndvi_cmap)

# Now, let's create the false color image from B08, B04, and B02 bands
# Download these bands as GeoTIFFs
cube_b08 = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=["B08"],
    max_cloud_cover=80
).max_time()

cube_b04 = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=["B04"],
    max_cloud_cover=80
).max_time()

cube_b02 = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=["B02"],
    max_cloud_cover=80
).max_time()

# Download the individual bands
cube_b08.download("B08.tiff")
cube_b04.download("B04.tiff")
cube_b02.download("B02.tiff")

# Read the individual band data
with rasterio.open("B08.tiff") as b08_src:
    b08_data = b08_src.read(1)

with rasterio.open("B04.tiff") as b04_src:
    b04_data = b04_src.read(1)

with rasterio.open("B02.tiff") as b02_src:
    b02_data = b02_src.read(1)

# Stack the bands for False Color (B08 for Red, B04 for Green, B02 for Blue)
false_color_image = np.stack([b08_data, b04_data, b02_data], axis=-1)

# Normalize the false color image (clip and normalize)
false_color_image = np.clip(false_color_image, 0, 3000)  # Adjust based on sensor range
false_color_image = false_color_image / 3000  # Normalize to [0, 1]

# Save the False Color image as a PNG
plt.imsave("sentinel_false_color.png", false_color_image)

print("NDVI and False Color images have been saved as PNG files.")
