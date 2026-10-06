# Warping (reprojection) of SAR Imagery in Swath Geometry

`pyswarp` is a lightweight Python package for efficient warping (reprojection) of SAR imagery in swath geometry

## Installation

```bash
pip install pyswarp
```

Or install runtime dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick start

```python
from pyswarp import SwathGeometry, RegularGeometry, Warper
from osgeo import gdal
from pyproj import Proj

# Read the first SAR image with GDAL
# The image must have ground control points (GCPs)
ds0 = gdal.Open(tiff_file0)
geo0 = SwathGeometry.from_gdal_dataset(ds0)
img0 = ds0.ReadAsArray()

# Read the second SAR image with GDAL
# The image must have ground control points (GCPs)
ds1 = gdal.Open(tiff_file1)
geo1 = SwathGeometry.from_gdal_dataset(ds1)
img1 = ds1.ReadAsArray()

# Warp img0 on img1
warper_0_1 = Warper(geo0, geo1)
img0_warped_on_img1 = warper_0_1(img0)

# Define regular geometry
proj4str = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs'
geo2 = RegularGeometry(Proj(proj4str), xmin, xmax, ymin, ymax, (x_res, y_res))

# Warp img0 on regular geometry
warper_0_2 = Warper(geo0, geo2)
img0_warped_on_img2 = warper_0_2(img0)

# Warp data from regular grid to img0
regular_data = READ_YOUR_DATA()
warper_2_0 = Warper(geo2, geo0)
reg_data_warped_on_img0 = warper_2_0(img0)

# Create SwathGeometry without GDAL
# 1. Read full resolution longitude and latitude
lon_grid = READ_LONGITUDE_GRID()
lat_grid = READ_LATITUDE_GRID()
# Subset full-res grids
col_grid_sub, row_grid_sub = np.meshgrid(
    np.linspace(0, lon_grid.shape[1] - 1, 100),
    np.linspace(0, lon_grid.shape[0] - 1, 100),
)
lon_grid_sub = lon_grid[row_grid_sub, col_grid_sub]
lat_grid_sub = lat_grid[row_grid_sub, col_grid_sub]
geo3 = SwathGeometry(
    crs = Proj('+proj=longlat'), 
    x = lon_grid_sub,
    y = lat_grid_sub, 
    col = col_grid_sub, 
    row = row_grid_sub, 
    width = lon_grid.shape[1], 
    height = lon_grid.shape[0]
)


# Use geo3 in Warper similar to geo0, geo1, etc.
```

## Citation

Korosov A., and Telegina A., Efficient algorithm fow warping SAR imagery with motion compensation, https://github.com/nansencenter/sar_image_warping


## Coming soon

Sea ice drift compensation