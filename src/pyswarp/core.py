from functools import cached_property

import numpy as np
from pyproj import Proj
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.ndimage import map_coordinates

class Geometry:
    """Base class for geometric transformations between pixel/line and XY coordinates."""

    def __init__(self):
        self.width = None
        self.height = None
        self.crs = None

    @property
    def shape(self):
        """Return the shape of the geometry as (height, width)."""
        return self.height, self.width

    def colrow_to_xy(self, c, r):
        """Convert column and row coordinates to XY coordinates."""
        raise NotImplementedError("Subclasses should implement this method.")

    def get_colrow_grids(self, step):
        """Generate column and row coordinate grids with the given step size.

        Args:
            step (int): Step size for the grid.

        Returns:
            tuple: Two 2D arrays representing the column and row grids.
        """
        col_grid, row_grid = np.meshgrid(
            np.round(np.linspace(0, self.width - 1, self.width // step)).astype(int), 
            np.round(np.linspace(0, self.height - 1, self.height // step)).astype(int)
        )
        return col_grid, row_grid

    def get_xy_grids(self, step=1):
        """Generate XY coordinate grids with the given step size.

        Args:
            step (int): Step size for the grid.

        Returns:
            tuple: Two 2D arrays representing the X and Y grids.
        """
        col_grid, row_grid = self.get_colrow_grids(step)
        x_grid, y_grid = self.colrow_to_xy(col_grid, row_grid)
        return x_grid, y_grid

    def get_lonlat_grids(self, step=1):
        """Generate longitude and latitude coordinate grids with the given step size.

        Args:
            step (int): Step size for the grid.

        Returns:
            tuple: Two 2D arrays representing the longitude and latitude grids.
        """
        x_grid, y_grid = self.get_xy_grids(step)
        lon_grid, lat_grid = self.crs(x_grid, y_grid, inverse=True)
        return lon_grid, lat_grid

    def lonlat_to_xy(self, lon, lat):
        """Convert longitude and latitude coordinates to XY coordinates.

        Args:
            lon (array-like): Longitude coordinates.
            lat (array-like): Latitude coordinates.

        Returns:
            tuple: Two arrays representing the X and Y coordinates.
        """
        x, y = self.crs(lon, lat, inverse=False)
        return x, y


class SwathGeometry(Geometry):
    """Class representing a swath geometry with GCP-based transformations."""

    kernel = 'thin_plate_spline'
    epsilon = 1

    def __init__(self, crs, x, y, col, row, width, height):
        """Initialize the swath geometry with the given parameters.

        Args:
            crs (pyproj.Proj): Coordinate reference system.
            x (array-like): X coordinates of GCPs.
            y (array-like): Y coordinates of GCPs.
            col (array-like): Column coordinates of GCPs.
            row (array-like): Row coordinates of GCPs.
            width (int): Width of the raster.
            height (int): Height of the raster.
        """
        self.crs = crs
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.col = np.array(col).flatten()
        self.row = np.array(row).flatten()
        self.width = width
        self.height = height

    def copy(self):
        """Create a copy of the current SwathGeometry instance."""
        return SwathGeometry(
            self.crs,
            self.x.copy(),
            self.y.copy(),
            self.col.copy(),
            self.row.copy(),
            self.width,
            self.height
        )
                 
    @classmethod
    def from_gdal_dataset(cls, gdal_dataset, reproject_gcps=True):
        """Create a SwathGeometry instance from a GDAL dataset.

        Args:
            gdal_dataset (gdal.Dataset): GDAL dataset containing GCPs.

        Returns:
            SwathGeometry: Initialized SwathGeometry instance.
        """
        self = cls(
            crs=None,
            x=None,
            y=None,
            col=None,
            row=None,
            width=gdal_dataset.RasterXSize,
            height=gdal_dataset.RasterYSize
        )
        gcps = gdal_dataset.GetGCPs()
        self.crs = Proj(gdal_dataset.GetGCPProjection())
        self.x = np.array([p.GCPX for p in gcps])
        self.y = np.array([p.GCPY for p in gcps])
        self.col = np.array([p.GCPPixel for p in gcps])
        self.row = np.array([p.GCPLine for p in gcps])
        if reproject_gcps:
            self.reproject_gcps()
        return self

    def reproject_gcps(self):
        """Reproject the GCP coordinates to a local stereographic projection.

        This method modifies the X and Y coordinates of the GCPs in place.
        """
        if self.crs.crs.to_dict()['proj'] == 'lonlat':
            lon, lat = self.crs(self.x, self.y, inverse=True)
            clon = int(lon.mean() * 10000)/10000
            clat = int(lat.mean() * 10000)/10000
            dst_proj4str = f'+proj=stere +lat_0={clat} +lon_0={clon} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
            print('Reproject')
            dst_proj = Proj(dst_proj4str)
            self.x, self.y = dst_proj(lon, lat)
            print('Reproject - OK')

    @cached_property
    def x_interp(self):
        """Create an RBF interpolator for the X coordinates based on pixel and line coordinates."""
        return RBFInterpolator(np.column_stack([self.col, self.row]), self.x, kernel=self.kernel, epsilon=self.epsilon)

    @cached_property
    def y_interp(self):
        """Create an RBF interpolator for the Y coordinates based on column and row coordinates."""
        return RBFInterpolator(np.column_stack([self.col, self.row]), self.y, kernel=self.kernel, epsilon=self.epsilon)

    @cached_property
    def c_interp(self):
        """Create an RBF interpolator for the column coordinates based on X and Y coordinates."""
        return RBFInterpolator(np.column_stack([self.x, self.y]), self.col, kernel=self.kernel, epsilon=self.epsilon)

    @cached_property
    def r_interp(self):
        """Create an RBF interpolator for the row coordinates based on X and Y coordinates."""
        return RBFInterpolator(np.column_stack([self.x, self.y]), self.row, kernel=self.kernel, epsilon=self.epsilon)
    
    def crop(self, c_min, c_max, r_min, r_max):
        """Crop the swath geometry to the specified pixel and line ranges.

        Args:
            c_min (int): Minimum column coordinate.
            c_max (int): Maximum column coordinate.
            r_min (int): Minimum row coordinate.
            r_max (int): Maximum row coordinate.

        Returns:
            SwathGeometry: Cropped SwathGeometry instance.
        """
        new_sg = self.copy()
        new_sg.width = int(np.ceil(c_max - c_min))
        new_sg.height = int(np.ceil(r_max - r_min))
        new_sg.reproject_gcps()
        new_sg.col -= c_min
        new_sg.row -= r_min
        return new_sg

    def resize(self, factor):
        """Resize the swath geometry by the given factor.

        Args:
            factor (float): Scaling factor for width and height.

        Returns:
            SwathGeometry: Resized SwathGeometry instance.
        """
        new_sg = self.copy()
        new_sg.width = int(np.ceil(self.width * factor))
        new_sg.height = int(np.ceil(self.height * factor))  
        new_sg.reproject_gcps()
        new_sg.col *= factor
        new_sg.row *= factor
        return new_sg

    def xy_to_colrow(self, x, y):
        """Convert X and Y coordinates to column and row coordinates.

        Args:
            x (array-like): X coordinates.
            y (array-like): Y coordinates.

        Returns:
            tuple: Column and row coordinates corresponding to the input X and Y coordinates.
        """
        c = self.c_interp(np.column_stack([x.flatten(), y.flatten()]))
        r = self.r_interp(np.column_stack([x.flatten(), y.flatten()]))
        c.shape = x.shape
        r.shape = y.shape
        return c, r

    def colrow_to_xy(self, c, r):
        """Convert column and row coordinates to X and Y coordinates.

        Args:
            c (array-like): Column coordinates.
            r (array-like): Row coordinates.

        Returns:
            tuple: X and Y coordinates corresponding to the input column and row coordinates.
        """
        x = self.x_interp(np.column_stack([c.flatten(), r.flatten()]))
        y = self.y_interp(np.column_stack([c.flatten(), r.flatten()]))
        x.shape = c.shape
        y.shape = r.shape
        return x, y


class RegularGeometry(Geometry):
    """Regular grid geometry defined by a bounding box and resolution or size."""

    def __init__(self, crs, xmin, xmax, ymin, ymax, resolution=None, size=None):
        """Initialize the regular grid geometry.

        Args:
            crs (pyproj.Proj): Coordinate reference system.
            xmin (float): Minimum X coordinate of the bounding box.
            xmax (float): Maximum X coordinate of the bounding box.
            ymin (float): Minimum Y coordinate of the bounding box.
            ymax (float): Maximum Y coordinate of the bounding box.
            resolution (float or tuple, optional): Grid resolution in X and Y directions.
            size (tuple, optional): Grid size as (width, height).

        Raises:
            ValueError: If neither or both of 'resolution' and 'size' are provided, or if their values are invalid.
        """
        if (resolution is None) == (size is None):
            raise ValueError("Provide exactly one of 'resolution' or 'size'.")

        self.crs = crs
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

        if resolution is not None:
            if np.isscalar(resolution):
                self.xres = float(resolution)
                self.yres = float(resolution)
            else:
                self.xres, self.yres = map(float, resolution)

            if self.xres <= 0 or self.yres <= 0:
                raise ValueError("Resolution values must be positive.")

            self.width = int(np.ceil((self.xmax - self.xmin) / self.xres))
            self.height = int(np.ceil((self.ymax - self.ymin) / self.yres))
        else:
            self.width, self.height = map(int, size)
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Size values must be positive.")

            self.xres = (self.xmax - self.xmin) / self.width
            self.yres = (self.ymax - self.ymin) / self.height

    def colrow_to_xy(self, c, r):
        """Convert column and row coordinates to X and Y coordinates.

        Args:
            c (array-like): Column coordinates.
            r (array-like): Row coordinates.

        Returns:
            tuple: X and Y coordinates corresponding to the input pixel and line coordinates.
        """
        x = self.xmin + np.asarray(c) * self.xres
        y = self.ymax - np.asarray(r) * self.yres
        return x, y

    def xy_to_colrow(self, x, y):
        """Convert X and Y coordinates to column and row coordinates.

        Args:
            x (array-like): X coordinates.
            y (array-like): Y coordinates.

        Returns:
            tuple: Column and row coordinates corresponding to the input X and Y coordinates.
        """
        c = (np.asarray(x) - self.xmin) / self.xres
        r = (self.ymax - np.asarray(y)) / self.yres
        return c, r

class Warper:
    """Warp an image from a source geometry to a destination geometry using linear interpolation."""

    def __init__(self, src_geo, dst_geo, step=None, interpolator=LinearNDInterpolator, **kwargs):
        """Initialize the Warper with source and destination geometries.

        Args:
            src_geo (Geometry): Source geometry.
            dst_geo (Geometry): Destination geometry.
            step (int, optional): Step size for generating grids. Defaults to None.
        """
        if step is None:
            step = src_geo.shape[0] // 100
        c1, r1 = src_geo.get_colrow_grids(step)
        src_lon, src_lat = src_geo.get_lonlat_grids(step)
        c2, r2 = dst_geo.xy_to_colrow(*dst_geo.lonlat_to_xy(src_lon.flatten(), src_lat.flatten()))
        train_points = np.column_stack([r2, c2])
        interp_r1 = interpolator(train_points, r1.flatten(), **kwargs)
        interp_c1 = interpolator(train_points, c1.flatten(), **kwargs)

        cols2, rows2 = dst_geo.get_colrow_grids(1)
        points = np.column_stack([rows2.flatten(), cols2.flatten()])
        self.r1a = np.clip(interp_r1(points), 0, src_geo.shape[0] - 1).reshape(rows2.shape)
        self.c1a = np.clip(interp_c1(points), 0, src_geo.shape[1] - 1).reshape(cols2.shape)
        
    def __call__(self, src_img, order=0):
        """Warp the source image to the destination geometry.

        Args:
            src_img (array-like): Source image to be warped.
            order (int, optional): The order of the spline interpolation in map_coordinates. Defaults to 0.

        Returns:
            array-like: Warped image in the destination geometry.
        """
        dst_img = map_coordinates(src_img, (self.r1a, self.c1a), order=order)
        return dst_img