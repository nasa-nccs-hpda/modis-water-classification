from typing import List, Tuple
from osgeo import gdal
from pyproj import Proj, transform

import os
import tempfile
import xarray as xr


TRANSFORM_KEY: str = "GeoTransform"
PROJ_KEY: str = "crs_wkt"
WATER_MASK_KEY: str = "water_mask"
MODIS_PROJ: str = 'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
WGS84_CRS: str = "epsg:4326"


# ----------------------------------------------------------------------------
# latlon_to_projection
# ----------------------------------------------------------------------------
def latlon_to_projection(lat, lon):
    wgs84_proj = Proj(init=WGS84_CRS)  # WGS84 EPSG code
    target_proj = Proj(MODIS_PROJ)
    x, y = transform(wgs84_proj, target_proj, lon, lat)
    return x, y


# ----------------------------------------------------------------------------
# open_and_write_temp
# ----------------------------------------------------------------------------
def open_and_write_temp(
    data_array, transform, projection, year, tile, name=None, files_to_rm=None
) -> str:
    tmpdir = tempfile.gettempdir()
    name_to_use = data_array.name if not name else name
    tempfile_name = f"MOD44W.A{year}001.{tile}.061.{name_to_use}.tif"
    tempfile_fp = os.path.join(tmpdir, tempfile_name)
    if os.path.exists(tempfile_fp):
        os.remove(tempfile_fp)
    driver = gdal.GetDriverByName("GTiff")
    outDs = driver.Create(
        tempfile_fp, 4800, 4800, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"]
    )
    outDs.SetGeoTransform(transform)
    outDs.SetProjection(projection)
    outBand = outDs.GetRasterBand(1)
    outBand.WriteArray(data_array.data[0, :, :])
    outBand.SetNoDataValue(250)
    outDs.FlushCache()
    outDs = None
    outBand = None
    driver = None
    files_to_rm.append(tempfile_fp)
    return tempfile_fp


# ----------------------------------------------------------------------------
# get_geo_info
# ----------------------------------------------------------------------------
def get_geo_info(dataset: xr.Dataset) -> Tuple[str, Tuple[float]]:
    projection = dataset.spatial_ref.attrs[PROJ_KEY]
    transform = dataset.spatial_ref.attrs[TRANSFORM_KEY]
    transform = [float(e) for e in transform.split(" ")]
    return (projection, transform)


# ----------------------------------------------------------------------------
# cleanup
# ----------------------------------------------------------------------------
def cleanup(temp_files_to_rm: List[str]) -> None:
    for path_to_delete in temp_files_to_rm:
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)
        temp_files_to_rm.remove(path_to_delete)


# ----------------------------------------------------------------------------
# zoom_to_bounds
# ----------------------------------------------------------------------------
def zoom_to_bounds(m, bounds):
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])


# ----------------------------------------------------------------------------
# cleanup
# ----------------------------------------------------------------------------
def cleanup(files_to_rm: list) -> None:
    for path_to_delete in files_to_rm:
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)

# ----------------------------------------------------------------------------
def zoom_to_bounds(m, bounds):
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])


# ----------------------------------------------------------------------------
# cleanup
# ----------------------------------------------------------------------------
def cleanup(files_to_rm: list) -> None:
    for path_to_delete in files_to_rm:
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)
