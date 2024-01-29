from typing import Optional, Union, List, Dict, Tuple
import ipywidgets as widgets
from osgeo import gdal
import rasterio
import ipyleaflet
from pyproj import Proj, transform

from localtileserver import (
    get_leaflet_tile_layer,
    TileClient,
)
import os
from io import BytesIO
from base64 import encodebytes
import pathlib
import rioxarray as rxr
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt


NUMBER_YEARS: int = (
    23  # number of files a single tile's time-series should match
)
DATE_SPLIT_KEY: str = "MOD44W.A"
TRANSFORM_KEY: str = "GeoTransform"
PROJ_KEY: str = "crs_wkt"
WATER_MASK_KEY: str = "water_mask"
CSS_BASE_PATH: str = "/css/modis/Collection6.1/L3/MOD44W-LandWaterMask"
MODIS_PROJ: str = 'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
WGS84_CRS: str = "epsg:4326"


def latlon_to_projection(lat, lon):
    # Define WGS84 projection for latitude/longitude
    wgs84_proj = Proj(init=WGS84_CRS)  # WGS84 EPSG code

    # Convert latitude and longitude to the target projection
    target_proj = Proj(MODIS_PROJ)
    x, y = transform(wgs84_proj, target_proj, lon, lat)

    return x, y


def get_mod44w_time_series(tile: str) -> List[pathlib.Path]:
    base_dir = pathlib.Path(CSS_BASE_PATH)
    regex = f"*/*/MOD44W.A*.{tile}.061.*.hdf"
    files_matching_regex = sorted(
        [str(file_match) for file_match in base_dir.glob(regex)]
    )
    n_files_matching_regex = len(files_matching_regex)
    assert n_files_matching_regex == NUMBER_YEARS, (
        f"Expected {NUMBER_YEARS} files matching {regex},"
        + f" got {n_files_matching_regex}"
    )
    return files_matching_regex


def merge_water_masks(layers: List[str]) -> xr.Dataset:
    datasets = []

    for file_path in layers:
        ds = rxr.open_rasterio(file_path)
        time_str = file_path.split(DATE_SPLIT_KEY)[-1][:4]
        time_format = "%Y"
        time = datetime.strptime(time_str, time_format)
        ds["time"] = time
        datasets.append(ds)

    merged_dataset = xr.concat(datasets, dim="time")
    merged_dataset = merged_dataset.sortby("time")

    return merged_dataset


def open_and_write_temp(
    data_array, transform, projection, year, tile, name=None, files_to_rm=None
) -> str:
    import tempfile

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


def get_geo_info(dataset: xr.Dataset) -> Tuple[str, Tuple[float]]:
    projection = dataset.spatial_ref.attrs[PROJ_KEY]
    transform = dataset.spatial_ref.attrs[TRANSFORM_KEY]
    transform = [float(e) for e in transform.split(" ")]
    return (projection, transform)


def cleanup(temp_files_to_rm: List[str]) -> None:
    for path_to_delete in temp_files_to_rm:
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)
        temp_files_to_rm.remove(path_to_delete)


def images_to_tiles(
    images: Union[str, List[str]], names: List[str] = None, **kwargs
) -> Dict[str, ipyleaflet.TileLayer]:
    """Convert a list of images to a dictionary of ipyleaflet.TileLayer objects.

    Args:
        images (str | list): The path to a directory of images or a list of image paths.
        names (list, optional): A list of names for the layers. Defaults to None.
        **kwargs: Additional arguments to pass to get_local_tile_layer().

    Returns:
        dict: A dictionary of ipyleaflet.TileLayer objects.
    """

    tiles = {}

    if names is None:
        names = [
            os.path.splitext(os.path.basename(image))[0] for image in images
        ]

    for index, image in enumerate(images):
        name = names[index]
        try:
            tile = get_local_tile_layer(image, layer_name=name, **kwargs)
            tiles[name] = tile
        except Exception as e:
            print(image, e)

    return tiles


def get_local_tile_layer(
    source,
    port="default",
    debug=False,
    projection="EPSG:3857",
    band=None,
    vmin=None,
    vmax=None,
    nodata=None,
    attribution=None,
    layer_name="Local COG",
    return_client=False,
    **kwargs,
):
    """Generate an ipyleaflet/folium TileLayer from a local raster dataset or remote Cloud Optimized GeoTIFF (COG).
        If you are using this function in JupyterHub on a remote server and the raster does not render properly, try
        running the following two lines before calling this function:

        import os
        os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

    Args:
        source (str): The path to the GeoTIFF file or the URL of the Cloud Optimized GeoTIFF.
        port (str, optional): The port to use for the server. Defaults to "default".
        debug (bool, optional): If True, the server will be started in debug mode. Defaults to False.
        projection (str, optional): The projection of the GeoTIFF. Defaults to "EPSG:3857".
        band (int, optional): The band to use. Band indexing starts at 1. Defaults to None.
        vmin (float, optional): The minimum value to use when colormapping the palette when plotting a single band. Defaults to None.
        vmax (float, optional): The maximum value to use when colormapping the palette when plotting a single band. Defaults to None.
        nodata (float, optional): The value from the band to use to interpret as not valid data. Defaults to None.
        attribution (str, optional): Attribution for the source raster. This defaults to a message about it being a local file.. Defaults to None.
        layer_name (str, optional): The layer name to use. Defaults to None.
        return_client (bool, optional): If True, the tile client will be returned. Defaults to False.

    Returns:
        ipyleaflet.TileLayer | folium.TileLayer: An ipyleaflet.TileLayer or folium.TileLayer.
    """

    # ... and suppress errors
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    if "max_zoom" not in kwargs:
        kwargs["max_zoom"] = 100
    if "max_native_zoom" not in kwargs:
        kwargs["max_native_zoom"] = 100

    if "show_loading" not in kwargs:
        kwargs["show_loading"] = False

    if isinstance(source, str):
        source = os.path.abspath(source)

    elif isinstance(source, TileClient) or isinstance(
        source, rasterio.io.DatasetReader
    ):
        pass

    else:
        raise ValueError("The source must either be a string or TileClient")

    if isinstance(source, str) or isinstance(source, rasterio.io.DatasetReader):
        tile_client = TileClient(source, port=port, debug=debug)

    else:
        tile_client = source

    tile_layer = get_leaflet_tile_layer(
        tile_client,
        port=port,
        debug=debug,
        projection=projection,
        band=band,
        vmin=vmin,
        vmax=vmax,
        nodata=nodata,
        attribution=attribution,
        name=layer_name,
        **kwargs,
    )

    if return_client:
        return tile_layer, tile_client
    else:
        return tile_layer


def zoom_to_bounds(m, bounds):
    """Zooms to a bounding box in the form of [minx, miny, maxx, maxy].

    Args:
        bounds (list | tuple): A list/tuple containing minx, miny, maxx, maxy values for the bounds.
    """
    #  The ipyleaflet fit_bounds method takes lat/lon bounds in the form [[south, west], [north, east]].
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])


def time_slider(
    m,
    years: List,
    layers: Optional[Union[Dict, List, str]] = None,
    labels: Optional[List] = None,
    position: Optional[str] = "bottomright",
    slider_length: Optional[str] = "150px",
    zoom_to_layer: Optional[bool] = False,
    **kwargs,
):
    """Adds a time slider to the map.

    Args:
        years: List: The list of years for each tile time interval.
        layers (dict, optional): The dictionary containing a set of XYZ tile layers.
        labels (list, optional): The list of labels to be used for the time series.
        Defaults to None.
        time_interval (int, optional): Time interval in seconds. Defaults to 1.
        position (str, optional): Position to place the time slider, can
        be any of ['topleft', 'topright', 'bottomleft', 'bottomright']. Defaults to
        "bottomright".
        slider_length (str, optional): Length of the time slider. Defaults to "150px".
        zoom_to_layer (bool, optional): Whether to zoom to the extent of the layer.
        Defaults to False.

    """

    bounds = None

    if isinstance(layers, list):
        if zoom_to_layer:
            layer0 = layers[0]
            _, tile_client = get_local_tile_layer(
                layer0,
                return_client=True,
            )

            bounds = tile_client.bounds()  # [ymin, ymax, xmin, xmax]
            bounds = (
                bounds[2],
                bounds[0],
                bounds[3],
                bounds[1],
            )  # [minx, miny, maxx, maxy]
            zoom_to_bounds(m, bounds)

        layers = images_to_tiles(layers, names=labels, **kwargs)

    if not isinstance(layers, dict):
        raise TypeError("The layers must be a dictionary.")

    if labels is None:
        labels = list(layers.keys())
    if len(labels) != len(layers):
        raise ValueError("The length of labels is not equal to that of layers.")

    slider = widgets.IntSlider(
        min=min(years),
        max=max(years),
        readout=False,
        continuous_update=False,
        layout=widgets.Layout(width=slider_length),
    )
    label = widgets.Label(
        value=labels[0], layout=widgets.Layout(padding="0px 5px 0px 5px")
    )

    slider_widget = widgets.HBox([label, slider])

    keys = list(layers.keys())
    layer = layers[keys[0]]
    m.add(layer)

    def slider_changed(change):
        m.default_style = {"cursor": "wait"}
        index = slider.value - min(years)
        print(f"Slider value: {slider.value}")
        print(f"Layer index: {index}")
        label.value = labels[index]
        layer.url = layers[label.value].url
        layer.name = layers[label.value].name
        m.default_style = {"cursor": "default"}

    slider.observe(slider_changed, "value")

    slider_ctrl = ipyleaflet.WidgetControl(
        widget=slider_widget, position=position
    )
    m.add(slider_ctrl)
    m.slider_ctrl = slider_ctrl


def build_url(date_value: str) -> str:
    pass


def date_picker(
    m,
    **kwargs,
):
    """Adds a time slider to the map."""

    datepicker = widgets.DatePicker(
        readout=False,
        continuous_update=False,
    )

    slider_widget = widgets.HBox([datepicker])

    keys = list(layers.keys())
    layer = layers[keys[0]]
    m.add(layer)

    def dp_changed(change):
        m.default_style = {"cursor": "wait"}
        date = datepicker.value
        print(f"Slider value: {datepicker.value}")
        # label.value = labels[index]
        # layer.url = layers[label.value].url
        # layer.name = layers[label.value].name
        m.default_style = {"cursor": "default"}

    datepicker.observe(dp_changed, "value")

    datepicker_ctrl = ipyleaflet.WidgetControl(
        widget=datepicker, position="bottomleft"
    )
    m.add(datepicker_ctrl)
    m.datepicker_ctrl = datepicker_ctrl


def plotTS(m, marker, datarray, y, x):
    image = BytesIO()
    datarray.sel(x=x, y=y, method="nearest").plot()
    plt.savefig(image, format="png", bbox_inches="tight")
    data = "data:image/png;base64," + (encodebytes(image.getvalue())).decode(
        "ascii"
    )
    plt.close()
    message = widgets.HTML()
    message.value = (
        '<div><img height="350" width="450" src="{0}"/></div>'.format(data)
    )
    popup = ipyleaflet.Popup(location=marker.location, child=message)
    m.add_layer(popup)


def render(tile: str) -> None:
    raw_hdf_layers = get_mod44w_time_series(tile)

    mod44w_time_series_dataset = merge_water_masks(raw_hdf_layers)

    years = mod44w_time_series_dataset.time.data.astype("datetime64[Y]")

    projection, transform = get_geo_info(mod44w_time_series_dataset)

    years = [int(str(year)) for year in years]

    projected_layers = []
    files_to_delete = []
    for time_step in mod44w_time_series_dataset[WATER_MASK_KEY].time:
        layer_for_ts = mod44w_time_series_dataset[WATER_MASK_KEY].sel(
            time=time_step
        )
        year = time_step.data.astype("datetime64[Y]")
        projected_layer = open_and_write_temp(
            layer_for_ts,
            transform,
            projection,
            year,
            tile,
            files_to_rm=files_to_delete,
        )
        projected_layers.append(projected_layer)

    center = [38.128, 2.588]
    zoom = 5

    def onMapInteractionCallback(*args, **kwargs):
        # Many events happen that call this, we only want click events
        if kwargs["type"] == "click":
            lat, lon = kwargs["coordinates"]

            # Update the marker position
            marker.location = (lat, lon)

            # Triggers an event that updates the time-series plots
            print(lat)
            print(lon)
            y, x = latlon_to_projection(lat, lon)
            plotTS(m, marker, mod44w_time_series_dataset["water_mask"], x, y)

    marker = ipyleaflet.Marker(location=(0, 0))

    m = ipyleaflet.Map(
        center=center,
        zoom=zoom,
        basemap=ipyleaflet.basemaps.Esri.WorldImagery,
        scroll_wheel_zoom=True,
        keyboard=True,
        layout=widgets.Layout(height="600px"),
    )

    m.add_control(ipyleaflet.LayersControl())
    m.add(ipyleaflet.FullScreenControl())
    m.on_interaction(onMapInteractionCallback)
    m.add_layer(marker)

    time_slider(
        m,
        years=years,
        layers=projected_layers,
        zoom_to_layer=True,
        cmap=["#194d33", "#8ed1fc"],
    )

    # cleanup(files_to_delete)

    return m


def cleanup(files_to_rm: list) -> None:
    for path_to_delete in files_to_rm:
        if os.path.exists(path_to_delete):
            os.remove(path_to_delete)


if __name__ == "__main__":
    # Example usage:
    input_numbers = set(range(2001, 2022))
    min_year = min(input_numbers)
    number_mapping = set([year - min_year for year in input_numbers])

    print("Input Numbers:", input_numbers)
    print("Number Mapping:", number_mapping)

    print(get_mod44w_time_series("h09v05"))
    print(get_mod44w_time_series("h30v11"))

    layers = get_mod44w_time_series("h09v05")
    #  layers = images_to_tiles(layers,)
    time_series = merge_water_masks(layers)
    print(time_series)

    print(get_geo_info(time_series))
