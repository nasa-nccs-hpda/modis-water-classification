from typing import Optional, Union, List, Dict
import ipywidgets as widgets
from osgeo import gdal
import rasterio
import ipyleaflet

from localtileserver import (
    get_leaflet_tile_layer,
    TileClient,
)
import os
from io import BytesIO
from base64 import encodebytes
import datetime
import pathlib
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt

from utils import (
    latlon_to_projection,
    open_and_write_temp,
    get_geo_info,
    zoom_to_bounds,
)


NUMBER_YEARS: int = (
    23  # number of files a single tile's time-series should match
)
DATE_SPLIT_KEY: str = "MOD44W.A"
WATER_MASK_KEY: str = "water_mask"
CSS_BASE_PATH: str = "/css/modis/Collection6.1/L3/MOD44W-LandWaterMask"

MODIS_TILE_URLS: dict = {
    # 'MOD09GA-143': 'MODIS_Terra_SurfaceReflectance_Bands143',
    # 'MOD09GQ-121': 'MODIS_Terra_SurfaceReflectance_Bands121',
    # 'MOD09GA-721': 'MODIS_Terra_SurfaceReflectance_Bands721',
    'MOD09A1-143': 'MODIS_Terra_L3_SurfaceReflectance_Bands143_8Day',
    'MOD09Q1-121': 'MODIS_Terra_L3_SurfaceReflectance_Bands121_8Day',
    'MOD09A1-721': 'MODIS_Terra_L3_SurfaceReflectance_Bands721_8Day',
}

GIBS_URL_PRE_STR: str = 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best'
GIBS_URL_POST_STR_500m: str = 'GoogleMapsCompatible_Level8/{z}/{y}/{x}.png'
GIBS_URL_POST_STR_250m: str = 'GoogleMapsCompatible_Level9/{z}/{y}/{x}.png'

# ----------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------
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
            plotTS(m,
                   marker,
                   mod44w_time_series_dataset["water_mask"],
                   x, y,
                   lat, lon)
    
    def onMarkerClickCallback(*args, **kwargs):
        lat, lon = marker.location
        print(lat)
        print(lon)
        y, x = latlon_to_projection(lat, lon)
        plotTS(m,
               marker,
               mod44w_time_series_dataset["water_mask"],
               x, y,
               lat, lon) 

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
    marker.on_click(onMarkerClickCallback) 

    date_picker(m,
                layers=[])

    time_slider(
        m,
        years=years,
        layers=projected_layers,
        zoom_to_layer=True,
        cmap=["#194d33", "#8ed1fc"],
    )

    return m, files_to_delete


# ----------------------------------------------------------------------------
# get_mod44w_time_series
# ----------------------------------------------------------------------------
def get_mod44w_time_series(tile: str) -> List[pathlib.Path]:
    base_dir = pathlib.Path(CSS_BASE_PATH)
    regex = f"*/*/MOD44W.A*.{tile}.061.*.hdf"
    files_matching_regex = sorted(
        [str(file_match) for file_match in base_dir.glob(regex)]
    )
    n_files_matching_regex = len(files_matching_regex)
    assert n_files_matching_regex >= NUMBER_YEARS, (
        f"Expected {NUMBER_YEARS} files matching {regex},"
        + f" got {n_files_matching_regex}"
    )
    return files_matching_regex


# ----------------------------------------------------------------------------
# merge_water_masks
# ----------------------------------------------------------------------------
def merge_water_masks(layers: List[str]) -> xr.Dataset:
    datasets = []
    for file_path in layers:
        ds = rxr.open_rasterio(file_path)
        time_str = file_path.split(DATE_SPLIT_KEY)[-1][:4]
        time_format = "%Y"
        time = datetime.datetime.strptime(time_str, time_format)
        ds["time"] = time
        datasets.append(ds)
    merged_dataset = xr.concat(datasets, dim="time")
    merged_dataset = merged_dataset.sortby("time")
    return merged_dataset


# ----------------------------------------------------------------------------
# plotTS
# ----------------------------------------------------------------------------
def plotTS(m, marker, datarray, y, x, lat, lon):
    image = BytesIO()
    xticks = list(datarray.time.values)
    print(xticks)
    datarray.sel(x=x, y=y, method="nearest").plot(figsize=(10,5), xticks=xticks)
    plt.grid()
    plt.savefig(image, format="png", bbox_inches="tight")
    data = "data:image/png;base64," + (encodebytes(image.getvalue())).decode(
        "ascii"
    )
    plt.close()
    message = widgets.HTML()
    image_html = '<div><img height="350" width="550" src="{0}"/>'.format(data)
    legend_text = f'<b>Legend:</b> 1 = water, 0 = not water</b>'
    location_text = f'<br><b>Location:</b><br>lat/lon: ({lat}, {lon})' + \
        f'<br>modis sinu x/y: ({y},{x})'
    location_html ='</div><div><p>{0}{1}</p></div>'.format(legend_text,
                                                           location_text)
    message.value = (
        image_html + location_html
    )
    popup = ipyleaflet.Popup(location=marker.location, child=message)
    m.add_layer(popup)


# ----------------------------------------------------------------------------
# time_slider
# ----------------------------------------------------------------------------
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
        raise ValueError(
            "The length of labels is not equal to that of layers."
            )
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


# ----------------------------------------------------------------------------
# images_to_tiles
# ----------------------------------------------------------------------------
def images_to_tiles(
    images: Union[str, List[str]], names: List[str] = None, **kwargs
) -> Dict[str, ipyleaflet.TileLayer]:

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


# ----------------------------------------------------------------------------
# get_local_tile_layer
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# get_tile_layer
# ----------------------------------------------------------------------------
def get_tile_layer(source, layer_key, name=''):
    max_zoom = 22
    if '121' in layer_key:
        max_native_zoom = 9
    else:
        max_native_zoom = 8
    return ipyleaflet.TileLayer(url=source,
                                name=name,
                                max_zoom=max_zoom,
                                max_native_zoom=max_native_zoom)


# ----------------------------------------------------------------------------
# build_url
# ----------------------------------------------------------------------------
def build_url(layer_key: str, date_value: str) -> str:
    layer = MODIS_TILE_URLS[layer_key]
    if '121' in layer_key:
        url = os.path.join(
            GIBS_URL_PRE_STR, layer, 'default',
            date_value, GIBS_URL_POST_STR_250m)
    else:
       url = os.path.join(
           GIBS_URL_PRE_STR, layer, 'default',
           date_value, GIBS_URL_POST_STR_500m) 
    return url 


# ----------------------------------------------------------------------------
# get_modis_gibs_layers
# ----------------------------------------------------------------------------
def get_modis_gibs_layers(layer_url: str, layer_name: str):
    return get_tile_layer(layer_url, layer_key=layer_name,
                          name=layer_name)


# ----------------------------------------------------------------------------
# date_picker
# ----------------------------------------------------------------------------
def date_picker(
    m,
    layers,
    **kwargs,
):
    datepicker = widgets.DatePicker(
        readout=False,
        continuous_update=False,
        value=datetime.date(2001, 1, 1)
    )

    dp_widget = widgets.HBox([datepicker])

    modis_tiles_urls = {} 
    layers = {} 
    for layer_key in MODIS_TILE_URLS.keys():
        layer_url = build_url(layer_key, str(datepicker.value))
        modis_tiles_urls[layer_key] = layer_url
        layers[layer_key] = get_modis_gibs_layers(layer_url, layer_key)
    
    for layer_key in layers.keys():
        m.add_layer(layers[layer_key])

    def dp_changed(change):
        m.default_style = {"cursor": "wait"}
        date = datepicker.value
        print(f"Datepicker value: {datepicker.value}")
        for layer_key in layers.keys():
            layer = layers[layer_key]
            layer.url = build_url(layer_key, str(date))
            layer.name = layer_key
        m.default_style = {"cursor": "default"}

    datepicker.observe(dp_changed, "value")

    datepicker_ctrl = ipyleaflet.WidgetControl(
        widget=dp_widget, position="bottomleft"
    )
    m.add(datepicker_ctrl)
    m.datepicker_ctrl = datepicker_ctrl
