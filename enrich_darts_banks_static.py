"""Enrichment process of current darts dataset with ArcticDEM data."""

import gc
import logging
import multiprocessing as mp
from collections import defaultdict
from math import ceil
from typing import Literal

import geopandas as gpd
import numpy as np
import odc.geo.geom
import odc.geo.xr
import smart_geocubes
import xarray as xr
from odc.geo.geobox import GeoBox
from rich import pretty, traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from stopuhr import FunkUhr, funkuhr
from xrspatial import convolution, slope
from xrspatial.utils import has_cuda_and_cupy

if has_cuda_and_cupy():
    import cupy as cp
    import cupy_xarray  # noqa: F401

console = Console()
logging.getLogger("smart_geocubes").setLevel(logging.DEBUG)
logging.getLogger("smart_geocubes").addHandler(RichHandler(console=console))
pretty.install()
traceback.install(show_locals=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(RichHandler(console=console))

funker = FunkUhr(printer=logger.info)

YEAR = 2023
# REGION = "Sakha (Yakutia)"
# REGION = "Northwest Territories"


def free_cupy():
    """Free the CUDA memory of cupy."""
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if cp is not None:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


@funkuhr("Calculate TPI", printer=logger.info)
def calculate_topographic_position_index(arcticdem_ds: xr.Dataset, outer_radius: int, inner_radius: int) -> xr.Dataset:
    """Calculate the Topographic Position Index (TPI) from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        outer_radius (int, optional): The outer radius of the annulus kernel in m.
        inner_radius (int, optional): The inner radius of the annulus kernel in m.

    Returns:
        xr.Dataset: The input Dataset with the calculated TPI added as a new variable 'tpi'.

    """
    cellsize_x, cellsize_y = convolution.calc_cellsize(arcticdem_ds.dem)  # Should be equal to the resolution of the DEM
    # Use an annulus kernel if inner_radius is greater than 0
    outer_radius_m = f"{outer_radius}m"
    outer_radius_px = f"{ceil(outer_radius / cellsize_x)}px"
    if inner_radius > 0:
        inner_radius_m = f"{inner_radius}m"
        inner_radius_px = f"{ceil(inner_radius / cellsize_x)}px"
        kernel = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_radius_m, inner_radius_m)
        attr_cell_description = (
            f"within a ring at a distance of {inner_radius_px}-{outer_radius_px} cells "
            f"({inner_radius_m}-{outer_radius_m}) away from the focal cell."
        )
        logger.debug(
            f"Calculating Topographic Position Index with annulus kernel of "
            f"{inner_radius_px}-{outer_radius_px} ({inner_radius_m}-{outer_radius_m}) cells."
        )
    else:
        kernel = convolution.circle_kernel(cellsize_x, cellsize_y, outer_radius_m)
        attr_cell_description = (
            f"within a circle at a distance of {outer_radius_px} cells ({outer_radius_m}) away from the focal cell."
        )
        logger.debug(
            f"Calculating Topographic Position Index with circle kernel of {outer_radius_px} ({outer_radius_m}) cells."
        )

    if has_cuda_and_cupy() and arcticdem_ds.cupy.is_cupy:
        kernel = cp.asarray(kernel)

    tpi = arcticdem_ds.dem - convolution.convolution_2d(arcticdem_ds.dem, kernel) / kernel.sum()
    tpi.attrs = {
        "long_name": "Topographic Position Index",
        "units": "m",
        "description": "The difference between the elevation of a cell and the mean elevation of the surrounding"
        f"cells {attr_cell_description}",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }

    arcticdem_ds["tpi"] = tpi.compute()

    return arcticdem_ds


@funkuhr("Calculate Slope", printer=logger.info)
def calculate_slope(arcticdem_ds: xr.Dataset) -> xr.Dataset:
    """Calculate the slope of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.

    Returns:
        xr.Dataset: The input Dataset with the calculated slope added as a new variable 'slope'.

    """
    logger.debug("Calculating slope of the terrain surface.")

    slope_deg = slope(arcticdem_ds.dem)
    slope_deg.attrs = {
        "long_name": "Slope",
        "units": "degrees",
        "description": "The slope of the terrain surface in degrees.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }
    arcticdem_ds["slope"] = slope_deg.compute()

    return arcticdem_ds


@funkuhr("Preprocess ArcticDEM", printer=logger.info)
def preprocess_legacy_arcticdem_fast(
    ds_arcticdem: xr.Dataset, tpi_outer_radius: int, tpi_inner_radius: int, device: Literal["cuda", "cpu"] | int
):
    """Preprocess the ArcticDEM data with legacy (DARTS v1) preprocessing steps.

    Args:
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        tpi_outer_radius (int): The outer radius of the annulus kernel for the tpi calculation in number of cells.
        tpi_inner_radius (int): The inner radius of the annulus kernel for the tpi calculation in number of cells.
        device (Literal["cuda", "cpu"] | int): The device to run the tpi and slope calculations on.
            If "cuda" take the first device (0), if int take the specified device.

    Returns:
        xr.Dataset: The preprocessed ArcticDEM dataset.

    """
    use_gpu = device == "cuda" or isinstance(device, int)

    # Warn user if use_gpu is set but no GPU is available
    if use_gpu and not has_cuda_and_cupy():
        logger.warning(
            f"Device was set to {device}, but GPU acceleration is not available. Calculating TPI and slope on CPU."
        )
        use_gpu = False

    # Calculate TPI and slope from ArcticDEM on GPU
    if use_gpu:
        device_nr = device if isinstance(device, int) else 0
        logger.debug(f"Moving arcticdem to GPU:{device}.")
        # Check if dem is dask, if not persist it, since tpi and slope can't be calculated from cupy-dask arrays
        if ds_arcticdem.chunks is not None:
            ds_arcticdem = ds_arcticdem.persist()
        # Move and calculate on specified device
        with cp.cuda.Device(device_nr):
            ds_arcticdem = ds_arcticdem.cupy.as_cupy()
            ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
            ds_arcticdem = calculate_slope(ds_arcticdem)
            ds_arcticdem = ds_arcticdem.cupy.as_numpy()
            free_cupy()

    # Calculate TPI and slope from ArcticDEM on CPU
    else:
        ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
        ds_arcticdem = calculate_slope(ds_arcticdem)

    return ds_arcticdem


class FeatureStats:
    def __init__(self, feat: xr.DataArray):
        self.avg_elevation = feat.dem.mean().item()
        self.avg_slope = np.abs(feat.slope).mean().item()
        self.avg_tpi = np.abs(feat.tpi).mean().item()

        self.max_elevation = feat.dem.max().item()
        self.min_elevation = feat.dem.min().item()
        self.max_slope = feat.slope.max().item()
        self.min_slope = feat.slope.min().item()
        self.max_tpi = feat.tpi.max().item()
        self.min_tpi = feat.tpi.min().item()

        self.std_elevation = feat.dem.std().item()
        self.std_slope = feat.slope.std().item()
        self.std_tpi = feat.tpi.std().item()

        self.count = feat.dem.count().item()
        self.count_nan = feat.dem.isnull().sum().item()
        self.count_valid = self.count - self.count_nan
        if self.count > 0:
            self.count_valid_ratio = self.count_valid / self.count
        else:
            self.count_valid_ratio = 0

    def __to_dict__(self):
        return {
            "avg_elevation": self.avg_elevation,
            "avg_slope": self.avg_slope,
            "avg_tpi": self.avg_tpi,
            "max_elevation": self.max_elevation,
            "min_elevation": self.min_elevation,
            "max_slope": self.max_slope,
            "min_slope": self.min_slope,
            "max_tpi": self.max_tpi,
            "min_tpi": self.min_tpi,
            "std_elevation": self.std_elevation,
            "std_slope": self.std_slope,
            "std_tpi": self.std_tpi,
            "count": self.count,
            "count_nan": self.count_nan,
            "count_valid": self.count_valid,
            "count_valid_ratio": self.count_valid_ratio,
        }


@funker("Process feature", log=False)
def _process_feature(feat, adem):
    geom = odc.geo.geom.Geometry(feat.geometry, crs="EPSG:3413")
    geobox = GeoBox.from_geopolygon(geom, resolution=smart_geocubes.ArcticDEM2m.extent.resolution, crs="EPSG:3413")
    adem_aoi = adem.odc.crop(geobox.extent, apply_mask=False)
    rasterized = odc.geo.xr.rasterize(geom, adem_aoi.odc.geobox)
    adem_feat = adem_aoi.where(rasterized)
    stats = FeatureStats(adem_feat)
    return feat["id"], stats


if __name__ == "__main__":
    mp.set_start_method("forkserver")

    v1features = gpd.read_file("./data/DARTS_NitzeEtAl_v1_features_2018_2023_level2.gpkg")
    logger.info(f"Number of features: {len(v1features)}")

    # Filter for 2023
    v1features = v1features[v1features["year"] == YEAR].reset_index(drop=True)
    logger.info(f"Number of features in {YEAR}: {len(v1features)}")

    # Filter for Banks Island
    canadian_terretories = gpd.read_file("./data/lpr_000b16a_e/lpr_000b16a_e.shp")
    nwt = canadian_terretories[canadian_terretories.PREABBR == "N.W.T."]
    banks_island = nwt.explode().reset_index()[2:3]
    # First enrich the v1features with the admin1 information
    v1features = (
        gpd.sjoin(v1features, banks_island.to_crs(v1features.crs)[["geometry"]], how="left")
        .reset_index(drop=True)
        .dropna(subset=["index_right"])
    )
    # Now filter for the region
    logger.info(f"Number of features in {YEAR} and Banks: {len(v1features)}")

    v1features = v1features.to_crs("EPSG:3413")

    accessor = smart_geocubes.ArcticDEM32m("./data/arcticdem32m.icechunk")
    try:
        accessor.create()
    except FileExistsError:
        pass

    v1features_geobox = GeoBox.from_bbox(
        v1features.total_bounds, resolution=accessor.extent.resolution, crs="EPSG:3413"
    )
    adem = accessor.load(v1features_geobox, concurrency_mode="blocking", persist=True, buffer=320)
    adem = preprocess_legacy_arcticdem_fast(adem, 100, 30, device="cuda")

    progress = Progress(
        SpinnerColumn("bouncingBall"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    allstats = defaultdict(dict)
    with progress:
        task = progress.add_task("Processing features", total=len(v1features))
        for idx, feat in v1features.iterrows():
            feature_id, stats = _process_feature(feat, adem)
            stats_dict = stats.__to_dict__()
            for key, value in stats_dict.items():
                allstats[key][feature_id] = value

            progress.update(task, advance=1)
        logger.debug("All features processed")

        # with ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(_process_feature, feat, adem) for idx, feat in v1features.iterrows()]
        #     for future in as_completed(futures):
        #         feature_id, elevation = future.result()
        #         if elevation is not None:
        #             avg_elevations[feature_id] = elevation
        #         progress.update(task, advance=1)
        #     logger.debug("All features processed")

    funker.summary(res=4)
    for key, value in allstats.items():
        v1features[key] = v1features["id"].map(value)

    v1features.to_file(f"./data/DARTS_NitzeEtAl_v1_features_{YEAR}_BANKS_level2_enriched.gpkg", driver="GPKG")
