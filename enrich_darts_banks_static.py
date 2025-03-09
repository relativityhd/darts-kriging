"""Enrichment process of current darts dataset with ArcticDEM data."""

import logging
import multiprocessing as mp

import geopandas as gpd
import odc.geo.geom
import odc.geo.xr
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
from smart_geocubes.datasets.arcticdem import ArcticDEM32m

console = Console()
logging.getLogger("smart_geocubes").setLevel(logging.DEBUG)
logging.getLogger("smart_geocubes").addHandler(RichHandler(console=console))
pretty.install()
traceback.install(show_locals=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(RichHandler(console=console))

YEAR = 2023
# REGION = "Sakha (Yakutia)"
# REGION = "Northwest Territories"


def _process_feature(feat, adem):
    geom = odc.geo.geom.Geometry(feat.geometry, crs="EPSG:3413")
    geobox = GeoBox.from_geopolygon(geom, resolution=ArcticDEM32m.extent.resolution, crs="EPSG:3413")
    adem_aoi = adem.odc.crop(geobox.extent, apply_mask=False)
    rasterized = odc.geo.xr.rasterize(geom, adem_aoi.odc.geobox)
    adem_feat = adem_aoi.where(rasterized)
    avg_elevation = adem_feat.dem.mean().item()
    return feat["id"], avg_elevation


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

    accessor = ArcticDEM32m("./data/arcticdem32m.icechunk")
    try:
        accessor.create()
    except FileExistsError:
        pass

    v1features_geobox = GeoBox.from_bbox(
        v1features.total_bounds, resolution=ArcticDEM32m.extent.resolution, crs="EPSG:3413"
    )
    adem = accessor.load(v1features_geobox, concurrency_mode="threading", persist=True)

    progress = Progress(
        SpinnerColumn("bouncingBall"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    avg_elevations = {}
    with progress:
        task = progress.add_task("Processing features", total=len(v1features))
        for idx, feat in v1features.iterrows():
            feature_id, elevation = _process_feature(feat, adem)
            if elevation is not None:
                avg_elevations[feature_id] = elevation
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

    logger.debug(f"Number of enriched features: {len(avg_elevations)}")
    v1features["avg_elevation"] = v1features["id"].map(avg_elevations)

    v1features.to_file(f"./data/DARTS_NitzeEtAl_v1_features_{YEAR}_BANKS_level2_enriched.gpkg", driver="GPKG")
