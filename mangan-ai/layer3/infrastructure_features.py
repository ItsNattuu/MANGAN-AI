import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


class InfrastructureFeatureEngine:
    def __init__(
        self,
        roads_path,
        railways_path,
        power_lines_path,
        existing_mines_csv,
        processing_facilities_csv,
        metric_crs="EPSG:32644",  # match whatever UTM zone Layer 2 uses
    ):
        self.metric_crs = metric_crs

        self.roads_m = gpd.read_file(roads_path).to_crs(metric_crs)
        self.railways_m = gpd.read_file(railways_path).to_crs(metric_crs)
        self.power_lines_m = gpd.read_file(power_lines_path).to_crs(metric_crs)
        self.existing_mines_m = self._points_from_csv(existing_mines_csv, metric_crs)
        self.processing_facilities_m = self._points_from_csv(processing_facilities_csv, metric_crs)

    @staticmethod
    def _points_from_csv(csv_path, metric_crs):
        df = pd.read_csv(csv_path)
        geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        return gdf.to_crs(metric_crs)

    def _nearest_distance_m(self, point_m, gdf_m):
        if len(gdf_m) == 0:
            return float("nan")
        return gdf_m.distance(point_m).min()

    def get_infrastructure_features(self, lon, lat):
        """Returns distances (in metres) to each infrastructure type."""
        point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(self.metric_crs)
        point_m = point_gdf.geometry.iloc[0]

        return {
            "distance_to_road_m": self._nearest_distance_m(point_m, self.roads_m),
            "distance_to_railway_m": self._nearest_distance_m(point_m, self.railways_m),
            "distance_to_power_m": self._nearest_distance_m(point_m, self.power_lines_m),
            "distance_to_existing_mine_m": self._nearest_distance_m(point_m, self.existing_mines_m),
            "distance_to_processing_facility_m": self._nearest_distance_m(point_m, self.processing_facilities_m),
        }


if __name__ == "__main__":
    engine = InfrastructureFeatureEngine(
        roads_path= "data/roads.geojson",
        railways_path="data/railways.geojson",
        power_lines_path="data/power_lines.geojson",
        existing_mines_csv="data/existing_mines.csv",
        processing_facilities_csv="data/processing_facilities.csv",
    )
    print(engine.get_infrastructure_features(lon=83.5, lat=21.2))
