import geopandas as gpd
from shapely.geometry import Point


class LandConstraintEngine:
    def __init__(self, protected_areas_path, crs="EPSG:4326"):
        self.protected_areas = gpd.read_file(protected_areas_path).to_crs(crs)
        self.crs = crs

    def is_restricted(self, lon, lat):
        """Returns True if the point falls inside any protected/restricted zone."""
        point = Point(lon, lat)
        match = self.protected_areas[self.protected_areas.contains(point)]
        return len(match) > 0

    def get_constraint_features(self, lon, lat):
        restricted = self.is_restricted(lon, lat)
        return {
            "is_restricted_area": int(restricted),
            "land_eligible": int(not restricted),
        }


if __name__ == "__main__":
    engine = LandConstraintEngine(protected_areas_path="data/protected_areas.geojson")
    print(engine.get_constraint_features(lon=83.5, lat=21.2))
