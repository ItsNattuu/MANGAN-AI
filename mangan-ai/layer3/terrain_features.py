import rasterio


class TerrainFeatureEngine:

    def __init__(self, dem_path):

        self.dem_path = dem_path

    def get_terrain_features(
        self,
        lon,
        lat
    ):

        with rasterio.open(
            self.dem_path
        ) as src:

            for value in src.sample(
                [(lon, lat)]
            ):

                elevation = float(
                    value[0]
                )

        # Temporary mock slope.
        # Replace with real DEM-derived
        # slope calculation.

        slope = 8.0

        return {
            "elevation": elevation,
            "slope_deg": slope
        }
