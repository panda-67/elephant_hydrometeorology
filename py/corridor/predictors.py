from src.corridor.predictors import CorridorPredictors


def main():
    print("=" * 60)
    print("CORRIDOR PREDICTORS")
    print("=" * 60)

    corridor = CorridorPredictors()

    # ------------------------------------------------------------
    # GEE
    # ------------------------------------------------------------
    corridor.initialize()

    # ------------------------------------------------------------
    # P5 Corridor Boundary
    # ------------------------------------------------------------
    corridor.load_aoi()

    # ------------------------------------------------------------
    # P6.1 Elevation & P6.2 Slope
    # ------------------------------------------------------------
    corridor.initialize_terrain()
    corridor.export_elevation()
    corridor.export_slope()

    # ------------------------------------------------------------
    # P6.3 Land Cover
    # ------------------------------------------------------------
    corridor.initialize_landcover()
    corridor.export_landcover()

    # ------------------------------------------------------------
    # P6.4 NDVI
    # ------------------------------------------------------------
    corridor.initialize_ndvi()
    corridor.export_ndvi()

    # ------------------------------------------------------------
    # P6.5 Distance to Water
    # ------------------------------------------------------------
    corridor.initialize_water()
    corridor.export_distance_to_water()

    print("\nCorridor predictors completed.")


if __name__ == "__main__":
    main()
