try:
    import geopandas
    import mgwr
    import libpysal
    import matplotlib
    import seaborn
    print("All libraries available")
except ImportError as e:
    print(f"Missing library: {e.name}")
