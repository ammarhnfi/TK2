import geopandas as gpd
import pandas as pd
import os

def convert_geojson_to_excel():
    input_file = "d:/tk2/5_Our/rumah_27_8_25_clean.geojson"
    output_file = "d:/tk2/5_Our/rumah_27_8_25_clean.xlsx"
    
    print(f"[INFO] Membaca {input_file} ...")
    try:
        # Gunakan Geopandas untuk membaca GeoJSON
        gdf = gpd.read_file(input_file)
        
        # Ekstrak Latitude & Longitude dari kolom geometry untuk kemudahan di Excel
        gdf['Longitude'] = gdf.geometry.x
        gdf['Latitude'] = gdf.geometry.y
        
        # Hapus kolom geometry karena Excel tidak mendukung objek spasial secara langsung
        df = pd.DataFrame(gdf.drop(columns=['geometry']))
        
        print("[INFO] Mengekspor ke format Excel (.xlsx)... Tahan sebentar.")
        df.to_excel(output_file, index=False, engine='openpyxl')
        
        print(f"[SELESAI] Data berhasil dikonversi dan disimpan ke: {output_file}")
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan: {e}")

if __name__ == "__main__":
    convert_geojson_to_excel()
