import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
import base64
import os

# Sample data for demonstration
provinces = pd.DataFrame({
    "province": ["CaoBang", "HaGiang", "QuangNinh"],
    "latitude": [22.666, 22.823, 21.117],
    "longitude": [106.25, 104.98, 107.25],
    "audio_file": [
        "applied_tool/data/audio/North_CaoBang_spk_11_0001_11_0001.wav",
        "applied_tool/data/audio/North_HaGiang_spk_23_0001_23_0001.wav",
        "applied_tool/data/audio/North_QuangNinh_spk_14_0001_14_0001.wav"
    ]
})

geometry = [Point(xy) for xy in zip(provinces["longitude"], provinces["latitude"])]
geo_df = gpd.GeoDataFrame(provinces, geometry=geometry)

# Create a folium map centered around Vietnam
m = folium.Map(location=[16.0, 107.5], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

# Add audio tooltips to the map
for idx, row in geo_df.iterrows():
    if os.path.exists(row["audio_file"]):
        encoded = base64.b64encode(open(row["audio_file"], 'rb').read()).decode()
        audio_html = f"""
        <audio controls>
            <source src="data:audio/wav;base64,{encoded}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        """
        popup = folium.Popup(audio_html, max_width=300)
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup,
            tooltip=row["province"]
        ).add_to(marker_cluster)

m.save("vietnam_audio_map.html")