# ==============================================================================================================================
# File:        [Curvature Radius Calculator.py]
# Author:      [Mihai-Serban Morar]
# Student ID:  [001235169]
# Date:        [15/04/2025]
# Description: The calculate_curve_radii.py script processes London Underground track data to compute the 
# curvature of rail segments near specific stations. It begins by loading station names from a noise and speed dataset, 
# then uses OpenStreetMap (via OSMnx) to download the subway track geometry for London. 
# The script calculates the curve radius along the tracks using a 3-point geometric method and associates each curved segment 
# with its nearest station. It filters the results to include only the stations from the dataset, then aggregates the curvature 
# values—providing average, minimum, and maximum radius per station—and saves the final results to a CSV file for use in 
# machine learning models or analysis.

# =============================================================================================================================


import pandas as pd
import osmnx as ox
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
import webbrowser
import os


# === CONFIG ===
CITY = "London, UK"
INPUT_CSV = "underground_noise_dataset.csv"
OUTPUT_CSV = "Station_curve_radii.csv"

# === Load station pairs ===
df = pd.read_csv(INPUT_CSV)
station_names = pd.unique(df[['From', 'To']].values.ravel())

# === Download OSM Subway Rail Data ===
tags = {'railway': 'subway'}
rail = ox.graph_from_place(CITY, custom_filter='["railway"="subway"]', network_type='all_private')
edges = ox.graph_to_gdfs(rail, nodes=False)

# === Function to calculate radius from 3 points ===
def radius_from_3_points(p1, p2, p3):
    a = np.linalg.norm(np.array(p2) - np.array(p1))
    b = np.linalg.norm(np.array(p3) - np.array(p2))
    c = np.linalg.norm(np.array(p3) - np.array(p1))
    s = (a + b + c) / 2
    area = max(np.sqrt(abs(s * (s - a) * (s - b) * (s - c))), 1e-6)
    radius = (a * b * c) / (4.0 * area)
    return radius

# === Calculate radii ===
curve_data = []

for _, row in edges.iterrows():
    geom = row['geometry']
    if not isinstance(geom, LineString):
        continue
    coords = list(geom.coords)
    if len(coords) < 3:
        continue
    for i in range(len(coords) - 2):
        p1, p2, p3 = coords[i], coords[i+1], coords[i+2]
        radius = radius_from_3_points(p1, p2, p3)
        segment = LineString([p1, p2, p3])
        curve_data.append({'radius_m': radius, 'geometry': segment})

curve_gdf = gpd.GeoDataFrame(curve_data, crs=edges.crs)

# === Approximate nearest station to each curve segment ===
station_tags = {'railway': 'station', 'station': 'subway'}
stations = ox.features_from_place(CITY, station_tags)
stations = stations[stations.geometry.type == 'Point'][['name', 'geometry']].dropna().reset_index(drop=True)

# Match to stations
curve_gdf = curve_gdf.to_crs(epsg=3857)
stations = stations.to_crs(epsg=3857)
curve_gdf['centroid'] = curve_gdf.geometry.centroid
nearest = gpd.sjoin_nearest(curve_gdf.set_geometry('centroid'), stations.set_geometry('geometry'), how='left')
nearest = nearest.rename(columns={'name': 'nearest_station'})

# === Filter to only include stations in your CSV ===
nearest = nearest[nearest['nearest_station'].isin(station_names)]

# === Aggregate radius per station ===
radius_summary = nearest.groupby('nearest_station').agg(
    avg_radius_m=('radius_m', 'mean'),
    min_radius_m=('radius_m', 'min'),
    max_radius_m=('radius_m', 'max'),
    curve_count=('radius_m', 'count')
).reset_index()

# === Save final result ===
radius_summary = radius_summary.round(8)
radius_summary.to_csv(OUTPUT_CSV, index=False)
stations_map = stations.to_crs(epsg=4326)
map_data = stations_map.merge(radius_summary, left_on='name', right_on='nearest_station', how='inner')
print(f"Saved station curvature data to: {OUTPUT_CSV}")

# === Reproject station geometry to WGS84 (for folium) ===
map_data = map_data.to_crs(epsg=4326)

# === Center of the map ===
center_latlon = map_data.geometry.unary_union.centroid.coords[0][::-1]
m = folium.Map(location=center_latlon, zoom_start=12)

# === Color scale: tighter = darker ===
colormap = cm.linear.YlOrRd_09.scale(
    map_data['avg_radius_m'].min(),
    map_data['avg_radius_m'].max()
)
colormap.caption = 'Average Curve Radius (m)'
colormap.add_to(m)

# === Optional: Intro popup marker ===
folium.Marker(
    location=center_latlon,
    icon=folium.Icon(color='blue', icon='info-sign'),
    popup=folium.Popup(
        "<b>How to Read This Map</b><br>"
        "Each point is a London Underground station.<br>"
        "Color = track curvature near the station.<br>"
        "<b>Darker = tighter curves = more noise potential.</b><br>"
        "Click any station for details.",
        max_width=300
    )
).add_to(m)

# === Add stations using CircleMarkers + color scale + popup ===
marker_cluster = MarkerCluster().add_to(m)

for _, row in map_data.iterrows():
    lat, lon = row.geometry.y, row.geometry.x
    color = colormap(row['avg_radius_m'])

    popup_html = f"""
        <b>{row['nearest_station']}</b><br>
        <b>Avg Radius:</b> {row['avg_radius_m']:.3f} m<br>
        <b>Min Radius:</b> {row['min_radius_m']:.3f} m<br>
        <b>Max Radius:</b> {row['max_radius_m']:.3f} m<br>
        <b>Curve Segments:</b> {row['curve_count']}
    """

    folium.CircleMarker(
        location=(lat, lon),
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(marker_cluster)

# === Save and optionally open ===
m.save("station_curve_radii_map.html")
print("✅ Interactive map saved as: station_curve_radii_map.html")

# === Open the map in the default web browser ===
file_path = os.path.abspath("station_curve_radii_map.html")
webbrowser.open(f"file://{file_path}")


# === ADDITION: Manually calculate curvature for Leytonstone–Wanstead and Liverpool Street–Bethnal Green ===

# Manually specify the new segments
special_segments = [
    ('Leytonstone', 'Wanstead'),
    ('Liverpool Street', 'Bethnal Green')
]

# Function to calculate curvature along a manually selected segment
def calculate_segment_curvature(station_a, station_b):
    # Filter track edges close to the two stations
    station_points = stations.to_crs(epsg=4326)
    point_a = station_points[station_points['name'] == station_a].geometry.values[0]
    point_b = station_points[station_points['name'] == station_b].geometry.values[0]

    # Select edges near both points
    buffer_distance = 0.002  # ~200 meters
    edges_near_a = edges.cx[point_a.x - buffer_distance:point_a.x + buffer_distance,
                            point_a.y - buffer_distance:point_a.y + buffer_distance]
    edges_near_b = edges.cx[point_b.x - buffer_distance:point_b.x + buffer_distance,
                            point_b.y - buffer_distance:point_b.y + buffer_distance]

    segment_edges = pd.concat([edges_near_a, edges_near_b])

    # Calculate radii on this segment
    radii = []
    for _, row in segment_edges.iterrows():
        geom = row['geometry']
        if not isinstance(geom, LineString):
            continue
        coords = list(geom.coords)
        if len(coords) < 3:
            continue
        for i in range(len(coords) - 2):
            p1, p2, p3 = coords[i], coords[i+1], coords[i+2]
            radius = radius_from_3_points(p1, p2, p3)
            radii.append(radius)
    
    # Aggregate results
    if radii:
        avg_r = np.mean(radii)
        min_r = np.min(radii)
        max_r = np.max(radii)
        count = len(radii)
    else:
        avg_r = min_r = max_r = 99999
        count = 0
    
    return {
        'nearest_station': f"{station_a}–{station_b}",
        'avg_radius_m': avg_r,
        'min_radius_m': min_r,
        'max_radius_m': max_r,
        'curve_count': count
    }

# Calculate and add manual segments
manual_rows = []
for station_a, station_b in special_segments:
    result = calculate_segment_curvature(station_a, station_b)
    manual_rows.append(result)

manual_df = pd.DataFrame(manual_rows)

# Combine original results with manual segments
radius_summary = pd.concat([radius_summary, manual_df], ignore_index=True)

# Re-round
radius_summary = radius_summary.round(8)