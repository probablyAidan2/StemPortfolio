import numpy as np
import csv
from sklearn.cluster import DBSCAN

def read_coordinates_from_csv(filename):
    coords = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                lat = float(row[2])
                lon = float(row[3])
                coords.append((lat, lon))
            except (ValueError, IndexError):
                continue  # Skip malformed rows
    return coords

def group_by_dbscan(coords, max_km):
    rad = np.radians(coords)
    eps = max_km / 6371.0088
    clustering = DBSCAN(eps=eps, min_samples=1, metric='haversine').fit(rad)

    clusters = {}
    for label, point in zip(clustering.labels_, coords):
        clusters.setdefault(label, []).append(point)
    return clusters

def compute_centroids(clusters):
    centroids = {}
    for label, points in clusters.items():
        arr = np.array(points)
        avg_lat = np.mean(arr[:, 0])
        avg_lon = np.mean(arr[:, 1])
        centroids[label] = (avg_lat, avg_lon)
    return centroids

def main():
    filename = input("Enter the path to your CSV file: ").strip()
    coordinates = read_coordinates_from_csv(filename)
    if not coordinates:
        print("No valid coordinates found in the file.")
        return

    max_km = float(input("Enter clustering radius in kilometers: "))
    clusters = group_by_dbscan(coordinates, max_km)
    centroids = compute_centroids(clusters)

    for label in sorted(clusters.keys()):
        lat, lon = centroids[label]
        print(f"\nCluster {label}: Picture Point → ({lat:.6f}, {lon:.6f})")
        print("  Coordinates in cluster:")
        for point in clusters[label]:
            print(f"    ({point[0]:.6f}, {point[1]:.6f})")

if __name__ == "__main__":
    main()
