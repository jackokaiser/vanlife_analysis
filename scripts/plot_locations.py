import argparse
import os
import json
import shutil
import tempfile
from bisect import bisect_left
from datetime import datetime
import pandas as pd
from geopy import distance
import numpy as np
from dataclasses import dataclass, asdict

from vanlife_analysis.utils import parse_date_interval


@dataclass
class Location:
    latitude: float
    longitude: float
    timestamp: int
    accuracy: int

    @classmethod
    def deserialize(cls, loc: dict):
        return cls(**{
            'latitude': float(loc['latitudeE7']) / 1e7,
            'longitude': float(loc['longitudeE7']) / 1e7,
            'timestamp': int(loc['timestampMs']) // 1000,
            'accuracy': loc['accuracy']
        })

    @property
    def latlon(self) -> tuple:
        return self.latitude, self.longitude


def load_json(path_to_json: str) -> dict:
    with open(path_to_json) as file:
        return json.load(file)


def to_kml(locations: list, save_path: str) -> None:
    coordinates = '\n'.join([f'{location.longitude}, {location.latitude}' for location in locations])
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
        <name>Trip</name>
        <description>Paths in KML.</description>
        <Style id="lineStyle">
          <LineStyle>
                <color>eb3446ff</color>
                <width>4</width>
          </LineStyle>
        </Style>
        <Placemark>
          <name>Trip</name>
          <description>Path of the trip</description>
          <styleUrl>#lineStyle</styleUrl>
          <LineString>
                <extrude>1</extrude>
                <tessellate>1</tessellate>
                <coordinates>
                    {coordinates}
                </coordinates>
          </LineString>
        </Placemark>
  </Document>
</kml>'''
    with open(save_path, "w") as file:
        file.write(kml)


def load_location_history(path_to_location_zip: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # unzip location history to temporary directory
        shutil.unpack_archive(path_to_location_zip, tmp_dir, 'zip')
        location_history_path = os.path.join(tmp_dir, 'Takeout/Location History/Location History.json')
        location_history = load_json(location_history_path)
        return location_history


def get_date(location: dict) -> datetime:
    return datetime.fromtimestamp(int(location['timestampMs']) / 1000)


def subsample_locations(locations: list, min_km_thresh: float = 10, outlier_km_thresh: float = 50) -> list:
    distances = np.array([distance.distance(loc_a.latlon, loc_b.latlon).km
                          for loc_a, loc_b in zip(locations[:-1], locations[1:])])

    def is_outlier(ii: int) -> bool:
        return distances[ii] > outlier_km_thresh

    subsampled_locations = [locations[0]]
    for ii, location in enumerate(locations[1:]):
        dist = distance.distance(subsampled_locations[-1].latlon, location.latlon).km
        if min_km_thresh < dist and not is_outlier(ii):
            subsampled_locations.append(location)
    return subsampled_locations


def extract_locations_in_interval(locations: list,
                                  start_date: datetime, stop_date: datetime) -> list:
    start_idx = bisect_left(locations, start_date, key=lambda location: get_date(location))
    stop_idx = bisect_left(locations, stop_date, key=lambda location: get_date(location))
    return locations[start_idx:stop_idx]


def convert_to_df(locations: list) -> pd.DataFrame:
    return pd.DataFrame((asdict(location) for location in locations))


def plot_locations(path_to_location_zip: str, date_interval: list, save_dir: str) -> None:
    start_date, stop_date = parse_date_interval(date_interval)
    location_history = load_location_history(path_to_location_zip)
    locations = extract_locations_in_interval(location_history['locations'], start_date, stop_date)
    locations = [Location.deserialize(loc) for loc in locations]
    locations = subsample_locations(locations)
    convert_to_df(locations).to_csv(os.path.join(save_dir, 'locations.csv'), index=False)
    to_kml(locations, os.path.join(save_dir, 'locations.kml'))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to plot locations in google my maps')
    parser.add_argument('--location_zip', help='Path to the zipped exported takeout location', type=str, required=True,
                        dest='path_to_location_zip')
    parser.add_argument('--date_interval', help='Date interval for plotting locations', nargs='+', type=str,
                        required=True)
    parser.add_argument('--save_dir', help='Directory where the computed data is saved', type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_locations(**vars(args))
