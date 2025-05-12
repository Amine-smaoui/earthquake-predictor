import pandas as pd
import numpy as np

def prepare_events(events_path, seed=2025):
    """
    Load and process the earthquake events data.
    Includes feature extraction, label creation, and cleaning.
    """
    np.random.seed(seed)

    events = pd.read_csv(events_path)

    # Convert time to datetime
    events['time'] = pd.to_datetime(events['time'])
    events['date'] = events['time'].dt.date

    # Extract value after ":" for the feature Depth
    events['depth'] = events['depth'].astype(str).str.split(":").str[-1].str.strip().astype(float)

    # Remove 'net' as it is a unique ID, and does not provide useful information
    events.drop(columns=['net'], inplace=True)

    # Create the label: 1 if mag > 4.4, else 0
    events['is_high_magnitude'] = (events['mag'] > 4.4).astype(int)

    return events


def prepare_weather(weather_path):
    """
    Load and aggregate hourly weather data to daily values per location.
    """
    weather = pd.read_csv(weather_path)

    # Convert time to datetime
    weather['time'] = pd.to_datetime(weather['time'])
    weather['date'] = weather['time'].dt.date

    # Aggregate hourly weather data to daily values per location
    agg_weather = weather.groupby(['date', 'lat', 'lng']).agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'precipitation': 'mean',
        'sealevelPressure': 'mean',
        'surfacePressure': 'mean',
        'nst': 'mean'
    }).reset_index()

    return agg_weather


def merge_data(events, agg_weather):
    """
    Merge the events and weather datasets based on date and location.
    """
    combined = pd.merge(
        events,
        agg_weather,
        how='left',
        left_on=['date', 'latitude', 'longitude'],
        right_on=['date', 'lat', 'lng']
    )

    return combined
