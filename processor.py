"""
Pickup/Dropoff Counts Data Processor

This module processes raw taxi GPS trajectory data to count pickup and dropoff events
for each spatiotemporal key (x_grid, y_grid, time_bucket, day_of_week).
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from itertools import product


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    grid_size: float = 0.01  # degrees
    time_interval: int = 5   # minutes (288 buckets per day)
    exclude_sunday: bool = True
    
    # Index offsets for alignment with existing dataset
    x_grid_offset: int = 1  # Add 2 to x_grid indices
    y_grid_offset: int = 1  # Add 1 to y_grid indices
    time_offset: int = 1    # Add 1 to time indices (0-based -> 1-based)
    
    # Dense dataset generation
    generate_dense: bool = True  # Include zero entries for all grid cells
    include_saturday_zeros: bool = False  # Include Saturday (day 6) with zeros
    
    # Explicit grid max values (for alignment with existing dataset)
    # When set (not None), these override the empirical max from data
    x_grid_max: Optional[int] = None  # Set to 48 to match existing dataset
    y_grid_max: Optional[int] = None  # Set to 90 to match existing dataset
    
    # Paths
    raw_data_dir: Path = Path(__file__).parent.parent / "raw_data"
    output_dir: Path = Path(__file__).parent / "output"
    
    # Input files
    input_files: Tuple[str, ...] = (
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl",
        "taxi_record_09_50drivers.pkl",
    )


@dataclass
class GlobalBounds:
    """Stores the global GPS coordinate bounds for quantization."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    
    def to_dict(self) -> dict:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


@dataclass
class ProcessingStats:
    """Statistics from the processing run."""
    total_records: int = 0
    records_after_sunday_filter: int = 0
    total_pickups: int = 0
    total_dropoffs: int = 0
    unique_keys: int = 0
    unique_plates: int = 0
    processing_time_seconds: float = 0.0
    global_bounds: Optional[GlobalBounds] = None


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Load raw taxi GPS data from a pickle file.
    
    The raw data structure is:
    {
        plate_id: [              # Dict keyed by driver plate ID
            [                    # List of days (e.g., 21 days for July)
                [record],        # Day contains list of GPS records
                [record],
                ...
            ],
            ...
        ],
        ...
    }
    
    Each record is a 6-element list:
    [plate_id, latitude, longitude, seconds_since_midnight, passenger_indicator, timestamp_str]
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        DataFrame with columns: plate_id, latitude, longitude, seconds, passenger_indicator, timestamp
        
    Raises:
        EOFError: If the file is empty or corrupted
        ValueError: If the data structure is unexpected
    """
    # Check for empty file
    if filepath.stat().st_size == 0:
        raise EOFError(f"File is empty: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise EOFError(f"Failed to load pickle file {filepath}: {e}")
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    
    # Data is a dict keyed by plate_id
    # Each value is a list of days, each day is a list of GPS records
    all_records = []
    for plate_id, days_list in data.items():
        for day_records in days_list:
            # day_records is a list of GPS record lists
            all_records.extend(day_records)
    
    if not all_records:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'plate_id', 'latitude', 'longitude', 'seconds', 'passenger_indicator', 'timestamp'
        ])
    
    # Convert to DataFrame with appropriate column names
    df = pd.DataFrame(all_records, columns=[
        'plate_id', 'latitude', 'longitude', 'seconds', 'passenger_indicator', 'timestamp'
    ])
    
    return df


def compute_global_bounds(dfs: List[pd.DataFrame]) -> GlobalBounds:
    """
    Compute global GPS bounds from a list of DataFrames.
    
    Critical: Bounds must be computed from the entire combined dataset
    before any quantization occurs.
    
    Args:
        dfs: List of DataFrames containing latitude and longitude columns
        
    Returns:
        GlobalBounds object with min/max lat/lon
    """
    all_lats = pd.concat([df['latitude'] for df in dfs])
    all_lons = pd.concat([df['longitude'] for df in dfs])
    
    return GlobalBounds(
        lat_min=all_lats.min(),
        lat_max=all_lats.max(),
        lon_min=all_lons.min(),
        lon_max=all_lons.max(),
    )


def gps_to_grid(
    lat: np.ndarray, 
    lon: np.ndarray, 
    bounds: GlobalBounds, 
    grid_size: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GPS coordinates to grid cell indices using numpy.digitize.
    
    Args:
        lat: Array of latitude values
        lon: Array of longitude values
        bounds: GlobalBounds object with coordinate bounds
        grid_size: Size of each grid cell in degrees
        
    Returns:
        Tuple of (x_grid, y_grid) arrays
    """
    lat_bins = np.arange(bounds.lat_min, bounds.lat_max + grid_size, grid_size)
    lon_bins = np.arange(bounds.lon_min, bounds.lon_max + grid_size, grid_size)
    
    x_grid = np.digitize(lat, lat_bins) - 1   # latitude → x_grid
    y_grid = np.digitize(lon, lon_bins) - 1   # longitude → y_grid
    
    return x_grid, y_grid


def timestamp_to_time_bin(timestamps: pd.Series) -> pd.Series:
    """
    Convert timestamps to time-of-day bins (5-minute intervals).
    
    Args:
        timestamps: Series of datetime objects
        
    Returns:
        Series of time bin indices [0, 287]
    """
    minutes_since_midnight = timestamps.dt.hour * 60 + timestamps.dt.minute
    time_bin = minutes_since_midnight // 5
    return time_bin


def timestamp_to_day(timestamps: pd.Series, exclude_sunday: bool = True) -> pd.Series:
    """
    Convert timestamps to day-of-week indices.
    
    Monday = 1, Tuesday = 2, ..., Saturday = 6
    Sunday is optionally excluded (returns NaN).
    
    Args:
        timestamps: Series of datetime objects
        exclude_sunday: If True, Sunday records get NaN
        
    Returns:
        Series of day indices [1, 6] with NaN for Sundays if excluded
    """
    dow = timestamps.dt.weekday  # Monday=0, Sunday=6
    
    if exclude_sunday:
        # Convert to 1-indexed, Sundays become NaN
        day = dow.where(dow != 6) + 1
    else:
        # Convert to 1-indexed including Sunday (7)
        day = dow + 1
    
    return day


def detect_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect pickup and dropoff transitions for each taxi.
    
    A pickup occurs when passenger_indicator transitions from 0 → 1.
    A dropoff occurs when passenger_indicator transitions from 1 → 0.
    
    Args:
        df: DataFrame sorted by (plate_id, timestamp)
        
    Returns:
        DataFrame with 'transition' column:
            +1 = pickup
            -1 = dropoff
            NaN = no transition or first record
    """
    # Sort by plate_id and timestamp
    df = df.sort_values(['plate_id', 'timestamp']).reset_index(drop=True)
    
    # Compute transitions within each plate_id group
    df['transition'] = df.groupby('plate_id')['passenger_indicator'].diff()
    
    return df


def process_data(
    config: ProcessingConfig,
    progress_callback: Optional[callable] = None
) -> Tuple[Dict[Tuple[int, int, int, int], Tuple[int, int]], ProcessingStats]:
    """
    Main processing function that loads raw data and computes pickup/dropoff counts.
    
    Args:
        config: ProcessingConfig object with processing parameters
        progress_callback: Optional callback function(stage, progress) for progress updates
        
    Returns:
        Tuple of (pickup_dropoff_counts dict, ProcessingStats)
    """
    import time
    start_time = time.time()
    
    stats = ProcessingStats()
    
    def update_progress(stage: str, progress: float):
        if progress_callback:
            progress_callback(stage, progress)
    
    # Stage 1: Load all raw data files
    update_progress("Loading raw data files...", 0.0)
    dfs = []
    skipped_files = []
    for i, filename in enumerate(config.input_files):
        filepath = config.raw_data_dir / filename
        if filepath.exists():
            try:
                df = load_raw_data(filepath)
                if len(df) > 0:
                    dfs.append(df)
                    update_progress(f"Loaded {filename}", (i + 1) / len(config.input_files) * 0.2)
                else:
                    skipped_files.append((filename, "empty data"))
                    update_progress(f"Skipped {filename} (empty)", (i + 1) / len(config.input_files) * 0.2)
            except (EOFError, ValueError) as e:
                skipped_files.append((filename, str(e)))
                update_progress(f"Skipped {filename} (error)", (i + 1) / len(config.input_files) * 0.2)
        else:
            skipped_files.append((filename, "file not found"))
            update_progress(f"Skipped {filename} (not found)", (i + 1) / len(config.input_files) * 0.2)
    
    if not dfs:
        raise ValueError(f"No valid data files found. Skipped: {skipped_files}")
    
    # Concatenate all data
    combined_df = pd.concat(dfs, ignore_index=True)
    stats.total_records = len(combined_df)
    stats.unique_plates = combined_df['plate_id'].nunique()
    
    # Stage 2: Compute global bounds
    update_progress("Computing global bounds...", 0.2)
    bounds = compute_global_bounds([combined_df])
    stats.global_bounds = bounds
    
    # Stage 3: Parse timestamps
    update_progress("Parsing timestamps...", 0.25)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    # Stage 4: Filter out Sundays
    update_progress("Filtering Sundays...", 0.3)
    combined_df['day'] = timestamp_to_day(combined_df['timestamp'], config.exclude_sunday)
    if config.exclude_sunday:
        combined_df = combined_df.dropna(subset=['day'])
    combined_df['day'] = combined_df['day'].astype(int)
    stats.records_after_sunday_filter = len(combined_df)
    
    # Stage 5: Apply spatial quantization
    update_progress("Applying spatial quantization...", 0.4)
    x_grid, y_grid = gps_to_grid(
        combined_df['latitude'].values,
        combined_df['longitude'].values,
        bounds,
        config.grid_size
    )
    # Apply offsets for alignment with existing dataset
    combined_df['x_grid'] = x_grid + config.x_grid_offset
    combined_df['y_grid'] = y_grid + config.y_grid_offset
    
    # Stage 6: Apply temporal quantization
    update_progress("Applying temporal quantization...", 0.5)
    combined_df['time'] = timestamp_to_time_bin(combined_df['timestamp']) + config.time_offset
    
    # Stage 7: Detect transitions
    update_progress("Detecting pickup/dropoff transitions...", 0.6)
    combined_df = detect_transitions(combined_df)
    
    # Stage 8: Identify pickup and dropoff events
    update_progress("Identifying events...", 0.7)
    pickups_df = combined_df[combined_df['transition'] == 1].copy()
    dropoffs_df = combined_df[combined_df['transition'] == -1].copy()
    
    stats.total_pickups = len(pickups_df)
    stats.total_dropoffs = len(dropoffs_df)
    
    # Stage 9: Aggregate counts by spatiotemporal key
    update_progress("Aggregating counts...", 0.8)
    
    pickup_counts = pickups_df.groupby(
        ['x_grid', 'y_grid', 'time', 'day']
    ).size().reset_index(name='pickup_count')
    
    dropoff_counts = dropoffs_df.groupby(
        ['x_grid', 'y_grid', 'time', 'day']
    ).size().reset_index(name='dropoff_count')
    
    # Stage 10: Merge pickup and dropoff counts
    update_progress("Merging counts...", 0.9)
    
    # Full outer join to capture all keys
    merged = pd.merge(
        pickup_counts,
        dropoff_counts,
        on=['x_grid', 'y_grid', 'time', 'day'],
        how='outer'
    ).fillna(0)
    
    merged['pickup_count'] = merged['pickup_count'].astype(int)
    merged['dropoff_count'] = merged['dropoff_count'].astype(int)
    
    # Stage 11: Generate dense dataset if requested
    if config.generate_dense:
        update_progress("Generating dense dataset...", 0.95)
        
        # Determine grid ranges from data (empirical)
        x_min_empirical, x_max_empirical = int(combined_df['x_grid'].min()), int(combined_df['x_grid'].max())
        y_min_empirical, y_max_empirical = int(combined_df['y_grid'].min()), int(combined_df['y_grid'].max())
        time_min, time_max = int(combined_df['time'].min()), int(combined_df['time'].max())
        
        # Use explicit max values if configured, otherwise use empirical
        x_min = x_min_empirical
        x_max = config.x_grid_max if config.x_grid_max is not None else x_max_empirical
        y_min = y_min_empirical
        y_max = config.y_grid_max if config.y_grid_max is not None else y_max_empirical
        
        # Determine day range
        existing_days = sorted(combined_df['day'].unique())
        if config.include_saturday_zeros and 6 not in existing_days:
            # Add Saturday (day 6) to the list
            all_days = existing_days + [6]
        else:
            all_days = existing_days
        
        # Create all possible combinations
        all_keys = list(product(
            range(x_min, x_max + 1),
            range(y_min, y_max + 1),
            range(time_min, time_max + 1),
            all_days
        ))
        
        # Create dense DataFrame with all combinations
        dense_df = pd.DataFrame(all_keys, columns=['x_grid', 'y_grid', 'time', 'day'])
        
        # Merge with actual counts (left join to keep all dense keys)
        dense_merged = pd.merge(
            dense_df,
            merged,
            on=['x_grid', 'y_grid', 'time', 'day'],
            how='left'
        ).fillna(0)
        
        dense_merged['pickup_count'] = dense_merged['pickup_count'].astype(int)
        dense_merged['dropoff_count'] = dense_merged['dropoff_count'].astype(int)
        
        merged = dense_merged
    
    # Convert to dictionary format
    pickup_dropoff_counts = {}
    for _, row in merged.iterrows():
        key = (int(row['x_grid']), int(row['y_grid']), int(row['time']), int(row['day']))
        value = (int(row['pickup_count']), int(row['dropoff_count']))
        pickup_dropoff_counts[key] = value
    
    stats.unique_keys = len(pickup_dropoff_counts)
    stats.processing_time_seconds = time.time() - start_time
    
    update_progress("Complete!", 1.0)
    
    return pickup_dropoff_counts, stats


def save_output(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]], 
    output_path: Path
) -> None:
    """
    Save the pickup/dropoff counts dictionary to a pickle file.
    
    Args:
        data: Dictionary with (x_grid, y_grid, time, day) keys and (pickup, dropoff) values
        output_path: Path to save the pickle file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def load_output(filepath: Path) -> Dict[Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Load a previously saved pickup/dropoff counts dictionary.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary with (x_grid, y_grid, time, day) keys and (pickup, dropoff) values
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_existing_volume_pickups(filepath: Path) -> Dict[Tuple[int, int, int, int], List]:
    """
    Load the existing latest_volume_pickups.pkl for validation comparison.
    
    Args:
        filepath: Path to latest_volume_pickups.pkl
        
    Returns:
        Dictionary with (x_grid, y_grid, time, day) keys and [pickup_count, volume] values
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


@dataclass
class ValidationResult:
    """Results from validation comparison."""
    total_existing_keys: int
    total_generated_keys: int
    matching_keys: int
    exact_pickup_matches: int
    close_matches: int  # within 20%
    total_existing_pickups: int
    total_generated_pickups: int
    correlation: float
    discrepancies: pd.DataFrame  # Top discrepancies for analysis


def validate_against_existing(
    generated: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    existing_path: Path,
    top_n_discrepancies: int = 100
) -> ValidationResult:
    """
    Compare generated pickup counts against existing dataset.
    
    Args:
        generated: Generated pickup/dropoff counts dictionary
        existing_path: Path to latest_volume_pickups.pkl
        top_n_discrepancies: Number of top discrepancies to return
        
    Returns:
        ValidationResult with comparison statistics
    """
    existing = load_existing_volume_pickups(existing_path)
    
    # Convert existing keys from string tuples to actual tuples if needed
    existing_parsed = {}
    for key, value in existing.items():
        if isinstance(key, str):
            # Parse string tuple like "(36, 75, 161, 4)"
            key = tuple(map(int, key.strip('()').split(',')))
        existing_parsed[key] = value
    
    existing_keys = set(existing_parsed.keys())
    generated_keys = set(generated.keys())
    
    matching_keys = existing_keys & generated_keys
    
    # Build comparison DataFrame
    comparison_data = []
    for key in matching_keys:
        existing_pickup = existing_parsed[key][0]  # pickup is index 0
        generated_pickup = generated[key][0]  # pickup is index 0
        comparison_data.append({
            'key': key,
            'x_grid': key[0],
            'y_grid': key[1],
            'time': key[2],
            'day': key[3],
            'existing_pickup': existing_pickup,
            'generated_pickup': generated_pickup,
            'diff': abs(existing_pickup - generated_pickup),
            'pct_diff': abs(existing_pickup - generated_pickup) / max(existing_pickup, 1) * 100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        exact_matches = (comparison_df['existing_pickup'] == comparison_df['generated_pickup']).sum()
        close_matches = (comparison_df['pct_diff'] <= 20).sum()
        
        # Correlation
        if comparison_df['existing_pickup'].std() > 0 and comparison_df['generated_pickup'].std() > 0:
            correlation = comparison_df['existing_pickup'].corr(comparison_df['generated_pickup'])
        else:
            correlation = 0.0
        
        # Top discrepancies
        discrepancies = comparison_df.nlargest(top_n_discrepancies, 'diff')
    else:
        exact_matches = 0
        close_matches = 0
        correlation = 0.0
        discrepancies = pd.DataFrame()
    
    total_existing_pickups = sum(v[0] for v in existing_parsed.values())
    total_generated_pickups = sum(v[0] for v in generated.values())
    
    return ValidationResult(
        total_existing_keys=len(existing_keys),
        total_generated_keys=len(generated_keys),
        matching_keys=len(matching_keys),
        exact_pickup_matches=int(exact_matches),
        close_matches=int(close_matches),
        total_existing_pickups=total_existing_pickups,
        total_generated_pickups=total_generated_pickups,
        correlation=correlation,
        discrepancies=discrepancies
    )


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Process taxi GPS data for pickup/dropoff counts")
    parser.add_argument("--output", type=str, default="output/pickup_dropoff_counts.pkl",
                        help="Output pickle file path")
    args = parser.parse_args()
    
    config = ProcessingConfig()
    
    def progress_callback(stage, progress):
        print(f"[{progress*100:.1f}%] {stage}")
    
    print("Starting data processing...")
    counts, stats = process_data(config, progress_callback)
    
    print(f"\nProcessing complete!")
    print(f"  Total records: {stats.total_records:,}")
    print(f"  After Sunday filter: {stats.records_after_sunday_filter:,}")
    print(f"  Unique plates: {stats.unique_plates:,}")
    print(f"  Total pickups: {stats.total_pickups:,}")
    print(f"  Total dropoffs: {stats.total_dropoffs:,}")
    print(f"  Unique keys: {stats.unique_keys:,}")
    print(f"  Processing time: {stats.processing_time_seconds:.2f}s")
    
    output_path = Path(args.output)
    save_output(counts, output_path)
    print(f"\nSaved to {output_path}")
