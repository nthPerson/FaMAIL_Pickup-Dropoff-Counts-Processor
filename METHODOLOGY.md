# Pickup and Dropoff Event Counting Methodology

## Overview

This document describes the algorithmic approach for detecting and counting taxi pickup and dropoff events from raw GPS trajectory data, as implemented in the Pickup/Dropoff Counts Processor. The methodology is designed for use in transportation research and imitation learning applications.

## 1. Data Aggregation

### 1.1 Multi-Driver Aggregation

**Counts are aggregated across all drivers/experts.** The processor combines data from all taxi drivers (identified by unique plate IDs) into a single unified dataset before counting events. Individual driver identities are used only for:

1. **Transition detection** - ensuring state changes are computed within individual driver trajectories
2. **Temporal ordering** - maintaining chronological sequence of events per vehicle

After event detection, all drivers' pickup and dropoff events are pooled together and aggregated by spatiotemporal location, effectively treating the entire taxi fleet as a collective source of ground truth for passenger demand patterns.

### 1.2 Data Sources

The implementation processes three months of GPS trajectory data (July-September 2016) from 50 taxi drivers in Shenzhen, China:

- **taxi_record_07_50drivers.pkl**: ~2.7M GPS records
- **taxi_record_08_50drivers.pkl**: ~3.0M GPS records  
- **taxi_record_09_50drivers.pkl**: ~2.5M GPS records

**Total**: ~8.1M individual GPS observations across all drivers and time periods.

## 2. Generalized Algorithm

### 2.1 High-Level Pipeline

```
Input: Raw GPS trajectories from multiple taxi drivers
Output: Dictionary mapping spatiotemporal keys to (pickup_count, dropoff_count)

1. DATA LOADING
   └─ Load and combine all driver trajectories

2. GLOBAL BOUNDS COMPUTATION
   └─ Compute min/max lat/lon from entire dataset

3. TEMPORAL PARSING
   └─ Convert timestamp strings to datetime objects

4. TEMPORAL FILTERING
   └─ Remove Sunday records (no Sunday records in dataset; no effect)

5. SPATIAL QUANTIZATION
   └─ Convert (lat, lon) → (x_grid, y_grid) using global bounds

6. TEMPORAL QUANTIZATION
   └─ Convert timestamps → (time_bucket, day_of_week)

7. TRANSITION DETECTION
   └─ Identify passenger_indicator changes per driver

8. EVENT EXTRACTION
   └─ Classify transitions as pickups or dropoffs

9. SPATIOTEMPORAL AGGREGATION
   └─ Group events by (x_grid, y_grid, time, day) and count

10. DENSIFICATION
    └─ Fill in zero counts for unobserved spatiotemporal cells
```

### 2.2 Detailed Algorithm Steps

#### Step 1: Data Loading and Consolidation

```
FOR each monthly data file:
    LOAD pickle file containing nested structure:
        {plate_id: [[day1_records], [day2_records], ...]}
    
    FOR each driver (plate_id):
        FOR each day in driver's data:
            EXTRACT all GPS records: [plate_id, lat, lon, seconds, passenger, timestamp]
            APPEND to combined record list
    
CONCATENATE all records into single DataFrame
RESULT: ~8.1M unified GPS observations
```

**Key Point**: All driver data is flattened into a single dataframe, removing hierarchical structure while preserving plate_id for transition detection.

#### Step 2: Global Bounds Computation

```
lat_min ← MINIMUM(all latitudes across all records)
lat_max ← MAXIMUM(all latitudes across all records)
lon_min ← MINIMUM(all longitudes across all records)
lon_max ← MAXIMUM(all longitudes across all records)

STORE as GlobalBounds(lat_min, lat_max, lon_min, lon_max)
```

**Rationale**: Computing bounds from the complete dataset ensures consistent spatial quantization and prevents edge cases where different subsets would produce different grid alignments.

**Observed Bounds** (Shenzhen data):
- Latitude: 22.4425° to 22.8700° (span: ~47 km)
- Longitude: 113.7501° to 114.5582° (span: ~90 km)

#### Step 3: Spatial Quantization

```
DEFINE grid_size = 0.01 degrees (approximately 1.1 km)

FOR each GPS coordinate (lat, lon):
    # Create bin edges for digitization
    lat_bins ← ARANGE(lat_min, lat_max + grid_size, step=grid_size)
    lon_bins ← ARANGE(lon_min, lon_max + grid_size, step=grid_size)
    
    # Assign to grid cell using numpy.digitize
    x_grid ← DIGITIZE(lat, lat_bins) - 1
    y_grid ← DIGITIZE(lon, lon_bins) - 1
    
    # Apply alignment offsets (optional, for compatibility)
    x_grid ← x_grid + x_grid_offset
    y_grid ← y_grid + y_grid_offset
```

**Implementation Note**: `numpy.digitize` assigns each continuous coordinate to the nearest discrete bin. The `-1` adjustment accounts for digitize's 1-based indexing. Offsets are applied to align with existing datasets (default: x_offset=1, y_offset=1).

**Result**: Each GPS point is assigned to one of ~4,000 grid cells (48 × 90 grid).

#### Step 4: Temporal Quantization

```
FOR each timestamp:
    # Extract temporal features
    hour ← EXTRACT_HOUR(timestamp)
    minute ← EXTRACT_MINUTE(timestamp)
    weekday ← EXTRACT_WEEKDAY(timestamp)  # Monday=0, Sunday=6
    
    # Convert to 5-minute time buckets
    minutes_since_midnight ← hour × 60 + minute
    time_bucket ← FLOOR(minutes_since_midnight / 5) + time_offset
    
    # Convert to 1-indexed day (Monday=1, ..., Saturday=6)
    IF weekday == 6 AND exclude_sunday THEN
        day ← NULL  # Mark for removal
    ELSE
        day ← weekday + 1
    
FILTER OUT records where day == NULL
```

**Result**: 
- Time dimension: 288 buckets per day (5-minute resolution, 1-indexed: 1-288)
- Day dimension: 6 values (Monday=1 through Saturday=6)
- Sundays excluded by default (can be optionally included)

#### Step 5: Transition Detection (Core Algorithm)

This is the critical step where actual pickups and dropoffs are identified.

```
# Sort all records by driver and time to ensure chronological order
SORT records BY (plate_id, timestamp) ASCENDING

# Compute passenger indicator changes within each driver's trajectory
FOR each driver (plate_id):
    trajectory ← GET_RECORDS_FOR_DRIVER(plate_id)
    
    FOR i FROM 1 TO LENGTH(trajectory):
        current_state ← trajectory[i].passenger_indicator
        previous_state ← trajectory[i-1].passenger_indicator
        
        transition ← current_state - previous_state
        
        IF transition == +1 THEN
            # State changed from 0 (empty) to 1 (occupied)
            MARK trajectory[i] as PICKUP event
        
        ELSE IF transition == -1 THEN
            # State changed from 1 (occupied) to 0 (empty)
            MARK trajectory[i] as DROPOFF event
        
        ELSE
            # No transition (0→0 or 1→1) or first record
            MARK trajectory[i] as NO_EVENT
```

**Mathematical Formulation**:
```
Let S(t) = passenger_indicator at time t ∈ {0, 1}

Transition(t) = S(t) - S(t-1)

Event(t) = {
    PICKUP    if Transition(t) = +1
    DROPOFF   if Transition(t) = -1
    NULL      if Transition(t) = 0 or t = 0
}
```

**Key Properties**:
1. Transitions are computed **per-driver** to ensure temporal continuity
2. First record of each driver's trajectory has no transition (no previous state)
3. Only state changes are counted; sustained states (passenger onboard for multiple records) are ignored
4. Method is robust to missing data within a driver's trajectory

**Implementation**: Uses pandas `groupby().diff()` to vectorize computation across all drivers simultaneously.

#### Step 6: Event Extraction and Filtering

```
pickups_df ← FILTER(combined_df WHERE transition == +1)
dropoffs_df ← FILTER(combined_df WHERE transition == -1)

# Each record in pickups_df represents one pickup event with:
# - Location: (x_grid, y_grid)
# - Time: (time_bucket, day)
# - Context: (plate_id, timestamp, lat, lon)

# Similarly for dropoffs_df
```

**Observed Event Counts** (typical run):
- Total pickups detected: ~326,000 events
- Total dropoffs detected: ~326,000 events
- Ratio: ~1.00 (expected, as pickups ≈ dropoffs)

#### Step 7: Spatiotemporal Aggregation

```
# Group pickup events by spatiotemporal key and count
pickup_counts ← GROUP_BY(pickups_df, [x_grid, y_grid, time, day])
                .AGGREGATE(COUNT)

# Result: DataFrame with columns [x_grid, y_grid, time, day, pickup_count]

# Repeat for dropoffs
dropoff_counts ← GROUP_BY(dropoffs_df, [x_grid, y_grid, time, day])
                 .AGGREGATE(COUNT)

# Merge pickup and dropoff counts
merged ← FULL_OUTER_JOIN(pickup_counts, dropoff_counts,
                         ON=[x_grid, y_grid, time, day])
         .FILL_MISSING_WITH(0)

# Convert to dictionary format
result ← {}
FOR each row IN merged:
    key ← (row.x_grid, row.y_grid, row.time, row.day)
    value ← (row.pickup_count, row.dropoff_count)
    result[key] ← value
```

**Output Structure**:
```python
{
    (x_grid, y_grid, time_bucket, day): (pickup_count, dropoff_count),
    # Example entries:
    (36, 75, 161, 4): (3, 2),  # 3 pickups, 2 dropoffs at this cell/time/day
    (12, 45, 96, 1): (5, 4),
    ...
}
```

**Key Characteristics**:
- Each key is a unique spatiotemporal cell
- Counts represent ALL drivers' events at that location/time
- Sparse representation: only cells with non-zero activity (optional)
- Dense representation: all possible cells including zeros (optional)

#### Step 8: Dense Dataset Generation (Optional)

```
IF generate_dense == TRUE THEN
    # Determine actual ranges from data
    x_range ← [MIN(x_grid), MAX(x_grid)]
    y_range ← [MIN(y_grid), MAX(y_grid)]
    time_range ← [MIN(time), MAX(time)]
    day_range ← UNIQUE(day)  # Typically [1, 2, 3, 4, 5]
    
    IF include_saturday_zeros AND 6 NOT IN day_range THEN
        day_range ← day_range + [6]  # Add Saturday with zeros
    
    # Generate all possible combinations
    all_keys ← CARTESIAN_PRODUCT(x_range, y_range, time_range, day_range)
    
    # Create dense dataframe with all combinations
    dense_df ← CREATE_DATAFRAME(all_keys)
    
    # Left join with actual counts (fills missing with 0)
    result ← LEFT_JOIN(dense_df, merged, ON=[x_grid, y_grid, time, day])
             .FILL_MISSING_WITH(0)
```

**Impact**:
- Sparse: ~234K keys (only non-zero cells)
- Dense: ~7.5M keys (all possible grid×time×day combinations)
- Dense format matches existing reference datasets for validation

## 3. Critical Implementation Details

### 3.1 Data-Driven Bounds

The spatial quantization uses **data-driven bounds** computed from the entire dataset before any grid assignment. This ensures:

1. **Consistency**: All records use the same grid definition
2. **Coverage**: Grid encompasses all observed locations
3. **Efficiency**: No wasted grid cells outside data range

**Alternative Rejected**: Fixed geographic bounds (e.g., Shenzhen city limits) would include many empty cells and potentially miss edge locations.

### 3.2 Transition Detection Order

The algorithm **must** sort records by `(plate_id, timestamp)` before computing transitions. This ensures:

1. Transitions are computed within each driver's trajectory
2. Temporal ordering is preserved (pickup precedes dropoff)
3. Cross-driver state changes are not erroneously detected

**Example Error if Unsorted**:
```
Driver A: [0, 1, 0]  (pickup at t=2, dropoff at t=3)
Driver B: [1, 0, 1]  (dropoff at t=2, pickup at t=3)

If records interleaved without grouping:
[A:0, B:1, A:1, B:0, A:0, B:1]
         ↑ False pickup     ↑ False dropoff
```

### 3.3 Aggregation Philosophy

**Individual Events → Spatiotemporal Counts**

The processor operates on two conceptual levels:

1. **Event Level**: Individual pickup/dropoff events with full context (driver, exact location, exact time)
2. **Aggregate Level**: Count of events within spatiotemporal cells

Aggregation **discards**:
- Individual driver identities
- Exact GPS coordinates (only grid cell retained)
- Exact timestamps (only time bucket retained)

Aggregation **preserves**:
- Total event counts per cell
- Spatiotemporal distribution patterns
- Pickup/dropoff balance

**Rationale**: For imitation learning and demand prediction, the aggregate pattern (where/when events occur) is more important than individual driver identities.

### 3.4 Temporal Resolution Trade-offs

**5-minute time buckets** (288 per day) provide:
- ✓ Sufficient resolution to capture rush hour dynamics
- ✓ Manageable data size (~6.6M keys for dense dataset)
- ✓ Balance between detail and statistical stability

**Alternatives considered**:
- 1-minute buckets (1440/day): Too sparse, many zero counts
- 15-minute buckets (96/day): Too coarse, misses short-term variations
- 30-minute buckets (48/day): Used in some literature, but loses peak detail

### 3.5 Sunday Exclusion Rationale

Sundays are excluded by default because:
1. **Different travel patterns**: Leisure vs. commute behavior
2. **Lower sample size**: Fewer trips on Sundays in dataset
3. **Domain convention**: Many transportation studies focus on weekday behavior
4. **Optional**: Can be re-included via configuration flag

## 4. Output Characteristics

### 4.1 Key Space

**Dimensions**:
- x_grid: 48 unique values (latitude dimension)
- y_grid: 90 unique values (longitude dimension)
- time: 288 unique values (5-min buckets, 1-indexed)
- day: 6 unique values (Mon-Sat, 1-indexed)

**Theoretical Maximum**: 48 × 90 × 288 × 6 = 7,464,960 possible keys

**Observed Coverage** (sparse):
- Non-zero keys: ~234,000 (~3.1% of maximum)
- Indicates sparse activity: most spatiotemporal cells have no events

### 4.2 Count Distribution

**Pickup Counts per Cell** (observed):
- Mean: ~1.39 pickups per non-zero cell
- Median: 1 pickup
- Max: Several hundred (at high-traffic locations)
- Distribution: Heavy right tail (typical for count data)

**Interpretation**: 
- Most cells have few events (1-5)
- A small number of "hotspot" cells have very high counts
- Reflects realistic urban mobility: concentrated activity at key locations

### 4.3 Validation Metrics

The processor includes validation against an existing reference dataset:

**Metrics Computed**:
1. **Key overlap**: % of generated keys that exist in reference
2. **Exact match rate**: % of matching keys with identical counts
3. **Close match rate**: % of matching keys within 20% difference
4. **Pearson correlation**: Linear correlation of counts on matching keys

**Success Criteria**:
- Correlation > 0.9 indicates excellent agreement
- Exact match rate > 80% indicates consistent counting methodology

## 5. Computational Complexity

### 5.1 Time Complexity

Let:
- N = number of GPS records (~8M)
- K = number of unique spatiotemporal keys (~234K sparse, ~6.6M dense)
- D = number of drivers (50)

**Algorithm stages**:
1. Data loading: O(N)
2. Bounds computation: O(N)
3. Quantization: O(N)
4. Sorting for transitions: O(N log N)
5. Transition detection: O(N)
6. Aggregation: O(N log K) [due to groupby operations]
7. Dense generation: O(K) [if enabled]

**Overall**: O(N log N) dominated by sorting step

**Observed Performance**: ~20 seconds on typical hardware for 8M records

### 5.2 Space Complexity

**Memory Requirements**:
- Raw data: ~8M records × 6 fields ≈ 1.5 GB (in-memory DataFrame)
- Output sparse: ~234K keys × 2 values ≈ 4 MB (pickle file)
- Output dense: ~6.6M keys × 2 values ≈ 100 MB (pickle file)

## 6. Assumptions and Limitations

### 6.1 Assumptions

1. **Passenger indicator reliability**: Assumes driver correctly toggles indicator
2. **GPS accuracy**: Assumes <50m error (typical for consumer GPS)
3. **Temporal continuity**: Assumes no large time gaps in individual trajectories
4. **State persistence**: Assumes passenger_indicator accurately reflects occupancy throughout ride

### 6.2 Limitations

1. **Shared rides not distinguished**: Multiple passengers counted as single pickup/dropoff
2. **Short trips may be missed**: If entire trip within one 5-minute bucket
3. **Grid boundary effects**: Events near cell boundaries may have quantization artifacts
4. **Missing data**: Drivers with GPS outages may have incomplete transition detection

### 6.3 Robustness Considerations

The algorithm is robust to:
- ✓ Variable sampling rates (GPS records not uniformly spaced)
- ✓ Missing records within trajectories (transition computed on available data)
- ✓ Different numbers of records per driver
- ✓ Different temporal coverage per driver

The algorithm is sensitive to:
- ✗ Incorrect passenger_indicator values (directly affects event detection)
- ✗ Timestamp errors (affects temporal quantization and ordering)
- ✗ Extremely sparse data (may miss rare events)

## 7. Research Applications

### 7.1 Imitation Learning

This counting methodology supports imitation learning by:

1. **Expert demonstration aggregation**: Pooling multiple expert trajectories
2. **State-action encoding**: Spatiotemporal cells as state space
3. **Demand estimation**: Pickup counts as ground truth for passenger demand
4. **Policy evaluation**: Comparing agent behavior to expert patterns

### 7.2 Demand Prediction

The output enables:

1. **Temporal demand patterns**: Time-of-day and day-of-week trends
2. **Spatial demand hotspots**: High-pickup locations
3. **Demand forecasting**: Historical patterns for prediction models
4. **Resource allocation**: Fleet positioning based on expected demand

### 7.3 Validation and Benchmarking

The methodology supports:

1. **Cross-dataset comparison**: Comparing multiple cities or time periods
2. **Algorithm evaluation**: Testing pickup detection algorithms
3. **Data quality assessment**: Identifying anomalies or data issues

## 8. References and Further Reading

### 8.1 Related Methodologies

- **Grid-based spatial discretization**: Common in transportation and urban computing
- **Transition detection**: Standard approach in state-based event detection
- **Spatiotemporal aggregation**: Widely used in taxi demand analysis

### 8.2 Validation Strategy

See [`ALIGNMENT_CHANGES.md`](ALIGNMENT_CHANGES.md) for details on:
- Index offset computation for dataset alignment
- Dense vs. sparse output format trade-offs
- Validation metrics and success criteria

### 8.3 Implementation Details

See [`processor.py`](processor.py) for complete implementation.

---

## Citation

When referencing this methodology in research papers, please include:

```
The pickup and dropoff events were detected using transition-based state
change detection on passenger indicator signals. GPS coordinates were
quantized to a 0.01° grid (~1.1 km resolution) using data-driven bounds,
and timestamps were discretized to 5-minute intervals (288 buckets per day).
Events from all drivers were aggregated by spatiotemporal cell (x_grid,
y_grid, time_bucket, day_of_week), resulting in counts representing
collective passenger demand patterns across the taxi fleet.
```

---

**Document Version**: 1.0  
**Last Updated**: January 7, 2026  
**Implementation**: Pickup/Dropoff Counts Processor v1.0
