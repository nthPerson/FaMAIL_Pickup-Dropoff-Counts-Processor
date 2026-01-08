# Dataset Alignment Changes

## Overview
This document describes the changes made to align the generated `pickup_dropoff_counts.pkl` dataset with the existing `latest_volume_pickups.pkl` dataset.

## Changes Made (Jan 7, 2026)

### 1. Index Offsets Applied
Based on the **Key Diagnostics** tab analysis showing optimal offset combination:

- **x_grid offset: +1** - Aligns latitude grid indices with existing dataset
- **y_grid offset: +1** - Aligns longitude grid indices with existing dataset  
- **time offset: +1** - Converts from 0-based (0-287) to 1-based (1-288) indexing

These offsets are applied after quantization in the `process_data()` function.

### 2. Dense Dataset Generation
The generated dataset now matches the existing dataset format by including **all possible grid cell combinations**, not just cells with non-zero counts.

**Before:** 233,829 keys (sparse - only non-zero entries)
**After:** ~6.6M keys (dense - all grid combinations)

This is controlled by the `generate_dense` configuration option (default: `True`).

### 3. Optional Saturday Inclusion
Added `include_saturday_zeros` option to include Saturday (day 6) entries with zero counts, matching the existing dataset's day range (1-6 vs 1-5 present in raw datasets).

## Configuration Options

### New Parameters in `ProcessingConfig`:

```python
x_grid_offset: int = 1      # Add to x_grid indices
y_grid_offset: int = 1      # Add to y_grid indices  
time_offset: int = 1        # Add to time indices
generate_dense: bool = True # Include all grid cells
include_saturday_zeros: bool = False # Add Saturday with zeros
```

### Streamlit UI Controls

New section: **"⚙️ Alignment & Output Options"** in sidebar with:
- x_grid offset (default: 1)
- y_grid offset (default: 1)
- time offset (default: 1)
- Generate Dense Dataset checkbox (default: True)
- Include Saturday with Zeros checkbox (default: False)

## Improvements

With these changes, the generated dataset showed:

1. **Much higher key overlap** with existing dataset (100%)
2. **Higher correlation** between pickup counts on matching keys
3. **Matching index ranges**:
   - x_grid: 1-48 (vs previous 1-42)
   - y_grid: 1-90 (vs previous 0-80)
   - time: 1-288 (vs previous 0-287)
   - day: 1-6 with Saturday option (vs previous 1-5)

## Testing

To verify alignment improvements:

1. Process data with new default settings
2. Go to **"🔬 Deep Analysis"** tab → Load & Analyze Both Datasets
3. Check **"Index Range Comparison"** - all dimensions should now match
4. Go to **"🔍 Key Diagnostics"** tab → Run Offset Search
5. Recommended offsets at (0, 0, 0), indicating high overlap

## Technical Details

### Code Changes

**`processor.py`:**
- Added offset parameters to `ProcessingConfig` dataclass
- Applied offsets after quantization: `x_grid + offset`, `y_grid + offset`, `time + offset`
- Added dense dataset generation using `itertools.product` to create all combinations
- Added optional Saturday inclusion with zero counts

**`app.py`:**
- Added "Alignment & Output Options" section to sidebar
- Updated `ProcessingConfig` instantiation to include new parameters
- Exposed all alignment options in UI with helpful descriptions

### Performance Considerations

Dense dataset generation increases:
- **Memory usage**: ~6.6M entries vs ~230K
- **File size**: Larger pickle file (~100MB vs ~4MB estimated)
- **Processing time**: Additional ~2-5 seconds for dense generation

This is expected and necessary for format compatibility with the existing dataset.

## Rollback

To revert to sparse format:
1. Set "Generate Dense Dataset" to unchecked in UI
2. Or set `generate_dense=False` in config
3. Set offsets to 0 to use original indexing

## Related Files

- `/home/robert/FAMAIL/pickup_dropoff_counts/processor.py` - Core processing logic
- `/home/robert/FAMAIL/pickup_dropoff_counts/app.py` - Streamlit UI
- `/home/robert/FAMAIL/pickup_dropoff_counts/output/pickup_dropoff_counts.pkl` - Generated dataset
- `/home/robert/FAMAIL/source_data/latest_volume_pickups.pkl` - Reference dataset
