"""
Pickup/Dropoff Counts - Streamlit Dashboard

A dashboard for processing raw taxi GPS data and validating pickup/dropoff counts
against existing datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import time

# Import processor module
from processor import (
    ProcessingConfig,
    process_data,
    save_output,
    load_output,
    validate_against_existing,
    load_existing_volume_pickups,
    ProcessingStats,
    ValidationResult
)

# Page configuration
st.set_page_config(
    page_title="Pickup/Dropoff Counts Processor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #00cc66;
    }
</style>
""", unsafe_allow_html=True)


def get_default_paths():
    """Get default file paths based on project structure."""
    base_dir = Path(__file__).parent.parent
    return {
        'raw_data_dir': base_dir / "../" / "raw_data",
        'output_dir': Path(__file__).parent / "output",
        'existing_volume_pickups': base_dir / "../" / "source_data" / "latest_volume_pickups.pkl",
        'sample_volume_pickups': base_dir / "../" / "data" / "dataset_samples" / "latest_volume_pickups.sample.json",
    }


def check_raw_data_files(raw_data_dir: Path) -> dict:
    """Check which raw data files exist."""
    expected_files = [
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl", 
        "taxi_record_09_50drivers.pkl",
    ]
    
    status = {}
    for filename in expected_files:
        filepath = raw_data_dir / filename
        status[filename] = {
            'exists': filepath.exists(),
            'path': filepath,
            'size_mb': filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0
        }
    return status


def main():
    st.title("🚕 Pickup/Dropoff Counts Processor")
    st.markdown("""
    This tool processes raw taxi GPS trajectory data to count pickup and dropoff events 
    for each spatiotemporal key `(x_grid, y_grid, time_bucket, day_of_week)`.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    paths = get_default_paths()
    
    # Path configuration
    with st.sidebar.expander("📁 File Paths", expanded=False):
        # Reset button to clear cached paths
        col1, col2 = st.columns([2, 1])
        with col2:
            if st.button("🔄 Reset", help="Reset to default paths", use_container_width=True):
                for key in ['raw_data_dir_input', 'output_dir_input', 'existing_pickups_input']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        raw_data_dir = st.text_input(
            "Raw Data Directory",
            value=str(paths['raw_data_dir']),
            key="raw_data_dir_input",
            help="Directory containing taxi_record_XX_50drivers.pkl files"
        )
        
        output_dir = st.text_input(
            "Output Directory", 
            value=str(paths['output_dir']),
            key="output_dir_input",
            help="Directory to save processed output"
        )
        
        existing_pickups_path = st.text_input(
            "Existing Volume Pickups Path",
            value=str(paths['existing_volume_pickups']),
            key="existing_pickups_input",
            help="Path to latest_volume_pickups.pkl for validation"
        )
    
    # Processing parameters
    with st.sidebar.expander("🔧 Processing Parameters", expanded=True):
        grid_size = st.number_input(
            "Grid Size (degrees)",
            value=0.01,
            min_value=0.001,
            max_value=0.1,
            step=0.001,
            format="%.3f",
            help="Size of each grid cell in degrees"
        )
        
        time_interval = st.number_input(
            "Time Interval (minutes)",
            value=5,
            min_value=1,
            max_value=60,
            help="Size of time bins in minutes"
        )
        
        exclude_sunday = st.checkbox(
            "Exclude Sundays",
            value=True,
            help="Exclude Sunday records from analysis"
        )
    
    # Alignment and output options
    with st.sidebar.expander("⚙️ Alignment & Output Options", expanded=True):
        st.markdown("**Index Offsets (for alignment with existing dataset)**")
        
        x_offset = st.number_input(
            "x_grid offset",
            value=2,
            min_value=-5,
            max_value=5,
            help="Add this value to x_grid indices (recommended: 2)"
        )
        
        y_offset = st.number_input(
            "y_grid offset",
            value=1,
            min_value=-5,
            max_value=5,
            help="Add this value to y_grid indices (recommended: 1)"
        )
        
        t_offset = st.number_input(
            "time offset",
            value=1,
            min_value=-5,
            max_value=5,
            help="Add this value to time indices (recommended: 1 for 1-based)"
        )
        
        st.divider()
        
        generate_dense = st.checkbox(
            "Generate Dense Dataset",
            value=True,
            help="Include zero entries for all grid cells (matches existing dataset format)"
        )
        
        include_saturday = st.checkbox(
            "Include Saturday with Zeros",
            value=False,
            help="Add Saturday (day 6) entries with zero counts if missing in data"
        )
        
        st.divider()
        st.markdown("**Grid Max Values (for dense generation)**")
        st.caption("Override empirical max values to match existing dataset dimensions")
        
        use_explicit_grid_max = st.checkbox(
            "Use Explicit Grid Max Values",
            value=False,
            help="Enable to set fixed max values instead of using data-derived values"
        )
        
        if use_explicit_grid_max:
            x_grid_max = st.number_input(
                "x_grid Max",
                value=48,
                min_value=1,
                max_value=100,
                help="Maximum x_grid value (existing dataset uses 48)"
            )
            
            y_grid_max = st.number_input(
                "y_grid Max",
                value=90,
                min_value=1,
                max_value=150,
                help="Maximum y_grid value (existing dataset uses 90)"
            )
        else:
            x_grid_max = None
            y_grid_max = None
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Process Data", 
        "✅ Validation", 
        "📈 Visualizations",
        "🔬 Deep Analysis",
        "🔍 Key Diagnostics"
    ])
    
    # =====================
    # TAB 1: Process Data
    # =====================
    with tab1:
        st.header("Process Raw Data")
        
        with st.expander("ℹ️ About Data Processing", expanded=False):
            st.markdown("""
            ### What This Does
            This tool processes raw taxi GPS trajectory data to count pickup and dropoff events 
            for each spatiotemporal cell defined by `(x_grid, y_grid, time_bucket, day_of_week)`.
            
            ### Processing Steps
            1. **Load raw data**: Read pickle files containing GPS records
            2. **Compute bounds**: Determine global lat/lon bounds from all data
            3. **Spatial quantization**: Convert GPS coordinates to grid cells using `numpy.digitize`
            4. **Temporal quantization**: Convert timestamps to 5-minute time buckets (0-287)
            5. **Detect transitions**: Identify pickup (0→1) and dropoff (1→0) events
            6. **Aggregate counts**: Sum events by spatiotemporal key
            
            ### Output Format
            The output is a dictionary with:
            - **Keys**: `(x_grid, y_grid, time, day)` tuples
            - **Values**: `(pickup_count, dropoff_count)` tuples
            
            ### Configuration Options
            - **Grid Size**: Spatial resolution in degrees (default: 0.01° ≈ 1km)
            - **Time Interval**: Temporal resolution in minutes (default: 5 min = 288 buckets/day)
            - **Exclude Sundays**: Remove Sunday data (day=7) from analysis
            """)
        
        # Debug information
        with st.expander("🐛 Debug Info", expanded=False):
            st.code(f"""
Computed default: {paths['raw_data_dir']}
Text input value: {raw_data_dir}
Resolved path: {Path(raw_data_dir).resolve()}
Path exists: {Path(raw_data_dir).exists()}
Session state keys: {list(st.session_state.keys())}
            """)
        
        # Check raw data files
        raw_data_path = Path(raw_data_dir)
        file_status = check_raw_data_files(raw_data_path)
        
        st.subheader("Input Files Status")
        cols = st.columns(3)
        all_files_exist = True
        for i, (filename, status) in enumerate(file_status.items()):
            with cols[i]:
                if status['exists']:
                    st.success(f"✅ {filename}")
                    st.caption(f"Size: {status['size_mb']:.1f} MB")
                else:
                    st.error(f"❌ {filename}")
                    st.caption("File not found")
                    all_files_exist = False
        
        if not all_files_exist:
            st.warning("⚠️ Some raw data files are missing. Please ensure all files are in the raw data directory.")
        
        st.divider()
        
        # Processing section
        output_filename = st.text_input(
            "Output Filename",
            value="pickup_dropoff_counts.pkl",
            help="Name for the output pickle file"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            process_button = st.button(
                "🚀 Start Processing",
                type="primary",
                disabled=not all_files_exist,
                use_container_width=True
            )
        
        if process_button:
            # Create config
            config = ProcessingConfig(
                grid_size=grid_size,
                time_interval=time_interval,
                exclude_sunday=exclude_sunday,
                x_grid_offset=x_offset,
                y_grid_offset=y_offset,
                time_offset=t_offset,
                generate_dense=generate_dense,
                include_saturday_zeros=include_saturday,
                x_grid_max=x_grid_max,
                y_grid_max=y_grid_max,
                raw_data_dir=raw_data_path,
                output_dir=Path(output_dir)
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(stage: str, progress: float):
                progress_bar.progress(progress)
                status_text.text(f"Stage: {stage}")
            
            try:
                with st.spinner("Processing data..."):
                    counts, stats = process_data(config, progress_callback)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Save output
                output_path = Path(output_dir) / output_filename
                save_output(counts, output_path)
                
                # Store in session state for validation
                st.session_state['processed_counts'] = counts
                st.session_state['processing_stats'] = stats
                st.session_state['output_path'] = output_path
                
                # Display statistics
                st.success(f"✅ Successfully processed and saved to `{output_path}`")
                
                st.subheader("📊 Processing Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", f"{stats.total_records:,}")
                with col2:
                    st.metric("After Sunday Filter", f"{stats.records_after_sunday_filter:,}")
                with col3:
                    st.metric("Unique Plates", f"{stats.unique_plates:,}")
                with col4:
                    st.metric("Processing Time", f"{stats.processing_time_seconds:.1f}s")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pickups", f"{stats.total_pickups:,}")
                with col2:
                    st.metric("Total Dropoffs", f"{stats.total_dropoffs:,}")
                with col3:
                    st.metric("Unique Keys", f"{stats.unique_keys:,}")
                with col4:
                    ratio = stats.total_pickups / stats.total_dropoffs if stats.total_dropoffs > 0 else 0
                    st.metric("Pickup/Dropoff Ratio", f"{ratio:.2f}")
                
                # Global bounds info
                if stats.global_bounds:
                    with st.expander("🌍 Global Coordinate Bounds"):
                        bounds = stats.global_bounds
                        st.json({
                            "Latitude": {"min": bounds.lat_min, "max": bounds.lat_max},
                            "Longitude": {"min": bounds.lon_min, "max": bounds.lon_max},
                            "Grid cells (lat)": int((bounds.lat_max - bounds.lat_min) / grid_size),
                            "Grid cells (lon)": int((bounds.lon_max - bounds.lon_min) / grid_size),
                        })
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
    
    # =====================
    # TAB 2: Validation
    # =====================
    with tab2:
        st.header("Validation Against Existing Dataset")
        
        with st.expander("ℹ️ About Validation", expanded=False):
            st.markdown("""
            ### Purpose
            This tab performs a quick validation comparing your generated pickup counts against 
            the existing `latest_volume_pickups.pkl` reference dataset.
            
            ### Metrics Explained
            - **Key Match Rate**: Percentage of generated keys that exist in the reference dataset. 
              Low match rate suggests different coverage or indexing issues.
            - **Exact Match Rate**: Among matching keys, how many have identical pickup counts.
            - **Close Matches (±20%)**: Keys where counts are within 20% of each other.
            - **Correlation**: Pearson correlation coefficient between pickup counts. 
              Values >0.9 indicate excellent agreement, <0.5 indicates significant discrepancies.
            
            ### Interpreting Results
            - **High correlation + Low exact matches** = Consistent patterns but different scales
            - **Low correlation + High key match** = Same coverage but different counting methodology
            - **Low key match** = Different spatial/temporal coverage, check indexing
            
            ### Next Steps
            If validation shows low correlation, use the **Deep Analysis** and **Key Diagnostics** 
            tabs to identify the specific issues.
            """)
        
        existing_path = Path(existing_pickups_path)
        
        # Check for processed data
        has_processed = 'processed_counts' in st.session_state
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Processed data:** {'✅ Available' if has_processed else '❌ Not available - run processing first'}")
        with col2:
            st.info(f"**Existing dataset:** {'✅ Found' if existing_path.exists() else '❌ Not found'}")
        
        if has_processed and existing_path.exists():
            if st.button("🔍 Run Validation", type="primary"):
                with st.spinner("Validating..."):
                    try:
                        result = validate_against_existing(
                            st.session_state['processed_counts'],
                            existing_path
                        )
                        st.session_state['validation_result'] = result
                    except Exception as e:
                        st.error(f"Validation error: {str(e)}")
                        st.exception(e)
        
        # Display validation results
        if 'validation_result' in st.session_state:
            result = st.session_state['validation_result']
            
            st.subheader("📊 Validation Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Existing Keys", f"{result.total_existing_keys:,}")
            with col2:
                st.metric("Generated Keys", f"{result.total_generated_keys:,}")
            with col3:
                st.metric("Matching Keys", f"{result.matching_keys:,}")
            with col4:
                match_rate = result.matching_keys / result.total_existing_keys * 100 if result.total_existing_keys > 0 else 0
                st.metric("Key Match Rate", f"{match_rate:.1f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Exact Pickup Matches", f"{result.exact_pickup_matches:,}")
            with col2:
                exact_rate = result.exact_pickup_matches / result.matching_keys * 100 if result.matching_keys > 0 else 0
                st.metric("Exact Match Rate", f"{exact_rate:.1f}%")
            with col3:
                st.metric("Close Matches (±20%)", f"{result.close_matches:,}")
            with col4:
                st.metric("Correlation", f"{result.correlation:.4f}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Existing Pickups", f"{result.total_existing_pickups:,}")
            with col2:
                st.metric("Total Generated Pickups", f"{result.total_generated_pickups:,}")
            
            # Success/Warning based on results
            if result.correlation > 0.9 and exact_rate > 80:
                st.success("✅ Validation passed! High correlation and match rate.")
            elif result.correlation > 0.7:
                st.warning("⚠️ Moderate correlation. Some discrepancies detected.")
            else:
                st.error("❌ Low correlation. Significant discrepancies detected.")
            
            # Discrepancies table
            st.subheader("📋 Top Discrepancies")
            if len(result.discrepancies) > 0:
                st.dataframe(
                    result.discrepancies[['x_grid', 'y_grid', 'time', 'day', 
                                          'existing_pickup', 'generated_pickup', 
                                          'diff', 'pct_diff']].head(50),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Potential causes
                with st.expander("🔍 Potential Causes of Discrepancies"):
                    st.markdown("""
                    - **Different global bounds:** Data-driven scope mismatch between datasets
                    - **Off-by-one errors:** Bin indexing differences in quantization
                    - **Edge case handling:** Different treatment of boundary conditions
                    - **Data filtering:** Different Sunday/invalid record handling
                    - **Time zone differences:** Timestamp interpretation variations
                    """)
            else:
                st.info("No discrepancies found in matching keys.")
        
        elif not has_processed:
            st.info("👆 Process the raw data first using the 'Process Data' tab.")
        elif not existing_path.exists():
            st.warning(f"⚠️ Existing dataset not found at: `{existing_path}`")
    
    # =====================
    # TAB 3: Visualizations
    # =====================
    with tab3:
        st.header("Data Visualizations")
        
        with st.expander("ℹ️ About Visualizations", expanded=False):
            st.markdown("""
            ### Purpose
            Explore the spatial and temporal patterns in your generated pickup/dropoff data.
            
            ### Available Visualizations
            - **Spatial**: Heatmap showing pickup/dropoff hotspots across the grid
            - **Temporal**: Time-of-day patterns (morning rush, evening rush, etc.)
            - **Daily**: Day-of-week patterns (weekday vs weekend differences)
            - **Distributions**: Statistical distribution of count values
            
            ### What to Look For
            - **Spatial hotspots** should correspond to known busy areas (airports, downtown, stations)
            - **Temporal peaks** typically occur during rush hours (7-9am, 5-7pm)
            - **Daily patterns** may show different behavior on Fridays vs Mondays
            - **Distribution shape** indicates data quality (heavy right tail is normal for count data)
            """)
        
        if 'processed_counts' not in st.session_state:
            st.info("👆 Process data first to generate visualizations.")
        else:
            counts = st.session_state['processed_counts']
            
            # Convert to DataFrame for visualization
            df = pd.DataFrame([
                {'x_grid': k[0], 'y_grid': k[1], 'time': k[2], 'day': k[3], 
                 'pickups': v[0], 'dropoffs': v[1]}
                for k, v in counts.items()
            ])
            
            viz_tabs = st.tabs(["🗺️ Spatial", "⏰ Temporal", "📅 Daily", "📊 Distributions"])
            
            # Spatial heatmap
            with viz_tabs[0]:
                st.subheader("Spatial Distribution")
                
                with st.expander("ℹ️ How to Interpret This Chart", expanded=False):
                    st.markdown("""
                    **What this shows:** A heatmap where brighter colors indicate more pickups/dropoffs.
                    
                    **Axes:**
                    - X-axis: y_grid (longitude index)
                    - Y-axis: x_grid (latitude index)
                    
                    **Interpretation:**
                    - Bright spots = high activity areas (likely downtown, airports, transit hubs)
                    - Dark areas = low activity (residential or rural areas)
                    - Pattern shape should match known city geography
                    """)
                
                metric = st.radio(
                    "Select Metric",
                    ["Pickups", "Dropoffs", "Total (Pickups + Dropoffs)"],
                    horizontal=True
                )
                
                if metric == "Pickups":
                    agg_col = 'pickups'
                elif metric == "Dropoffs":
                    agg_col = 'dropoffs'
                else:
                    df['total'] = df['pickups'] + df['dropoffs']
                    agg_col = 'total'
                
                spatial_agg = df.groupby(['x_grid', 'y_grid'])[agg_col].sum().reset_index()
                
                fig = px.density_heatmap(
                    spatial_agg,
                    x='y_grid',
                    y='x_grid',
                    z=agg_col,
                    color_continuous_scale='Viridis',
                    title=f"Spatial Heatmap: {metric}",
                    labels={'x_grid': 'Latitude Grid', 'y_grid': 'Longitude Grid', agg_col: 'Count'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Temporal patterns
            with viz_tabs[1]:
                st.subheader("Temporal Patterns (Time of Day)")
                
                with st.expander("ℹ️ How to Interpret This Chart", expanded=False):
                    st.markdown("""
                    **What this shows:** Pickup and dropoff counts aggregated by time of day.
                    
                    **X-axis:** Hour of day (0-24)
                    **Y-axis:** Total count across all grid cells
                    
                    **Expected patterns:**
                    - Morning rush: Peak around 7-9am (commuters going to work)
                    - Evening rush: Peak around 5-7pm (commuters going home)
                    - Late night: Lower activity, possible bar/club pickups around 2am
                    - Early morning dip: Lowest activity around 4-5am
                    
                    **Pickup vs Dropoff patterns:**
                    - Pickups often lead dropoffs slightly (people picked up, then dropped off)
                    - Different patterns suggest different trip types (short vs long trips)
                    """)
                
                time_agg = df.groupby('time').agg({
                    'pickups': 'sum',
                    'dropoffs': 'sum'
                }).reset_index()
                
                # Convert time buckets to hours
                time_agg['hour'] = time_agg['time'] * 5 / 60
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=time_agg['hour'], y=time_agg['pickups'],
                              name='Pickups', mode='lines', line=dict(color='blue')),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=time_agg['hour'], y=time_agg['dropoffs'],
                              name='Dropoffs', mode='lines', line=dict(color='red')),
                    secondary_y=False
                )
                
                fig.update_layout(
                    title="Pickups and Dropoffs by Time of Day",
                    xaxis_title="Hour of Day",
                    height=500
                )
                fig.update_yaxes(title_text="Count", secondary_y=False)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Daily patterns
            with viz_tabs[2]:
                st.subheader("Daily Patterns (Day of Week)")
                
                with st.expander("ℹ️ How to Interpret This Chart", expanded=False):
                    st.markdown("""
                    **What this shows:** Total pickups and dropoffs by day of week.
                    
                    **Expected patterns:**
                    - **Weekdays (Mon-Fri)**: Higher business activity, commuter patterns
                    - **Friday**: Often highest as it includes both work and leisure trips
                    - **Saturday**: Different pattern - leisure, shopping, nightlife
                    - **Sunday**: Typically lowest (excluded by default in this processor)
                    
                    **Analysis tips:**
                    - Compare relative heights between days
                    - Large variations may indicate data quality issues for specific days
                    - Missing days suggest filtering or data gaps
                    """)
                
                day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 
                            4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
                
                day_agg = df.groupby('day').agg({
                    'pickups': 'sum',
                    'dropoffs': 'sum'
                }).reset_index()
                day_agg['day_name'] = day_agg['day'].map(day_names)
                
                fig = go.Figure(data=[
                    go.Bar(name='Pickups', x=day_agg['day_name'], y=day_agg['pickups']),
                    go.Bar(name='Dropoffs', x=day_agg['day_name'], y=day_agg['dropoffs'])
                ])
                
                fig.update_layout(
                    barmode='group',
                    title="Pickups and Dropoffs by Day of Week",
                    xaxis_title="Day of Week",
                    yaxis_title="Count",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Distributions
            with viz_tabs[3]:
                st.subheader("Count Distributions")
                
                with st.expander("ℹ️ How to Interpret These Charts", expanded=False):
                    st.markdown("""
                    **What this shows:** Histogram of how many grid cells have each count value.
                    
                    **Expected shape:**
                    - **Right-skewed (long tail)**: Most cells have low counts, few have very high counts
                    - This is normal for count data following a Poisson or negative binomial distribution
                    
                    **What to look for:**
                    - **Extreme outliers**: Very high counts may indicate data errors or special locations
                    - **Gaps in distribution**: May suggest data processing issues
                    - **Similar shapes for pickups/dropoffs**: Expected since they're related
                    
                    **Summary statistics:**
                    - **Mean >> Median**: Confirms right-skewed distribution
                    - **High max values**: Indicates hotspot locations
                    """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df[df['pickups'] > 0],
                        x='pickups',
                        nbins=50,
                        title="Distribution of Pickup Counts per Cell",
                        labels={'pickups': 'Pickup Count', 'count': 'Frequency'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        df[df['dropoffs'] > 0],
                        x='dropoffs',
                        nbins=50,
                        title="Distribution of Dropoff Counts per Cell",
                        labels={'dropoffs': 'Dropoff Count', 'count': 'Frequency'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                
                stats_df = pd.DataFrame({
                    'Metric': ['Pickups', 'Dropoffs'],
                    'Total': [df['pickups'].sum(), df['dropoffs'].sum()],
                    'Mean per Cell': [df['pickups'].mean(), df['dropoffs'].mean()],
                    'Median per Cell': [df['pickups'].median(), df['dropoffs'].median()],
                    'Max per Cell': [df['pickups'].max(), df['dropoffs'].max()],
                    'Non-zero Cells': [(df['pickups'] > 0).sum(), (df['dropoffs'] > 0).sum()]
                })
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # =====================
    # TAB 4: Deep Analysis
    # =====================
    with tab4:
        st.header("🔬 Deep Comparative Analysis")
        
        with st.expander("ℹ️ About This Analysis", expanded=False):
            st.markdown("""
            ### Purpose
            This tab provides comprehensive side-by-side comparison of the **generated dataset** 
            (`pickup_dropoff_counts.pkl`) and the **existing dataset** (`latest_volume_pickups.pkl`).
            
            ### What to Look For
            - **Key Count Differences**: If the datasets have vastly different numbers of keys, 
              this suggests different coverage or filtering criteria
            - **Index Range Differences**: Different min/max values for x_grid, y_grid, time, or day 
              indicate different quantization schemes or indexing conventions (0-based vs 1-based)
            - **Value Distribution Differences**: Different total counts or distributions suggest 
              different data sources, time periods, or counting methodologies
            - **Sparse vs Dense**: If one dataset has many more keys but similar totals, 
              it may include zero-count entries while the other is sparse
            
            ### How to Interpret Results
            - **Low correlation** often indicates indexing misalignment rather than bad data
            - **Matching key counts** but **different values** suggest same spatial coverage but different counting
            - **Different index ranges** are a red flag for off-by-one errors or different grid definitions
            """)
        
        existing_path = Path(existing_pickups_path)
        generated_path = Path(output_dir) / "pickup_dropoff_counts.pkl"
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Generated dataset:** {'✅ Found' if generated_path.exists() else '❌ Not found'}")
        with col2:
            st.info(f"**Existing dataset:** {'✅ Found' if existing_path.exists() else '❌ Not found'}")
        
        if generated_path.exists() and existing_path.exists():
            if st.button("🔬 Load & Analyze Both Datasets", type="primary"):
                with st.spinner("Loading and analyzing datasets..."):
                    # Load both datasets
                    generated = load_output(generated_path)
                    existing = load_existing_volume_pickups(existing_path)
                    
                    # Parse existing keys if needed
                    existing_parsed = {}
                    for key, value in existing.items():
                        if isinstance(key, str):
                            key = tuple(map(int, key.strip('()').split(',')))
                        existing_parsed[key] = value
                    
                    st.session_state['analysis_generated'] = generated
                    st.session_state['analysis_existing'] = existing_parsed
                    st.success("✅ Both datasets loaded successfully!")
        
        if 'analysis_generated' in st.session_state and 'analysis_existing' in st.session_state:
            generated = st.session_state['analysis_generated']
            existing = st.session_state['analysis_existing']
            
            # ==== Section 1: Aggregate Statistics ====
            st.subheader("📊 Aggregate Statistics Comparison")
            
            with st.expander("ℹ️ About Aggregate Statistics", expanded=False):
                st.markdown("""
                **What this shows:** High-level metrics for each dataset including total keys, 
                total pickup/dropoff counts, and basic statistics.
                
                **Key insights:**
                - If **Total Keys** differs dramatically, check if one dataset is sparse (only non-zero values) 
                  vs dense (includes all grid cells even with zero counts)
                - **Total Pickups** difference indicates data volume disparity - could be different time periods 
                  or different detection methodologies
                - Large differences in **Mean/Median** suggest different data distributions
                """)
            
            gen_keys = list(generated.keys())
            exist_keys = list(existing.keys())
            
            gen_pickups = [v[0] for v in generated.values()]
            gen_dropoffs = [v[1] for v in generated.values()]
            exist_pickups = [v[0] if isinstance(v, (list, tuple)) else v for v in existing.values()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Generated Dataset")
                st.metric("Total Keys", f"{len(generated):,}")
                st.metric("Total Pickups", f"{sum(gen_pickups):,}")
                st.metric("Total Dropoffs", f"{sum(gen_dropoffs):,}")
                st.metric("Non-zero Pickup Keys", f"{sum(1 for p in gen_pickups if p > 0):,}")
                st.metric("Mean Pickups/Key", f"{np.mean(gen_pickups):.4f}")
                st.metric("Max Pickups", f"{max(gen_pickups):,}")
            
            with col2:
                st.markdown("#### Existing Dataset")
                st.metric("Total Keys", f"{len(existing):,}")
                st.metric("Total Pickups", f"{sum(exist_pickups):,}")
                st.metric("Total Dropoffs", "N/A (not available)")
                st.metric("Non-zero Pickup Keys", f"{sum(1 for p in exist_pickups if p > 0):,}")
                st.metric("Mean Pickups/Key", f"{np.mean(exist_pickups):.4f}")
                st.metric("Max Pickups", f"{max(exist_pickups):,}")
            
            st.divider()
            
            # ==== Section 2: Index Range Comparison ====
            st.subheader("📐 Index Range Comparison")
            
            with st.expander("ℹ️ About Index Ranges", expanded=False):
                st.markdown("""
                **What this shows:** The min/max values for each dimension (x_grid, y_grid, time, day).
                
                **Why this matters:**
                - **Different ranges indicate different indexing conventions** (0-based vs 1-based)
                - If existing uses 1-288 for time but generated uses 0-287, there's an off-by-one issue
                - Different x_grid/y_grid ranges could mean different spatial bounds or grid sizes
                
                **Common issues detected here:**
                - Time index offset (e.g., 1-based vs 0-based)
                - Spatial grid alignment issues
                - Missing days of week
                """)
            
            gen_x = [k[0] for k in gen_keys]
            gen_y = [k[1] for k in gen_keys]
            gen_t = [k[2] for k in gen_keys]
            gen_d = [k[3] for k in gen_keys]
            
            exist_x = [k[0] for k in exist_keys]
            exist_y = [k[1] for k in exist_keys]
            exist_t = [k[2] for k in exist_keys]
            exist_d = [k[3] for k in exist_keys]
            
            range_data = {
                'Dimension': ['x_grid', 'y_grid', 'time', 'day'],
                'Gen Min': [min(gen_x), min(gen_y), min(gen_t), min(gen_d)],
                'Gen Max': [max(gen_x), max(gen_y), max(gen_t), max(gen_d)],
                'Gen Unique': [len(set(gen_x)), len(set(gen_y)), len(set(gen_t)), len(set(gen_d))],
                'Exist Min': [min(exist_x), min(exist_y), min(exist_t), min(exist_d)],
                'Exist Max': [max(exist_x), max(exist_y), max(exist_t), max(exist_d)],
                'Exist Unique': [len(set(exist_x)), len(set(exist_y)), len(set(exist_t)), len(set(exist_d))],
            }
            
            range_df = pd.DataFrame(range_data)
            range_df['Min Match'] = range_df['Gen Min'] == range_df['Exist Min']
            range_df['Max Match'] = range_df['Gen Max'] == range_df['Exist Max']
            
            st.dataframe(range_df, use_container_width=True, hide_index=True)
            
            # Alert on mismatches
            mismatches = []
            if min(gen_t) != min(exist_t) or max(gen_t) != max(exist_t):
                mismatches.append(f"⚠️ **Time index mismatch**: Generated [{min(gen_t)}-{max(gen_t)}] vs Existing [{min(exist_t)}-{max(exist_t)}]")
            if min(gen_d) != min(exist_d) or max(gen_d) != max(exist_d):
                mismatches.append(f"⚠️ **Day index mismatch**: Generated [{min(gen_d)}-{max(gen_d)}] vs Existing [{min(exist_d)}-{max(exist_d)}]")
            if min(gen_x) != min(exist_x) or max(gen_x) != max(exist_x):
                mismatches.append(f"⚠️ **x_grid mismatch**: Generated [{min(gen_x)}-{max(gen_x)}] vs Existing [{min(exist_x)}-{max(exist_x)}]")
            if min(gen_y) != min(exist_y) or max(gen_y) != max(exist_y):
                mismatches.append(f"⚠️ **y_grid mismatch**: Generated [{min(gen_y)}-{max(gen_y)}] vs Existing [{min(exist_y)}-{max(exist_y)}]")
            
            if mismatches:
                st.warning("### Index Mismatches Detected\n" + "\n".join(mismatches))
            else:
                st.success("✅ All index ranges match!")
            
            st.divider()
            
            # ==== Section 3: Coverage Distribution ====
            st.subheader("📍 Coverage Distribution Comparison")
            
            with st.expander("ℹ️ About Coverage Distribution", expanded=False):
                st.markdown("""
                **What this shows:** Side-by-side histograms showing how pickups are distributed 
                across each dimension (x_grid, y_grid, time, day).
                
                **What to look for:**
                - **Similar shapes** suggest the datasets capture similar patterns
                - **Shifted distributions** indicate indexing offsets
                - **Different peaks** suggest different geographic or temporal focus
                - **Missing bars** in one dataset indicate coverage gaps
                """)
            
            # Convert to DataFrames for plotting
            gen_df = pd.DataFrame([
                {'x_grid': k[0], 'y_grid': k[1], 'time': k[2], 'day': k[3], 'pickups': v[0]}
                for k, v in generated.items()
            ])
            
            exist_df = pd.DataFrame([
                {'x_grid': k[0], 'y_grid': k[1], 'time': k[2], 'day': k[3], 
                 'pickups': v[0] if isinstance(v, (list, tuple)) else v}
                for k, v in existing.items()
            ])
            
            # x_grid distribution
            st.markdown("#### x_grid (Latitude) Distribution")
            col1, col2 = st.columns(2)
            with col1:
                x_agg_gen = gen_df.groupby('x_grid')['pickups'].sum().reset_index()
                fig = px.bar(x_agg_gen, x='x_grid', y='pickups', title="Generated: Pickups by x_grid")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                x_agg_exist = exist_df.groupby('x_grid')['pickups'].sum().reset_index()
                fig = px.bar(x_agg_exist, x='x_grid', y='pickups', title="Existing: Pickups by x_grid")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # y_grid distribution  
            st.markdown("#### y_grid (Longitude) Distribution")
            col1, col2 = st.columns(2)
            with col1:
                y_agg_gen = gen_df.groupby('y_grid')['pickups'].sum().reset_index()
                fig = px.bar(y_agg_gen, x='y_grid', y='pickups', title="Generated: Pickups by y_grid")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                y_agg_exist = exist_df.groupby('y_grid')['pickups'].sum().reset_index()
                fig = px.bar(y_agg_exist, x='y_grid', y='pickups', title="Existing: Pickups by y_grid")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Time distribution
            st.markdown("#### Time Distribution (5-min buckets)")
            col1, col2 = st.columns(2)
            with col1:
                t_agg_gen = gen_df.groupby('time')['pickups'].sum().reset_index()
                t_agg_gen['hour'] = t_agg_gen['time'] * 5 / 60
                fig = px.line(t_agg_gen, x='hour', y='pickups', title="Generated: Pickups by Hour")
                fig.update_layout(height=300, xaxis_title="Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                t_agg_exist = exist_df.groupby('time')['pickups'].sum().reset_index()
                t_agg_exist['hour'] = t_agg_exist['time'] * 5 / 60
                fig = px.line(t_agg_exist, x='hour', y='pickups', title="Existing: Pickups by Hour")
                fig.update_layout(height=300, xaxis_title="Hour of Day")
                st.plotly_chart(fig, use_container_width=True)
            
            # Day distribution
            st.markdown("#### Day of Week Distribution")
            day_names = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
            col1, col2 = st.columns(2)
            with col1:
                d_agg_gen = gen_df.groupby('day')['pickups'].sum().reset_index()
                d_agg_gen['day_name'] = d_agg_gen['day'].map(day_names)
                fig = px.bar(d_agg_gen, x='day_name', y='pickups', title="Generated: Pickups by Day")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                d_agg_exist = exist_df.groupby('day')['pickups'].sum().reset_index()
                d_agg_exist['day_name'] = d_agg_exist['day'].map(day_names)
                fig = px.bar(d_agg_exist, x='day_name', y='pickups', title="Existing: Pickups by Day")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # ==== Section 4: Key Overlap Analysis ====
            st.subheader("🔗 Key Overlap Analysis")
            
            with st.expander("ℹ️ About Key Overlap Analysis", expanded=False):
                st.markdown("""
                **What this shows:** Venn diagram-style analysis of which keys appear in each dataset.
                
                **Interpretation:**
                - **High overlap** means datasets cover similar spatiotemporal regions
                - **Many keys only in existing** suggests generated data has narrower scope
                - **Many keys only in generated** suggests different filtering or expanded coverage
                - **Low overlap** often indicates fundamental indexing differences
                """)
            
            gen_key_set = set(generated.keys())
            exist_key_set = set(existing.keys())
            
            overlap = gen_key_set & exist_key_set
            only_gen = gen_key_set - exist_key_set
            only_exist = exist_key_set - gen_key_set
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Keys Only in Generated", f"{len(only_gen):,}")
            with col2:
                st.metric("Keys in Both", f"{len(overlap):,}")
            with col3:
                st.metric("Keys Only in Existing", f"{len(only_exist):,}")
            
            # Calculate overlap percentage
            if len(gen_key_set) > 0:
                gen_overlap_pct = len(overlap) / len(gen_key_set) * 100
            else:
                gen_overlap_pct = 0
            if len(exist_key_set) > 0:
                exist_overlap_pct = len(overlap) / len(exist_key_set) * 100
            else:
                exist_overlap_pct = 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("% of Generated in Existing", f"{gen_overlap_pct:.1f}%")
            with col2:
                st.metric("% of Existing in Generated", f"{exist_overlap_pct:.1f}%")
            
            # Analyze non-overlapping keys
            if len(only_gen) > 0:
                st.markdown("##### Sample Keys Only in Generated")
                sample_only_gen = list(only_gen)[:10]
                st.code(str(sample_only_gen))
            
            if len(only_exist) > 0:
                st.markdown("##### Sample Keys Only in Existing")
                sample_only_exist = list(only_exist)[:10]
                st.code(str(sample_only_exist))
        
        else:
            st.info("👆 Click 'Load & Analyze Both Datasets' to start the analysis.")
    
    # =====================
    # TAB 5: Key Diagnostics
    # =====================
    with tab5:
        st.header("🔍 Key Diagnostics & Alignment Testing")
        
        with st.expander("ℹ️ About Key Diagnostics", expanded=False):
            st.markdown("""
            ### Purpose
            This tab helps diagnose and fix key alignment issues between datasets.
            
            ### Common Problems
            1. **Off-by-one errors**: One dataset uses 0-based indexing, the other uses 1-based
            2. **Different grid definitions**: Different lat/lon bounds or grid sizes
            3. **Time zone differences**: Timestamps interpreted differently
            4. **Different day definitions**: Monday=0 vs Monday=1
            
            ### What This Tool Does
            - Tests different index offset combinations to find optimal alignment
            - Shows correlation improvement when offsets are applied
            - Helps identify the exact transformation needed
            """)
        
        existing_path = Path(existing_pickups_path)
        generated_path = Path(output_dir) / "pickup_dropoff_counts.pkl"
        
        if 'analysis_generated' in st.session_state and 'analysis_existing' in st.session_state:
            generated = st.session_state['analysis_generated']
            existing = st.session_state['analysis_existing']
            
            # ==== Offset Testing ====
            st.subheader("🔧 Index Offset Testing")
            
            with st.expander("ℹ️ About Offset Testing", expanded=False):
                st.markdown("""
                **What this does:** Tests different combinations of index offsets to find the best alignment.
                
                **How to use:**
                1. Click "Run Offset Search" to test all common offset combinations
                2. Look for combinations with highest overlap and correlation
                3. Use the best offset in your data processing configuration
                
                **Example:** If time offset +1 shows best results, it means generated time indices 
                need +1 to match existing (generated uses 0-287, existing uses 1-288).
                """)
            
            if st.button("🔎 Run Offset Search", type="primary"):
                with st.spinner("Testing offset combinations..."):
                    results = []
                    
                    # Test offsets for each dimension
                    offsets_to_test = {
                        'x_offset': [-2, -1, 0, 1, 2],
                        'y_offset': [-2, -1, 0, 1, 2],
                        't_offset': [-1, 0, 1],
                        'd_offset': [0]  # Day offset usually not an issue
                    }
                    
                    # Quick test: just time offset first
                    progress = st.progress(0)
                    total_tests = len(offsets_to_test['t_offset']) * len(offsets_to_test['x_offset']) * len(offsets_to_test['y_offset'])
                    test_num = 0
                    
                    for t_off in offsets_to_test['t_offset']:
                        for x_off in offsets_to_test['x_offset']:
                            for y_off in offsets_to_test['y_offset']:
                                # Apply offsets to generated keys
                                transformed_gen = {}
                                for key, value in generated.items():
                                    new_key = (
                                        key[0] + x_off,
                                        key[1] + y_off,
                                        key[2] + t_off,
                                        key[3]  # day unchanged
                                    )
                                    transformed_gen[new_key] = value
                                
                                # Calculate overlap
                                overlap = len(set(transformed_gen.keys()) & set(existing.keys()))
                                
                                # Calculate correlation on overlapping keys
                                matching_keys = set(transformed_gen.keys()) & set(existing.keys())
                                if len(matching_keys) > 10:
                                    gen_vals = [transformed_gen[k][0] for k in matching_keys]
                                    exist_vals = [existing[k][0] if isinstance(existing[k], (list, tuple)) else existing[k] for k in matching_keys]
                                    try:
                                        corr = np.corrcoef(gen_vals, exist_vals)[0, 1]
                                    except:
                                        corr = 0
                                else:
                                    corr = 0
                                
                                results.append({
                                    'x_offset': x_off,
                                    'y_offset': y_off,
                                    't_offset': t_off,
                                    'overlap': overlap,
                                    'overlap_pct': overlap / len(generated) * 100 if len(generated) > 0 else 0,
                                    'correlation': corr
                                })
                                
                                test_num += 1
                                progress.progress(test_num / total_tests)
                    
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('overlap', ascending=False)
                    
                    st.session_state['offset_results'] = results_df
                    st.success(f"✅ Tested {len(results)} offset combinations")
            
            if 'offset_results' in st.session_state:
                results_df = st.session_state['offset_results']
                
                st.markdown("#### Top 10 Offset Combinations by Overlap")
                st.dataframe(
                    results_df.head(10).style.format({
                        'overlap': '{:,.0f}',
                        'overlap_pct': '{:.1f}%',
                        'correlation': '{:.4f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Best result analysis
                best = results_df.iloc[0]
                st.markdown(f"""
                #### Best Offset Combination
                - **x_offset**: {int(best['x_offset'])} (add to x_grid)
                - **y_offset**: {int(best['y_offset'])} (add to y_grid)  
                - **t_offset**: {int(best['t_offset'])} (add to time)
                - **Resulting overlap**: {int(best['overlap']):,} keys ({best['overlap_pct']:.1f}%)
                - **Correlation**: {best['correlation']:.4f}
                """)
                
                if best['correlation'] < 0.5:
                    st.warning("""
                    ⚠️ Even with the best offset, correlation is low. This suggests:
                    - Different data sources or time periods
                    - Different pickup detection methodology  
                    - Grid size or bounds mismatch
                    - Consider checking the raw data processing
                    """)
            
            st.divider()
            
            # ==== Manual Key Lookup ====
            st.subheader("🔎 Manual Key Lookup")
            
            with st.expander("ℹ️ About Manual Key Lookup", expanded=False):
                st.markdown("""
                **What this does:** Look up a specific key in both datasets to compare values directly.
                
                **Use cases:**
                - Verify a specific cell's pickup count
                - Check if a known high-activity location matches between datasets
                - Debug specific discrepancies
                """)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                lookup_x = st.number_input("x_grid", min_value=0, max_value=100, value=36)
            with col2:
                lookup_y = st.number_input("y_grid", min_value=0, max_value=100, value=75)
            with col3:
                lookup_t = st.number_input("time", min_value=0, max_value=287, value=161)
            with col4:
                lookup_d = st.number_input("day", min_value=1, max_value=7, value=4)
            
            lookup_key = (lookup_x, lookup_y, lookup_t, lookup_d)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Generated Dataset:**")
                if lookup_key in generated:
                    val = generated[lookup_key]
                    st.success(f"Key found: pickups={val[0]}, dropoffs={val[1]}")
                else:
                    st.error("Key not found")
                    # Try nearby keys
                    nearby = [(k, v) for k, v in generated.items() 
                              if abs(k[0]-lookup_x)<=2 and abs(k[1]-lookup_y)<=2 
                              and abs(k[2]-lookup_t)<=2 and k[3]==lookup_d][:5]
                    if nearby:
                        st.markdown("Nearby keys found:")
                        for k, v in nearby:
                            st.code(f"{k}: pickups={v[0]}, dropoffs={v[1]}")
            
            with col2:
                st.markdown("**Existing Dataset:**")
                if lookup_key in existing:
                    val = existing[lookup_key]
                    if isinstance(val, (list, tuple)):
                        st.success(f"Key found: pickups={val[0]}, volume={val[1] if len(val)>1 else 'N/A'}")
                    else:
                        st.success(f"Key found: value={val}")
                else:
                    st.error("Key not found")
                    # Try nearby keys
                    nearby = [(k, v) for k, v in existing.items() 
                              if abs(k[0]-lookup_x)<=2 and abs(k[1]-lookup_y)<=2 
                              and abs(k[2]-lookup_t)<=2 and k[3]==lookup_d][:5]
                    if nearby:
                        st.markdown("Nearby keys found:")
                        for k, v in nearby:
                            if isinstance(v, (list, tuple)):
                                st.code(f"{k}: pickups={v[0]}")
                            else:
                                st.code(f"{k}: value={v}")
            
            st.divider()
            
            # ==== Data Structure Inspection ====
            st.subheader("🗂️ Data Structure Inspection")
            
            with st.expander("ℹ️ About Data Structure Inspection", expanded=False):
                st.markdown("""
                **What this shows:** Raw sample data from both datasets to verify structure and format.
                
                **Use this to:**
                - Verify key format (tuple vs string)
                - Check value format (tuple vs list vs scalar)
                - Understand what each value field represents
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Generated Dataset Sample:**")
                sample_gen = dict(list(generated.items())[:5])
                for k, v in sample_gen.items():
                    st.code(f"Key: {k}\nValue: {v}\nTypes: key={type(k).__name__}, val={type(v).__name__}")
            
            with col2:
                st.markdown("**Existing Dataset Sample:**")
                sample_exist = dict(list(existing.items())[:5])
                for k, v in sample_exist.items():
                    st.code(f"Key: {k}\nValue: {v}\nTypes: key={type(k).__name__}, val={type(v).__name__}")
        
        else:
            st.info("👆 Load datasets in the 'Deep Analysis' tab first.")
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    ---
    **Pickup/Dropoff Counts Processor**  
    FaMAIL Project - San Diego State University - 2026
    """)


if __name__ == "__main__":
    main()
