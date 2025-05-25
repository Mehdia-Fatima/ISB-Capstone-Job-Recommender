# 📊 Complete Streamlit Dashboard: Jobs + Workers + India Map + Region Filter + State Drilldown + KPIs + Downloads

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import requests
import base64
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# Define regions and states
region_map = {
    "North": ["Delhi", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Punjab", "Rajasthan", "Uttarakhand", "Uttar Pradesh"],
    "South": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "West": ["Goa", "Gujarat", "Maharashtra", "Rajasthan"],
    "Central": ["Chhattisgarh", "Madhya Pradesh"],
    "Northeast": ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"]
}

# Load datasets
job_file = 'Data_Innodatatics1 - Data_Innodatatics1.csv'
worker_file = 'daily_wage_workers_5000_enhanced (1).csv'

df = pd.read_csv(job_file)
worker_df = pd.read_csv(worker_file)

# Preprocessing Jobs Data
if 'Min salary' in df.columns and 'Max salary' in df.columns:
    df['Average Salary'] = (df['Min salary'] + df['Max salary']) / 2

# Preprocessing Worker Data
worker_df[['Wage Min', 'Wage Max']] = worker_df['Wages (INR/day)'].str.split('-', expand=True)
worker_df['Wage Min'] = pd.to_numeric(worker_df['Wage Min'], errors='coerce')
worker_df['Expected Monthly Wage'] = worker_df['Wage Min'] * 26

# Sidebar Filters
st.sidebar.title("Filters")
selected_region = st.sidebar.selectbox("Filter by Region", options=["All"] + list(region_map.keys()))
selected_state = st.sidebar.multiselect('Select State(s)', df['State'].dropna().unique())
selected_job_type = st.sidebar.multiselect('Select Job Type(s)', df['Job type'].dropna().unique())
selected_time = st.sidebar.multiselect('Select Job Time', df['Part time/full time'].dropna().unique())

filtered_df = df.copy()
if selected_region != "All":
    filtered_df = filtered_df[filtered_df['State'].isin(region_map[selected_region])]
if selected_state:
    filtered_df = filtered_df[filtered_df['State'].isin(selected_state)]
if selected_job_type:
    filtered_df = filtered_df[filtered_df['Job type'].isin(selected_job_type)]
if selected_time:
    filtered_df = filtered_df[filtered_df['Part time/full time'].isin(selected_time)]

# KPIs
np.random.seed(42)
filtered_df['Distance_from_Worker'] = np.random.choice(range(1, 15), size=len(filtered_df))
jobs_within_2km = filtered_df[filtered_df['Distance_from_Worker'] <= 2].shape[0]
jobs_within_5km = filtered_df[filtered_df['Distance_from_Worker'] <= 5].shape[0]
jobs_within_10km = filtered_df[filtered_df['Distance_from_Worker'] <= 10].shape[0]
avg_distance = round(filtered_df['Distance_from_Worker'].mean(), 2)

matched_jobs = int(0.3 * filtered_df.shape[0])
total_jobs = filtered_df.shape[0]
unique_companies = filtered_df['Company'].nunique()
unique_job_types = filtered_df['Job type'].nunique()
unique_states = filtered_df['State'].nunique()
avg_min_salary = round(filtered_df['Min salary'].mean(), 2)
avg_max_salary = round(filtered_df['Max salary'].mean(), 2)

total_workers = worker_df.shape[0]
unique_skills = worker_df['Skills'].nunique()
unique_locations = worker_df['Location'].nunique()
avg_experience = round(worker_df['Years of Experience'].mean(), 2)
avg_expected_wage = round(worker_df['Expected Monthly Wage'].mean(), 2)

st.title("InnoAI Workforce Dashboard")

st.subheader("National Level Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Job Postings', total_jobs)
    st.metric('Job Matching Ratio', f"{round(matched_jobs/total_jobs, 2)}")
    st.metric('Jobs within 2 km', jobs_within_2km)
with col2:
    st.metric('Unique Companies', unique_companies)
    st.metric('Jobs within 5 km', jobs_within_5km)
    st.metric('Avg Proximity (km)', f"{avg_distance}")
with col3:
    st.metric('Avg Min Salary (₹)', f"{avg_min_salary}")
    st.metric('Jobs within 10 km', jobs_within_10km)
    st.metric('Avg Max Salary (₹)', f"{avg_max_salary}")



col4, col5, col6 = st.columns(3)
with col4:
    st.metric('Total Workers', total_workers)
    st.metric('Unique Skills', unique_skills)
with col5:
    st.metric('Unique Locations', unique_locations)
    st.metric('Avg Experience (Years)', avg_experience)
with col6:
    st.metric('Avg Expected Wage (₹)', avg_expected_wage)

# India GeoJSON for map
@st.cache_data

def load_geojson():
    try:
        with open("india_state.geojson", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load India GeoJSON map: {e}")
        return None

india_geojson = load_geojson()

# 🔍 Inspect a sample of GeoJSON to identify correct property key


if not india_geojson:
    st.stop()

# 🗺️ Pincode-based Worker Location Plot using Latitude/Longitude

# Sample Pincode to Coordinates mapping
pincode_latlong = {
    411001: (18.5196, 73.8553), 600001: (13.0827, 80.2707), 226001: (26.8467, 80.9462),
    500001: (17.3850, 78.4867), 110001: (28.6448, 77.2167), 560001: (12.9716, 77.5946),
    400001: (18.9388, 72.8354), 380001: (23.0225, 72.5714), 700001: (22.5726, 88.3639),
    302001: (26.9124, 75.7873)
}

def map_pincode(pin):
    try:
        return pd.Series(pincode_latlong.get(int(pin), (None, None)))
    except:
        return pd.Series([None, None])

if 'Pincode' in worker_df.columns:
    worker_df[['Latitude', 'Longitude']] = worker_df['Pincode'].apply(map_pincode)
    worker_mapped = worker_df.dropna(subset=['Latitude', 'Longitude'])

    st.subheader("📍 Worker Distribution by Pincode")
    fig_pinmap = px.scatter_geo(
        worker_mapped,
        lat='Latitude',
        color='Skills',
        size='Expected Monthly Wage',
        lon='Longitude',
        scope='asia',
        hover_name='Skills',
        size_max=10,
        opacity=0.5,
        title='Workers Geolocation from Pincode (Sample Based)'
    )
    fig_pinmap.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_pinmap, use_container_width=True)

# 🏙️ Top 5 Cities by Wage Density
if 'Pincode' in worker_df.columns:
    city_map = {
        411001: 'Pune', 600001: 'Chennai', 226001: 'Lucknow',
        500001: 'Hyderabad', 110001: 'Delhi', 560001: 'Bangalore',
        400001: 'Mumbai', 380001: 'Ahmedabad', 700001: 'Kolkata', 302001: 'Jaipur'
    }
    worker_df['City'] = worker_df['Pincode'].map(city_map)
    city_avg_wage = worker_df.groupby('City')['Expected Monthly Wage'].mean().dropna().sort_values(ascending=False).head(5)
    st.subheader("💸 Top 5 Cities by Avg Expected Wage")
    st.bar_chart(city_avg_wage)

# 👥 Top 5 Cities by Worker Count
    city_worker_count = worker_df['City'].value_counts().dropna().head(5)
    st.subheader("🏙️ Top 5 Cities by Worker Count")
    st.bar_chart(city_worker_count)

import geopandas as gpd

# Add a section for Raw Data Tables
st.subheader("📊 Raw Data Tables")

# Create tabs for different datasets
tab1, tab2 = st.tabs(["Jobs Data", "Workers Data"])

with tab1:
    st.subheader("Jobs Dataset")
    # Add a search/filter box for the jobs data
    search_jobs = st.text_input("Search Jobs Data", "")
    if search_jobs:
        filtered_jobs = df[df.astype(str).apply(lambda x: x.str.contains(search_jobs, case=False)).any(axis=1)]
        st.dataframe(filtered_jobs, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
    
    # Add download button for jobs data
    csv_jobs = df.to_csv(index=False)
    st.download_button(
        label="Download Jobs Data as CSV",
        data=csv_jobs,
        file_name="jobs_data.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Workers Dataset")
    # Add a search/filter box for the workers data
    search_workers = st.text_input("Search Workers Data", "")
    if search_workers:
        filtered_workers = worker_df[worker_df.astype(str).apply(lambda x: x.str.contains(search_workers, case=False)).any(axis=1)]
        st.dataframe(filtered_workers, use_container_width=True)
    else:
        st.dataframe(worker_df, use_container_width=True)
    
    # Add download button for workers data
    csv_workers = worker_df.to_csv(index=False)
    st.download_button(
        label="Download Workers Data as CSV",
        data=csv_workers,
        file_name="workers_data.csv",
        mime="text/csv"
    )

# Add some styling for the tables
st.markdown("""
    <style>
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Add a section for Pivot Tables
st.subheader("📊 Pivot Tables")

# Create tabs for different pivot tables
tab1, tab2 = st.tabs(["Jobs Pivot", "Workers Pivot"])

with tab1:
    st.subheader("Jobs Pivot Table")
    
    # Select columns for pivot table
    pivot_columns = st.multiselect(
        "Select columns for Jobs pivot table",
        options=df.columns,
        default=['State', 'Job type', 'Part time/full time']
    )
    
    # Select value column
    value_column = st.selectbox(
        "Select value column for Jobs pivot table",
        options=['Min salary', 'Max salary', 'Average Salary'],
        index=2
    )
    
    # Create pivot table
    if pivot_columns and value_column:
        pivot_df = pd.pivot_table(
            df,
            values=value_column,
            index=pivot_columns[0],
            columns=pivot_columns[1] if len(pivot_columns) > 1 else None,
            aggfunc='mean',
            fill_value=0
        )
        st.dataframe(pivot_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Download button for pivot table
        csv_pivot = pivot_df.to_csv(index=True)
        st.download_button(
            label="Download Jobs Pivot Table as CSV",
            data=csv_pivot,
            file_name="jobs_pivot.csv",
            mime="text/csv"
        )
    else:
        st.write("Please select columns for the pivot table.")

with tab2:
    st.subheader("Workers Pivot Table")
    
    # Select columns for pivot table
    pivot_columns_worker = st.multiselect(
        "Select columns for Workers pivot table",
        options=worker_df.columns,
        default=['Skills', 'Location']
    )
    
    # Select value column
    value_column_worker = st.selectbox(
        "Select value column for Workers pivot table",
        options=['Expected Monthly Wage', 'Years of Experience'],
        index=0
    )
    
    # Create pivot table
    if pivot_columns_worker and value_column_worker:
        pivot_df_worker = pd.pivot_table(
            worker_df,
            values=value_column_worker,
            index=pivot_columns_worker[0],
            columns=pivot_columns_worker[1] if len(pivot_columns_worker) > 1 else None,
            aggfunc='mean',
            fill_value=0
        )
        st.dataframe(pivot_df_worker.style.highlight_max(axis=0), use_container_width=True)
        
        # Download button for pivot table
        csv_pivot_worker = pivot_df_worker.to_csv(index=True)
        st.download_button(
            label="Download Workers Pivot Table as CSV",
            data=csv_pivot_worker,
            file_name="workers_pivot.csv",
            mime="text/csv"
        )
    else:
        st.write("Please select columns for the pivot table.")

# Add some styling for the pivot tables
st.markdown("""
    <style>
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)





# Calculate state-wise metrics first
state_count = filtered_df['State'].value_counts().reset_index()
state_count.columns = ['State', 'Job Count']
state_count['% Share'] = round(100 * state_count['Job Count'] / state_count['Job Count'].sum(), 2)

# Alternative Job Density Visualizations
st.subheader("Alternative Job Density Visualizations")

# Create tabs for different visualizations
viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Treemap", "Bar Chart", "Bubble Chart"])

with viz_tab1:
    st.subheader("📊 Job Distribution Treemap")
    # Treemap
    fig_treemap = px.treemap(
        state_count,
        path=['State'],
        values='Job Count',
        color='% Share',
        color_continuous_scale='Blues',
        title='Job Distribution by State (Treemap)',
        hover_data=['Job Count', '% Share']
    )
    fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    st.plotly_chart(fig_treemap, use_container_width=True)

with viz_tab2:
    st.subheader("📊 Top States by Job Count")
    
    # Sort states by job count
    sorted_states = state_count.sort_values('Job Count', ascending=True)
    
    # Create horizontal bar chart
    fig_bar = px.bar(
        sorted_states,
        x='Job Count',
        y='State',
        orientation='h',
        color='% Share',
        color_continuous_scale='Blues',
        title='Job Distribution by State (Horizontal Bar Chart)',
        text='Job Count'  # Display values on bars
    )
    
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Number of Jobs",
        yaxis_title="State",
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with viz_tab3:
    st.subheader("📊 Job Distribution Bubble Chart")
    
    # Calculate additional metrics for bubble chart
    state_metrics = filtered_df.groupby('State').agg({
        'Min salary': 'mean',  # Using Min salary instead of Average Salary
        'Company': 'nunique'
    }).reset_index()
    
    # Add job count to state_metrics
    state_metrics = state_metrics.merge(state_count[['State', 'Job Count']], on='State', how='left')
    state_metrics['Bubble Size'] = np.sqrt(state_metrics['Job Count'])
    
    # Create bubble chart
    fig_bubble = px.scatter(
        state_metrics,
        x='Company',
        y='Min salary',  # Using Min salary instead of Average Salary
        size='Bubble Size',
        color='Job Count',
        hover_name='State',
        color_continuous_scale='Blues',
        title='Job Distribution by State (Bubble Chart)',
        labels={
            'Company': 'Number of Unique Companies',
            'Min salary': 'Average Minimum Salary (₹)',
            'Job Count': 'Number of Jobs'
        }
    )
    
    fig_bubble.update_layout(
        showlegend=True,
        xaxis_title="Number of Unique Companies",
        yaxis_title="Average Minimum Salary (₹)"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

# Add Region-wise Analysis
st.subheader("📊 Region-wise Job Distribution")

# Calculate region-wise metrics
region_metrics = pd.DataFrame()
for region, states in region_map.items():
    region_data = filtered_df[filtered_df['State'].isin(states)]
    region_metrics = pd.concat([region_metrics, pd.DataFrame({
        'Region': [region],
        'Job Count': [len(region_data)],
        'Avg Min Salary': [region_data['Min salary'].mean()],
        'Companies': [region_data['Company'].nunique()],
        'Job Types': [region_data['Job type'].nunique()]
    })], ignore_index=True)

# Create region comparison chart
fig_region = px.bar(
    region_metrics,
    x='Region',
    y=['Job Count', 'Companies', 'Job Types'],
    title='Region-wise Comparison',
    barmode='group'
)

fig_region.update_layout(
    xaxis_title="Region",
    yaxis_title="Count",
    legend_title="Metrics"
)
st.plotly_chart(fig_region, use_container_width=True)

# Add interactive filters for detailed analysis
st.subheader("🔍 Detailed Analysis")

# Metric selector
selected_metric = st.selectbox(
    "Select Metric for Analysis",
    ["Job Count", "Average Minimum Salary", "Number of Companies"]
)

# Create dynamic visualization based on selection
if selected_metric == "Job Count":
    metric_data = state_count
    y_column = 'Job Count'
elif selected_metric == "Average Minimum Salary":
    metric_data = filtered_df.groupby('State')['Min salary'].mean().reset_index()
    metric_data.columns = ['State', 'Average Minimum Salary']
    y_column = 'Average Minimum Salary'
else:
    metric_data = filtered_df.groupby('State')['Company'].nunique().reset_index()
    metric_data.columns = ['State', 'Number of Companies']
    y_column = 'Number of Companies'

fig_dynamic = px.bar(
    metric_data,
    x='State',
    y=y_column,
    title=f'{selected_metric} by State',
    color=y_column,
    color_continuous_scale='Blues'
)

st.plotly_chart(fig_dynamic, use_container_width=True)
