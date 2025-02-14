import pandas as pd
import osmnx
import geopandas
import contextily as cx
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import esda
import folium
import libpysal.weights as weights

def drop_missing_data(df):
    """
    Drops rows with missing data in specific columns and removes certain columns entirely.
    Additionally, converts all string data to lowercase and interpolates missing values
    in the 'Weather_Timestamp' column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame after dropping and interpolating data.
    """
    df.dropna(subset=['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'City', 'Description', 'Weather_Condition'], how='any', inplace=True)
    df = df.drop(labels=['Street', 'Turning_Loop', 'Country', 'Wind_Chill(F)', 'Airport_Code', 'Wind_Direction'], axis=1)
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df = df.sort_values(by=['End_Time'])
    df.Weather_Timestamp.interpolate(method='ffill', inplace=True)
    return df

def fill_missing_data(df):
    """
    Fills missing numerical data with the mean and missing string data with the mode.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame with filled missing values.
    """
    number_data = df.select_dtypes(include=[np.number])
    for i in number_data.columns.tolist():
        df[i].fillna(df[i].mean(), inplace=True)

    string_data = df.select_dtypes(include='string')
    for i in string_data.columns.tolist():
        df[i].fillna(df[i].mode()[0], inplace=True)

    return df

def drop_columns(df):
    """
    Drops specific columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame after dropping specified columns.
    """
    df.drop([
        'NUMBER OF PEDESTRIANS INJURED',
        'NUMBER OF PEDESTRIANS KILLED',
        'NUMBER OF CYCLIST INJURED',
        'NUMBER OF CYCLIST KILLED',
        'NUMBER OF MOTORIST INJURED',
        'NUMBER OF MOTORIST KILLED',
        'CONTRIBUTING FACTOR VEHICLE 2',
        'CONTRIBUTING FACTOR VEHICLE 3',
        'CONTRIBUTING FACTOR VEHICLE 4',
        'CONTRIBUTING FACTOR VEHICLE 5',
        'VEHICLE TYPE CODE 1',
        'VEHICLE TYPE CODE 2',
        'VEHICLE TYPE CODE 3',
        'VEHICLE TYPE CODE 4',
        'VEHICLE TYPE CODE 5',
        'CROSS STREET NAME',
        'OFF STREET NAME'
    ], axis=1, inplace=True)
    return df

def get_years(df, column):
    """
    Extracts the year from a date column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Column name from which to extract the year.

    Returns:
    pd.Series: Series containing the extracted years.
    """
    return df[column].apply(lambda date: date[0:4])

def get_months(df, column):
    """
    Extracts the month from a date column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Column name from which to extract the month.

    Returns:
    pd.Series: Series containing the extracted months.
    """
    return df[column].apply(lambda date: date[5:7])

def get_hours(df, column):
    """
    Extracts the hour from a date column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Column name from which to extract the hour.

    Returns:
    pd.Series: Series containing the extracted hours.
    """
    return df[column].apply(lambda date: date[11:13])

def divide_timestamps(df):
    """
    Divides timestamps into separate columns for year, month, and hour for 'Start_Time', 'End_Time', and 'Weather_Timestamp'.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame with additional columns for year, month, and hour.
    """
    df['Start_Time_Month'] = get_months(df, 'Start_Time')
    df['Start_Time_Year'] = get_years(df, 'Start_Time')
    df['Start_Time_Hour'] = get_hours(df, 'Start_Time')

    df['End_Time_Month'] = get_months(df, 'End_Time')
    df['End_Time_Year'] = get_years(df, 'End_Time')
    df['End_Time_Hour'] = get_hours(df, 'End_Time')

    df['Weather_Timestamp_Month'] = get_months(df, 'Weather_Timestamp')
    df['Weather_Timestamp_Year'] = get_years(df, 'Weather_Timestamp')
    df['Weather_Timestamp_Hour'] = get_hours(df, 'Weather_Timestamp')

    return df

def correct_timestamps(crash_data_gdf):
    """
    Corrects timestamps in the crash data GeoDataFrame by converting date and time columns to datetime objects 
    and extracting year, month, and hour components.

    Parameters:
    crash_data_gdf (gpd.GeoDataFrame): GeoDataFrame containing crash data.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame with corrected timestamps and additional columns for year, month, and hour.
    """
    crash_data_gdf["CRASH DATE"] = pd.to_datetime(crash_data_gdf["CRASH DATE"])
    crash_data_gdf["CRASH DATE_YEAR"] = crash_data_gdf["CRASH DATE"].dt.year
    crash_data_gdf["CRASH DATE_MONTH"] = crash_data_gdf["CRASH DATE"].dt.month
    crash_data_gdf["CRASH TIME"] = pd.to_datetime(crash_data_gdf["CRASH TIME"])
    crash_data_gdf["CRASH TIME_HOUR"] = crash_data_gdf["CRASH TIME"].dt.hour
    return crash_data_gdf

def points_in_NY(df):
    """
    Filters points in the DataFrame to only include those within the boundaries of New York City.

    Parameters:
    df (gpd.GeoDataFrame): GeoDataFrame containing point data.

    Returns:
    gpd.GeoDataFrame: Filtered GeoDataFrame with points only within New York City.
    """
    nyc = osmnx.geocode_to_gdf("New York City, New York, USA")
    nyc_geom = nyc.geometry.iloc[0]
    df = df[df.intersects(nyc_geom)]
    return df

def temporal_analysis(df):
    """
    Performs a temporal analysis of crash data by plotting the number of accidents per year, per month, and per hour.

    Parameters:
    df (pd.DataFrame): DataFrame containing the crash data.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="CRASH DATE_YEAR")
    plt.title("Number of accidents per year")
    plt.xlabel("Year")
    plt.ylabel("Number of accidents")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="CRASH DATE_MONTH")
    plt.title("Number of accidents per month")
    plt.xlabel("Month")
    plt.ylabel("Number of accidents")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="CRASH TIME_HOUR")
    plt.title("Number of accidents per hour")
    plt.xlabel("Hour")
    plt.ylabel("Number of accidents")
    plt.show()

def contributing_factors_plot(df):
    """
    Plots the most common contributing factors for accidents.

    Parameters:
    df (pd.DataFrame): DataFrame containing the crash data.

    Returns:
    None
    """
    contributing_factors = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
    contributing_factors = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()
    contributing_factors = contributing_factors[contributing_factors.index != 'Unspecified']
    contributing_factors = contributing_factors[contributing_factors > 1000]
    plt.figure(figsize=(10, 5))
    contributing_factors.plot(kind='bar')
    plt.title('Contributing Factors')
    plt.xlabel('Contributing Factor')
    plt.ylabel('Number of Accidents')

    plt.show()

def hexbinning(df,crs):
    """
    Plots a hexbin map of the crash data.

    Parameters:
    df (gpd.GeoDataFrame): GeoDataFrame containing crash data.
    crs (str): Coordinate Reference System (CRS) for the plot.

    Returns:
    None
    """
    f, ax = plt.subplots(1, figsize=(15, 15))
    hexbinn = ax.hexbin(
        x = df["LONGITUDE"],
        y = df["LATITUDE"],
        gridsize=100,
        cmap="hot",
        linewidths=0,
        alpha = 0.8
    )

    cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron)
    plt.colorbar(hexbinn, label="Number of Crashes")
    plt.show()

def kdeplot(df):
    """
    Plots a kernel density estimate (KDE) of the crash data.

    Parameters:
    df (gpd.GeoDataFrame): GeoDataFrame containing crash data.

    Returns:
    None
    """
    f, ax = plt.subplots(1, figsize=(10, 10))

    sns.kdeplot(
        x = df["LONGITUDE"],
        y = df["LATITUDE"],
        cmap="hot",
        shade=True,
        alpha=0.5,
        n_levels=50,    
        ax=ax
    )

    cx.add_basemap(ax, crs=df.crs, source=cx.providers.CartoDB.Positron)
    plt.show()

def make_grid(df, size):
    """
    Creates a grid over New York City and calculates crash statistics for each grid cell.

    Parameters:
    df (gpd.GeoDataFrame): GeoDataFrame containing crash data.
    size (float): Size of the grid cells.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame with grid cells and crash statistics.
    """
    nyc_bbox = osmnx.geocode_to_gdf("New York City, New York, USA").total_bounds
    grid = geopandas.GeoDataFrame(
        geometry=[
            geometry.box(x, y, x + size, y + size) 
            for y in np.arange(nyc_bbox[1], nyc_bbox[3], size)  
            for x in np.arange(nyc_bbox[0], nyc_bbox[2], size) 
        ],
        crs="EPSG:4326"
    )

    grid["crash_count"] = 0
    grid["persons_injured"] = 0
    grid["persons_killed"] = 0
    grid["street_names"] = [[] for _ in range(len(grid))]
    grid["contributing_factors"] = [[] for _ in range(len(grid))]

    # Calculate values for each cell
    for idx, cell in enumerate(grid.geometry):
        cell_crashes = df[df.within(cell)]
        grid.at[idx, "crash_count"] = len(cell_crashes)
        grid.at[idx, "persons_injured"] = cell_crashes["NUMBER OF PERSONS INJURED"].sum()
        grid.at[idx, "persons_killed"] = cell_crashes["NUMBER OF PERSONS KILLED"].sum()
        grid.at[idx, "contributing_factors"] = cell_crashes["CONTRIBUTING FACTOR VEHICLE 1"].tolist()
        grid.at[idx, "street_names"] = cell_crashes["ON STREET NAME"].tolist()

    return grid

def plot_grid(grid,crs):
    """
    Plots the grid GeoDataFrame with a choropleth map based on crash counts.

    Parameters:
    grid (gpd.GeoDataFrame): GeoDataFrame representing the grid over the study area.
    crs (str): Coordinate Reference System (CRS) for the plot.

    Returns:
    None
    """   
    f, ax = plt.subplots(1, figsize=(10, 10))

    grid.plot(
        column="crash_count",
        scheme="FisherJenks",
        k=5,
        cmap="Reds",
        linewidth=0.0,
        alpha=0.75,
        legend=True,
        ax=ax
    )

    cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron)
    plt.show()

def plot_lag(grid,crs):
    """
    Plots two choropleth maps side by side for crash count and crash count lag.

    Parameters:
    grid (gpd.GeoDataFrame): GeoDataFrame containing grid cells with crash statistics.
    crs (str): Coordinate Reference System (CRS) for the plot.

    Returns:
    None
    """    
    f, axs = plt.subplots(1, 2, figsize=(20, 10))
    ax1, ax2 = axs

    grid.plot(
        column="crash_count",
        scheme="FisherJenks",
        k=5,
        cmap="Reds",
        linewidth=0.0,
        alpha=0.75,
        legend=True,
        ax=ax1
    )
    ax1.set_axis_off()
    ax1.set_title("Crash Count")
    cx.add_basemap(ax1, crs=crs, source=cx.providers.CartoDB.Positron)

    grid.plot(
        column="crash_count_lag",
        scheme="FisherJenks",
        k=5,
        cmap="Reds",
        linewidth=0.0,
        alpha=0.75,
        legend=True,
        ax=ax2
    )
    ax2.set_axis_off()
    ax2.set_title("Crash Count Lag")
    cx.add_basemap(ax2, crs=crs, source=cx.providers.CartoDB.Positron)

    plt.show()

def plot_HH_LL(grid,crs):
    """
    Plots a choropleth map categorizing grid cells into quadrants based on crash counts.

    Parameters:
    grid (gpd.GeoDataFrame): GeoDataFrame containing grid cells with crash statistics.
    crs (str): Coordinate Reference System (CRS) for the plot.

    Returns:
    None
    """    
    grid["quadrant"] = "HH"
    grid.loc[grid["crash_count"] < grid["crash_count"].mean(), "quadrant"] = "LH"
    grid.loc[grid["crash_count_lag"] < grid["crash_count_lag"].mean(), "quadrant"] = "HL"
    grid.loc[
        (grid["crash_count"] < grid["crash_count"].mean())
        & (grid["crash_count_lag"] < grid["crash_count_lag"].mean()),
        "quadrant",
    ] = "LL"

    f, ax = plt.subplots(1, figsize=(10, 10))

    grid.plot(
        column="quadrant",
        categorical=True,
        cmap="RdYlBu",
        edgecolor="white",
        linewidth=0.0,
        alpha=0.75,
        legend=True,
        ax=ax
    )

    cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron)
    plt.show()

def moran_scatter(grid,w):
    """
    Generates a Moran scatter plot for spatial autocorrelation analysis.

    Parameters:
    grid (gpd.GeoDataFrame): GeoDataFrame containing grid cells with crash statistics.
    w (libpysal.weights.W): Spatial weights matrix.

    Returns:
    esda.moran.Moran: Moran object containing Moran's I statistic.
    """
    grid["crash_count_std"] = (grid["crash_count"] - grid["crash_count"].mean())
    grid["crash_count_lag_std"] = weights.lag_spatial(
        w, grid["crash_count_std"]
    )
    f, ax = plt.subplots(1, figsize=(10, 10))
    sns.regplot(
        x="crash_count_std",
        y="crash_count_lag_std",
        data=grid,
        ci=None,
        ax=ax,
        line_kws={"color": "red"},
    )
    ax.axvline(0, c="k" , alpha= 0.5)
    ax.axhline(0, c="k", alpha=0.5)

    plt.show()
    w.transform = "R"
    moran = esda.moran.Moran(grid["crash_count"], w)
    print("Moran's I: ", moran.I)
    return moran

def calculate_accident_rates(weather_data,crash_data_gdf):
    """
    Calculates accident rates during rainy and non-rainy weather conditions.

    Parameters:
    weather_data (pd.DataFrame): DataFrame containing weather data.
    crash_data_gdf (gpd.GeoDataFrame): GeoDataFrame containing crash data.

    Returns:
    tuple: A tuple containing various accident rate metrics and DataFrames for rainy and non-rainy hours.
    """
    weather_data['time'] = pd.to_datetime(weather_data['time'])
    crash_data_gdf['CRASH DATE'] = pd.to_datetime(crash_data_gdf['CRASH DATE'])
    crash_data_gdf['CRASH TIME'] = pd.to_datetime(crash_data_gdf['CRASH TIME'])
    crash_data_gdf['CRASH DATETIME'] = crash_data_gdf['CRASH DATE'] + pd.to_timedelta(crash_data_gdf['CRASH TIME'].dt.strftime('%H:%M:%S'))
    crash_data_hourly = crash_data_gdf.resample('H', on='CRASH DATETIME').size().reset_index(name="crash_count")
    merged_data = pd.merge(weather_data, crash_data_hourly, left_on="time", right_on="CRASH DATETIME")
    rainy_df = merged_data[merged_data['precipitation (mm)'] > 0]
    non_rainy_df = merged_data[merged_data['precipitation (mm)'] == 0]
    total_accidents = merged_data['crash_count'].sum()
    total_rainy_hours = len(rainy_df)
    total_non_rainy_hours = len(non_rainy_df)
    total_rainy_accidents = rainy_df['crash_count'].sum()
    total_non_rainy_accidents = non_rainy_df['crash_count'].sum()
    accident_rate_rainy = total_rainy_accidents / total_rainy_hours 
    accident_rate_non_rainy = total_non_rainy_accidents / total_non_rainy_hours 

    return total_accidents, total_rainy_hours, total_non_rainy_hours, total_rainy_accidents, total_non_rainy_accidents, accident_rate_rainy, accident_rate_non_rainy, rainy_df, non_rainy_df

def plot_rainy_nonrainy_pdfs(rainy_df, non_rainy_df):
    """
    Plots probability density functions (PDFs) for crash counts during rainy and non-rainy hours.

    Parameters:
    rainy_df (pd.DataFrame): DataFrame containing crash counts during rainy hours.
    non_rainy_df (pd.DataFrame): DataFrame containing crash counts during non-rainy hours.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, figsize=(10, 5))

    rainy_df['crash_count'].plot.hist(bins=50, alpha=0.5, label="Rainy", ax=ax)
    non_rainy_df['crash_count'].plot.hist(bins=50, alpha=0.5, label="Non-Rainy", ax=ax)

    plt.legend()
    plt.xlabel("Number of Accidents")
    plt.ylabel("Frequency")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(10, 5))

    rainy_df['crash_count'].plot.kde(label="Rainy", ax=ax)
    non_rainy_df['crash_count'].plot.kde(label="Non-Rainy", ax=ax)

    plt.legend()
    plt.xlabel("Number of Accidents")
    plt.ylabel("Density")
    plt.show()

    fig, ax = plt.subplots(1, figsize=(10, 5))
    rainy_df.plot(x="time", y="crash_count", label="Rainy", ax=ax,color='yellow', alpha=0.7)
    non_rainy_df.plot(x="time", y="crash_count", label="Non-Rainy", ax=ax,color='blue',alpha=0.3)

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Accidents")
    plt.show()

def alcohol_analysis(crash_data_gdf):
    """
    Analyzes alcohol-related crashes by weekday and hour.

    Parameters:
    crash_data_gdf (gpd.GeoDataFrame): GeoDataFrame containing crash data.

    Returns:
    tuple: A tuple containing counts of alcohol-related crashes on weekends and weekdays.
    """
    crash_data_gdf['CRASH DATE'] = pd.to_datetime(crash_data_gdf['CRASH DATE'])
    crash_data_gdf['weekday'] = crash_data_gdf['CRASH DATE'].dt.dayofweek

    weekend = crash_data_gdf[crash_data_gdf['weekday'] >= 5]
    weekdays = crash_data_gdf[crash_data_gdf['weekday']<5]

    weekend_alcohol = weekend[weekend['CONTRIBUTING FACTOR VEHICLE 1'] == 'Alcohol Involvement']
    weekdays_alcohol = weekdays[weekdays['CONTRIBUTING FACTOR VEHICLE 1'] == 'Alcohol Involvement']

    weekend_alcohol_count = weekend_alcohol['CONTRIBUTING FACTOR VEHICLE 1'].count()
    weekdays_alcohol_count = weekdays_alcohol['CONTRIBUTING FACTOR VEHICLE 1'].count()
    weekend_alcohol_hourly = weekend_alcohol.resample("H", on="CRASH DATE").size().reset_index(name="crash_count")
    weekdays_alcohol_hourly = weekdays_alcohol.resample("H", on="CRASH DATE").size().reset_index(name="crash_count")

    fig, ax = plt.subplots(1, figsize=(10, 5))

    weekend_alcohol_hourly.plot(x="CRASH DATE", y="crash_count", label="Weekend", ax=ax)
    weekdays_alcohol_hourly.plot(x="CRASH DATE", y="crash_count", label="Weekdays", ax=ax)

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Number of Alcohol-Related Accidents")
    plt.show()

    return weekend_alcohol_count, weekdays_alcohol_count

def find_factos(grid,N):
    """
    Finds the most common contributing factors and street names for top grid areas with crash counts.

    Parameters:
    grid (gpd.GeoDataFrame): GeoDataFrame containing grid cells with crash statistics.
    N (int): Number of top grid areas to analyze.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame containing top grid areas with additional information about contributing factors and street names.
    """
    grid = grid.sort_values("crash_count", ascending=False)

    # Display grid with the highest crash counts
    top_grid = grid.head(N)  

    # Initialize a dictionary to store contributing factors for each top area
    top_area_factors_dict = {}
    top_area_street_names_dict = {}


    for idx, row in top_grid.iterrows():
        factors = row["contributing_factors"]
        street_names = row["street_names"]

        # Find the most common contributing factor
        top_factor = pd.Series(factors).mode()[0]
        if(top_factor == 'Unspecified'):
            #sort the factors and get the second most common factor
            factors_series = pd.Series(factors)
            factors_sorted = factors_series.value_counts().index.tolist()
            if len(factors_sorted) > 1:
                top_factor = factors_sorted[1]
            else:
                top_factor = factors_sorted[0]
            top_area_factors_dict[idx] = top_factor
        else:
            top_area_factors_dict[idx] = top_factor

        # Find the most common street name
        top_street_name = pd.Series(street_names).mode()[0]
        top_area_street_names_dict[idx] = top_street_name

    # Add the most common contributing factor and street name to the top grid
        
    top_grid["top_contributing_factor"] = top_grid.index.map(top_area_factors_dict)
    top_grid["top_street_name"] = top_grid.index.map(top_area_street_names_dict)

    return top_grid

def plot_top_grid(top_grid):
    """
    Plots the top grid areas on a map with different colors based on crash counts.

    Parameters:
    top_grid (gpd.GeoDataFrame): GeoDataFrame containing top grid areas with additional information.

    Returns:
    folium.Map: Folium map object displaying the top grid areas.
    """
    # Create a map centered around New York City
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    # Add grid cells to the map and different colors by crash count
    for idx, row in top_grid.iterrows():
        color = "red"
        if row["crash_count"] < 2200:
            color = "orange"
        if row["crash_count"] < 1900:
            color = "yellow"
        folium.GeoJson(
            row["geometry"],
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": "black",
                "weight": 2,
                "dashArray": "5, 5",
                "fillOpacity": 0.5,
            },
        ).add_to(m)
    # Add street names and crash counts to the map
    for idx, row in top_grid.iterrows():
        folium.Marker(
            location=[row["geometry"].centroid.y, row["geometry"].centroid.x],
            popup=f"Street Name: {row['top_street_name']}<br>Crash Count: {row['crash_count']}<br>Contributing Factor: {row['top_contributing_factor']}",
        ).add_to(m)
    return m

def onehot_encode(df, columns, prefixes):
    """
    One-hot encodes categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    columns (list): List of column names to be one-hot encoded.
    prefixes (list): List of prefixes for the new one-hot encoded columns.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df