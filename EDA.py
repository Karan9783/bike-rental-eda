# ============================================================
# BIKE RENTAL DATA - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. LOAD DATASETS
# ============================================================

day = pd.read_csv("day.csv")
hour = pd.read_csv("hour.csv")


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("Day dataset shape:", day.shape)
print("Hour dataset shape:", hour.shape)

print("\nDay dataset:")
print(day.head())

print("\nHour dataset:")
print(hour.head())


# ============================================================
# 4. DATA QUALITY ASSESSMENT
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY ASSESSMENT")
print("=" * 60)


# ----- Missing values -----

print("\nMissing values in Day dataset:")
print(day.isnull().sum())

print("\nMissing values in Hour dataset:")
print(hour.isnull().sum())


# ----- Duplicate rows -----

print("\nDuplicate rows in Day dataset:", day.duplicated().sum())
print("Duplicate rows in Hour dataset:", hour.duplicated().sum())


# ----- Data types -----

print("\nData types - Day dataset:")
print(day.dtypes)

print("\nData types - Hour dataset:")
print(hour.dtypes)


# ----- Convert date column to datetime -----

day["dteday"] = pd.to_datetime(day["dteday"])
hour["dteday"] = pd.to_datetime(hour["dteday"])

print("\nDay date data type:", day["dteday"].dtype)
print("Hour date data type:", hour["dteday"].dtype)


# ----- Check unique values -----

print("\nSeasons:", sorted(day["season"].unique()))
print("Years:", sorted(day["yr"].unique()))
print("Months:", sorted(day["mnth"].unique()))
print("Hours:", sorted(hour["hr"].unique()))
print("Weather situations:", sorted(day["weathersit"].unique()))


# ----- Check total rental calculation -----

day_check = day["casual"] + day["registered"] == day["cnt"]
hour_check = hour["casual"] + hour["registered"] == hour["cnt"]

print("\nIncorrect total rows in Day dataset:", (~day_check).sum())
print("Incorrect total rows in Hour dataset:", (~hour_check).sum())


# ============================================================
# 5. OUTLIER / DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL VARIABLE DISTRIBUTION")
print("=" * 60)

plt.figure(figsize=(12, 6))

day[
    ["temp", "hum", "windspeed", "casual", "registered", "cnt"]
].boxplot()

plt.title("Boxplot for Numerical Variables")
plt.ylabel("Values")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ============================================================
# 6. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(day.describe())


# ============================================================
# 7. UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)


# ----- Total Bike Rentals -----

plt.figure(figsize=(8, 5))

plt.hist(day["cnt"], bins=20, edgecolor="black")

plt.title("Distribution of Daily Bike Rentals")
plt.xlabel("Total Bike Rentals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

print("\nDaily Bike Rental Statistics:")
print("Mean:", day["cnt"].mean())
print("Median:", day["cnt"].median())
print("Minimum:", day["cnt"].min())
print("Maximum:", day["cnt"].max())
print("Standard Deviation:", day["cnt"].std())


# ----- Casual Users -----

plt.figure(figsize=(8, 5))

plt.hist(day["casual"], bins=20, edgecolor="black")

plt.title("Distribution of Casual Bike Users")
plt.xlabel("Casual Users")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

print("\nCasual Users Statistics:")
print("Mean:", day["casual"].mean())
print("Median:", day["casual"].median())
print("Minimum:", day["casual"].min())
print("Maximum:", day["casual"].max())


# ----- Registered Users -----

plt.figure(figsize=(8, 5))

plt.hist(day["registered"], bins=20, edgecolor="black")

plt.title("Distribution of Registered Bike Users")
plt.xlabel("Registered Users")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

print("\nRegistered Users Statistics:")
print("Mean:", day["registered"].mean())
print("Median:", day["registered"].median())
print("Minimum:", day["registered"].min())
print("Maximum:", day["registered"].max())


# ----- Temperature -----

plt.figure(figsize=(8, 5))

plt.hist(day["temp"], bins=20, edgecolor="black")

plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ----- Humidity -----

plt.figure(figsize=(8, 5))

plt.hist(day["hum"], bins=20, edgecolor="black")

plt.title("Distribution of Humidity")
plt.xlabel("Humidity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ----- Windspeed -----

plt.figure(figsize=(8, 5))

plt.hist(day["windspeed"], bins=20, edgecolor="black")

plt.title("Distribution of Windspeed")
plt.xlabel("Windspeed")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# 8. BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)


# ----- Temperature vs Total Rentals -----

plt.figure(figsize=(8, 5))

plt.scatter(day["temp"], day["cnt"])

plt.title("Temperature vs Total Bike Rentals")
plt.xlabel("Temperature")
plt.ylabel("Total Bike Rentals")

plt.tight_layout()
plt.show()

temp_corr = day["temp"].corr(day["cnt"])

print("\nCorrelation between Temperature and Bike Rentals:",
      temp_corr)


# ----- Humidity vs Total Rentals -----

plt.figure(figsize=(8, 5))

plt.scatter(day["hum"], day["cnt"])

plt.title("Humidity vs Total Bike Rentals")
plt.xlabel("Humidity")
plt.ylabel("Total Bike Rentals")

plt.tight_layout()
plt.show()

hum_corr = day["hum"].corr(day["cnt"])

print("Correlation between Humidity and Bike Rentals:",
      hum_corr)


# ----- Windspeed vs Total Rentals -----

plt.figure(figsize=(8, 5))

plt.scatter(day["windspeed"], day["cnt"])

plt.title("Windspeed vs Total Bike Rentals")
plt.xlabel("Windspeed")
plt.ylabel("Total Bike Rentals")

plt.tight_layout()
plt.show()

windspeed_corr = day["windspeed"].corr(day["cnt"])

print("Correlation between Windspeed and Bike Rentals:",
      windspeed_corr)


# ----- Weather Situation vs Total Rentals -----

weather_groups = [
    day[day["weathersit"] == 1]["cnt"],
    day[day["weathersit"] == 2]["cnt"],
    day[day["weathersit"] == 3]["cnt"],
    day[day["weathersit"] == 4]["cnt"]
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    weather_groups,
    tick_labels=[
        "Clear",
        "Mist/Cloudy",
        "Light Rain/Snow",
        "Heavy Rain/Snow"
    ]
)

plt.title("Weather Situation vs Total Bike Rentals")
plt.xlabel("Weather Situation")
plt.ylabel("Total Bike Rentals")

plt.tight_layout()
plt.show()

print("\nAverage rentals by weather situation:")
print(day.groupby("weathersit")["cnt"].mean())


# ----- Working Day vs Total Rentals -----

working_day_groups = [
    day[day["workingday"] == 0]["cnt"],
    day[day["workingday"] == 1]["cnt"]
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    working_day_groups,
    tick_labels=[
        "Non-Working Day",
        "Working Day"
    ]
)

plt.title("Working Day vs Total Bike Rentals")
plt.xlabel("Day Type")
plt.ylabel("Total Bike Rentals")

plt.tight_layout()
plt.show()

print("\nAverage rentals by working day:")
print(day.groupby("workingday")["cnt"].mean())


# ============================================================
# 9. GROUPING AND AGGREGATION
# ============================================================

print("\n" + "=" * 60)
print("GROUPING AND AGGREGATION")
print("=" * 60)


# ----- Average Rentals by Season -----

season_avg = day.groupby("season")["cnt"].mean()

print("\nAverage Bike Rentals by Season:")
print(season_avg)


# ----- Average Rentals by Month -----

month_avg = day.groupby("mnth")["cnt"].mean()

print("\nAverage Bike Rentals by Month:")
print(month_avg)


# ----- Average Rentals by Year -----

year_avg = day.groupby("yr")["cnt"].mean()

print("\nAverage Bike Rentals by Year:")
print(year_avg)


# ----- Average Rentals by Weather -----

weather_avg = day.groupby("weathersit")["cnt"].mean()

print("\nAverage Bike Rentals by Weather Situation:")
print(weather_avg)


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)


correlation_data = day[
    [
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt"
    ]
]

correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# ----- Correlation Heatmap -----

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix of Bike Rental Variables")

plt.tight_layout()
plt.show()


# ============================================================
# 11. VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)


# ----- 11.1 Average Rentals by Month -----

plt.figure(figsize=(10, 5))

plt.plot(
    month_avg.index,
    month_avg.values,
    marker="o"
)

plt.title("Average Bike Rentals by Month")
plt.xlabel("Month")
plt.ylabel("Average Bike Rentals")

plt.xticks(range(1, 13))
plt.grid()

plt.tight_layout()
plt.show()


# ----- 11.2 Average Rentals by Season -----

plt.figure(figsize=(8, 5))

plt.bar(
    season_avg.index,
    season_avg.values
)

plt.title("Average Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Average Bike Rentals")

plt.xticks([1, 2, 3, 4])

plt.tight_layout()
plt.show()


# ----- 11.3 Average Rentals by Year -----

plt.figure(figsize=(8, 5))

plt.bar(
    ["2011", "2012"],
    year_avg.values
)

plt.title("Average Bike Rentals by Year")
plt.xlabel("Year")
plt.ylabel("Average Bike Rentals")

plt.tight_layout()
plt.show()


# ----- 11.4 Average Rentals by Weather -----

weather_labels = [
    "Clear",
    "Mist/Cloudy",
    "Light Rain/Snow"
]

plt.figure(figsize=(8, 5))

plt.bar(
    weather_labels,
    weather_avg.values
)

plt.title("Average Bike Rentals by Weather Situation")
plt.xlabel("Weather Situation")
plt.ylabel("Average Bike Rentals")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# ----- 11.5 Temperature vs Bike Rentals -----

plt.figure(figsize=(8, 5))

plt.scatter(
    day["temp"],
    day["cnt"]
)

plt.title("Temperature vs Total Bike Rentals")
plt.xlabel("Temperature")
plt.ylabel("Total Bike Rentals")

plt.tight_layout()
plt.show()


# ----- 11.6 Correlation Heatmap -----

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix of Bike Rental Variables")

plt.tight_layout()
plt.show()


# ============================================================
# 12. FINAL ANALYSIS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL ANALYSIS SUMMARY")
print("=" * 60)

print("\nAverage daily bike rentals:",
      round(day["cnt"].mean(), 2))

print("Highest average season:",
      season_avg.idxmax(),
      "with",
      round(season_avg.max(), 2),
      "rentals")

print("Highest average month:",
      month_avg.idxmax(),
      "with",
      round(month_avg.max(), 2),
      "rentals")

print("Lowest average month:",
      month_avg.idxmin(),
      "with",
      round(month_avg.min(), 2),
      "rentals")

print("Average rentals in 2011:",
      round(year_avg.loc[0], 2))

print("Average rentals in 2012:",
      round(year_avg.loc[1], 2))

print("Temperature correlation:",
      round(temp_corr, 4))

print("Humidity correlation:",
      round(hum_corr, 4))

print("Windspeed correlation:",
      round(windspeed_corr, 4))


# ============================================================
# END OF PROJECT
# ============================================================

print("\n" + "=" * 60)
print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

