import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



# Load the CSV data
file_path = "FilePath\daily_cp_activity_dataset.csv"
data = pd.read_csv(file_path)


# Step 1: Imputing the missing values in the Temperature and Precipitation Column

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract month from the date column
data['month'] = data['date'].dt.month

# Compute the mean temperature for each month
mean_temp_by_month = data.groupby('month')['temperature'].mean()
mean_precp_by_month = data.groupby('month')['precipitation'].mean()

# Fill in missing temperature values with the mean temperature of the corresponding month
for index, row in data.iterrows():
    if pd.isnull(row['temperature']):
        month = row['month']
        mean_temp = mean_temp_by_month[month]
        data.at[index, 'temperature'] = mean_temp

    elif pd.isnull(row['precipitation']):
        month = row['month']
        mean_prep = mean_precp_by_month[month]
        data.at[index, 'precipitation'] = mean_prep
 


# Step 2: Finding the impact of weather conditions on couriers availability

# Corelation Heat Map
heat_map_data = data.rename(columns={
    'date': 'date',
    'courier_partners_online': 'partners',
    'temperature': 'temp',
    'relative_humidity': 'hum',
    'precipitation': 'pre'
})

# Calculate correlation matrix
corr = heat_map_data.corr()

# Plot heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Weather-Courier Correlation Heatmap')
plt.show()

# Correlation between temperature and courier efficiency
plt.figure(figsize=(8, 12))
sns.scatterplot(data=data, x='temperature', y='courier_partners_online')
plt.title('Correlation between Temperature and Courier Availability')
plt.xlabel('Temperature')
plt.ylabel('Courier Partners Online')
plt.show()


# Distribution of courier availability across different days or hours
data['day_of_week'] = data['date'].dt.day_name()

# Define the order of the days of the week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 5))

# Plot average courier availability by day
sns.barplot(data=data.groupby('day_of_week')['courier_partners_online'].mean().reset_index(), x='day_of_week', y='courier_partners_online', ax=ax, order=day_order)

# Set title and labels
ax.set(title='Average Courier Availability by Day', xlabel='Day of Week', ylabel='Average Courier Partners Online')

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()



# Step 3: Removing outliers from the data
outlier_dates = ['5/2/2021', '9/16/2021', '1/25/2022', '3/23/2022', '1/26/2023']

# Convert outlier dates to datetime format
outlier_dates = pd.to_datetime(outlier_dates, format='%m/%d/%Y')

# Filter out outlier dates from the dataset
filtered_data = data[~data['date'].isin(outlier_dates)]

# Reset the index 
filtered_data.reset_index(drop=True, inplace=True)


# Step 4: Model Development
data['date'] = pd.to_datetime(data['date'])


# Split the data into features and target variable
X = filtered_data[['temperature', 'relative_humidity', 'precipitation']]
y = filtered_data['courier_partners_online']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training, 20% testing


# Initialize and train the Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Predict the values for the testing set
y_pred = linear_reg.predict(X_test)

# Calculate the Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) and R- Squared Metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



print("Root Mean Squared Error (RMSE) for Linear Regression:", rmse)
print("Mean Absolute Error (MAE) for Linear Regression:", mae)
print("R-squared (R2) Score for Linear Regression:", r2)
# Root Mean Squared Error (RMSE) for Linear Regression: 7.505565542392949
# Mean Absolute Error (MAE) for Linear Regression: 5.693876474280064
# R-squared (R2) Score for Linear Regression: 0.32232193845158796


# Approach 2: Using the Random Forest Model
model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Root Mean Squared Error for Random Forest Regressor:", rmse)
print("Mean Absolute Error (MAE) for Random Forest Regressor:", mae)
print("R-squared (R2) Score for Random Forest Regressor:", r2)
# Root Mean Squared Error for Random Forest Regressor: 7.629369734847237
# Mean Absolute Error (MAE) for Random Forest Regressor: 5.897944305281472
# R-squared (R2) Score for Random Forest Regressor: 0.299780973554559

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
print("")
print("Feature Importance:")
print(feature_importance)

"""
Feature Importance:
temperature          0.492674
relative_humidity    0.253425
precipitation        0.253901
"""


# Step 5: Prediction of Courier Partners for the upcoming days
# Load the new data
new_data = pd.DataFrame({
    'date': ['01/30/2024', '01/31/2024', '02/01/2024', '02/02/2024', '02/03/2024', '02/04/2024'],
    'temperature': [0, 1, 1, -1, 1, -2],
    'relative_humidity': [0.89, 0.93, 0.75, 0.84, 0.76, 0.79],
    'precipitation': [0, 0.05, 0.35, 0, 0.50, 0]
})

# Convert date column to datetime format
new_data['date'] = pd.to_datetime(new_data['date'], format='%m/%d/%Y')


# Predict with RandomForestRegressor
predictions_rf = model.predict(new_data[['temperature', 'relative_humidity', 'precipitation']])

# Predict with Linear Regression
predictions_lr = linear_reg.predict(new_data[['temperature', 'relative_humidity', 'precipitation']])

new_data['Couriers_Predicted_LR'] = predictions_lr
new_data['Couriers_Predicted_RF'] = predictions_rf


# Print the predictions
print()
print(new_data)

"""

        date  temperature  relative_humidity  precipitation  Couriers_Predicted_LR  Couriers_Predicted_RF
0 2024-01-30            0               0.89           0.00              58.960435              63.116199
1 2024-01-31            1               0.93           0.05              59.530217              62.713709
2 2024-02-01            1               0.75           0.35              58.287239              61.927979
3 2024-02-02           -1               0.84           0.00              58.273508              62.602967
4 2024-02-03            1               0.76           0.50              58.115841              61.407785
5 2024-02-04           -2               0.79           0.00              57.586581              63.616557

"""