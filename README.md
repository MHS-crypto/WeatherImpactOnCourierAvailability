## Weather Impact on Courier Availability

### Introduction
This repository explores the impact of weather conditions on courier availability. It aims to analyze the relationship between various weather factors such as temperature, relative humidity, precipitation, and the number of courier partners available on a given day.

### Solution Overview
The solution involves several key steps:

1. **Data Preprocessing:**
   - Imputation of missing values: Mean values were computed for temperature and precipitation for each month to account for the significant variation in temperature experienced in Finland.
   - Removal of outliers: Dates with potentially unusual courier availability were omitted to improve model accuracy.
   
2. **Exploratory Data Analysis (EDA):**
   - Correlation Analysis: Heatmaps and correlation plots were used to understand the relationship between weather factors and courier availability.

3. **Model Development:**
   - Two models were developed: Linear Regression and Random Forest Regression.
   - Data was split into training and testing sets (80/20 ratio).
   - Model evaluation metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) score were calculated to assess model performance.

4. **Prediction:**
   - Utilized Google weather data to predict courier partners coming online for specific dates.

### Models Used
1. **Linear Regression:**
   - Simple and interpretable model suitable for understanding the linear relationship between variables.
   - Benefits: Provides insights into the impact of weather conditions on courier availability.
   
2. **Random Forest Regression:**
   - Handles nonlinear relationships and interactions between features, making it robust for complex datasets.
   - Benefits: Captures nonlinear relationships between weather conditions and courier availability.

### Evaluation Metrics
- **Root Mean Squared Error (RMSE):** Measures the average difference between predicted and actual values. Lower RMSE indicates better model performance.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values.
- **R-squared (R2) Score:** Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

### Future Steps
- Collect more data to further improve model accuracy.
- Explore advanced modeling techniques and ensemble methods for better predictions.

### Execution Steps
1. Ensure the required libraries are installed.
2. Update the file path for the dataset (`daily_cp_activity_dataset.csv`).
3. Execute the code to preprocess the data, develop models, and make predictions.
