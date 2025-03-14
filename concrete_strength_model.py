import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import seaborn as sns
from scipy.stats import levene
import joblib

# -------------------------------
# 1. Data Loading
# -------------------------------

df = pd.read_csv('Concrete_Data.csv')

#rename the data for easy analysis later
rename_dict = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "flyash",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarseagg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fineagg",
    "Age (day)": "age",
    "Concrete compressive strength(MPa, megapascals) ": "strength"
}
df = df.rename(columns=rename_dict)

# count the data number in each discrete aging 
age_counts = df['age'].value_counts().sort_index()
print("====Count of samples for each Age====")
print(age_counts,end="\n")



# Check columns
print("Columns in the DataFrame:", df.columns.tolist())

# -------------------------------
# 2. Define Features & Target
# -------------------------------

features = ['cement', 'slag', 'flyash', 'water',
            'superplasticizer', 'coarseagg', 'fineagg', 'age']
target = 'strength'

# Create X and y
X = df[features]
y = df[target]

# -------------------------------
# 3. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Baseline XGBoost (No Constraints)
# -------------------------------
baseline_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2_baseline = r2_score(y_test, y_pred_baseline)

print("\n=== Baseline XGBoost ===")
print(f"MAE  = {mae_baseline:.3f}")
print(f"RMSE = {rmse_baseline:.3f}")
print(f"R^2  = {r2_baseline:.3f}")

# -------------------------------
# 5. Monotonic XGBoost
# -------------------------------
# We assume 'age' is the 8th feature in the 'features' list. 
# So the monotonic tuple is 7 zeros + 1 for age.
monotone_constraints = (0, 0, 0, 0, 0, 0, 0, 1) #set the 8th feature, age to be monotonic constraint


mono_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    monotone_constraints=monotone_constraints
)
mono_model.fit(X_train, y_train)
joblib.dump(mono_model, 'monotonic_model.pkl')

y_pred_mono = mono_model.predict(X_test)

mae_mono = mean_absolute_error(y_test, y_pred_mono)
rmse_mono = np.sqrt(mean_squared_error(y_test, y_pred_mono))
r2_mono = r2_score(y_test, y_pred_mono)

print("\n=== Monotonic XGBoost ===")
print(f"MAE  = {mae_mono:.3f}")
print(f"RMSE = {rmse_mono:.3f}")
print(f"R^2  = {r2_mono:.3f}")

# -------------------------------
# 6. Uncertainty (Multi-run:5 runs, random seeds from 100-104)
# if the result shows the approximately the same MAE, RMSE, and R^2, it shows the model is stable.
# -------------------------------
n_runs = 5
preds_array = [] # store the prediction for each run

for seed in range(100, 100 + n_runs): #run the seed fronm 100-104
    temp_model = xgb.XGBRegressor(      # create a new temporary model (temp_model) to check for each random seed 
        objective='reg:squarederror',
        random_state=seed,
        monotone_constraints=monotone_constraints #use themonotonic constraints defined above in line 86
    )
    temp_model.fit(X_train, y_train)
    preds_array.append(temp_model.predict(X_test))

# Convert list of predictions into a NumPy array
preds_array = np.array(preds_array)  # shape (n_runs, n_test_samples)
# Calculate the mean and standard deviation of predictions across runs
pred_mean = np.mean(preds_array, axis=0)
pred_std = np.std(preds_array, axis=0) 


mae_mean = mean_absolute_error(y_test, pred_mean)
rmse_mean = np.sqrt(mean_squared_error(y_test, pred_mean))
r2_mean = r2_score(y_test, pred_mean)

print("\n=== Monotonic XGBoost (Multi-run Mean) ===")
print(f"MAE  = {mae_mean:.3f}")
print(f"RMSE = {rmse_mean:.3f}")
print(f"R^2  = {r2_mean:.3f}")
print(f"Avg. Std Dev = {np.mean(pred_std):.3f} ",end='\n')


# -------------------------------
# 7.Hyperparameter Tuning using gridsearchcv
# -------------------------------


# Prepare the model with baseline parameters and monotonic constraint for "age"
model_base = xgb.XGBRegressor(objective='reg:squarederror',
                              monotone_constraints=monotone_constraints,
                              random_state=42)

# define the parameter range to the grid search
param_grid = {
    'max_depth': [3, 5, 7], #try different tree depth
    'learning_rate': [0.01, 0.1, 0.2] # Try different learning rates
}

# Define a simple scoring function based on mean absolute error (MAE)
def mae_score(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# search the grid with 3-fold cross-validation, meaning splitting 2/3 for training and 1/3 for testing
grid_search = GridSearchCV(
    estimator=model_base,
    param_grid=param_grid,
    scoring=make_scorer(mae_score, greater_is_better=False),  # the lower thw MAE,the better
    cv=3,  # 3fold
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("=== Hyperparameter Tuning using gridsearchcv ===")
print("Best Parameters:", grid_search.best_params_)
print("Best Score (MAE):", -grid_search.best_score_)

# Train a final model with the best parameters from the grid search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Compute the Mean Absolute Error (MAE) and R² score for the final predictions (y_pred) from the best-tuned model.
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
print("Test MAE after tuning:", mae_test)
print("Test R² after tuning:", r2_test)

# -------------------------------
# 8. Visualization
# -------------------------------

#---data distribution----
# Create two subplots side by side (one for Baseline vs. True, one for Monotonic vs. True)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Subplot A: Baseline vs. True
axs[0].scatter(X_test['age'], y_test, color='black', label='True', alpha=0.7, s=15)
axs[0].scatter(X_test['age'], y_pred_baseline, color='blue', label='Baseline', alpha=0.7, s=15)
axs[0].set_title("Baseline vs. True")
axs[0].set_xlabel("Age (days)")
axs[0].set_ylabel("Concrete Strength (MPa)")
axs[0].legend()

# Subplot B: Monotonic vs. True
axs[1].scatter(X_test['age'], y_test, color='black', label='True', alpha=0.7, s=15)
axs[1].scatter(X_test['age'], y_pred_mono, color='red', label='Monotonic', alpha=0.7, s=15)
axs[1].set_title("Monotonic vs. True")
axs[1].set_xlabel("Age (days)")
axs[1].set_ylabel("Concrete Strength (MPa)")
axs[1].legend()

plt.tight_layout()
plt.show()

#----error bar plot grouped by age---
#  create a new dataframe,inclduding age, true_value , baseline_value, and monotonic_value
df_group = X_test.copy()  
df_group['strength_true'] = y_test.values
df_group['strength_base'] = y_pred_baseline
df_group['strength_mono'] = y_pred_mono

#  Group the data by 'age' for true_value , baseline_value, and monotonic_value 
# and calculate 'the mean and standard deviation' for true, baseline, and monotonic predictions.
grouped_true = df_group.groupby('age')['strength_true'].agg(['mean','std']).reset_index() #group the strength_true by age,and calculate the mean and std in each age
grouped_base = df_group.groupby('age')['strength_base'].agg(['mean','std']).reset_index()
grouped_mono = df_group.groupby('age')['strength_mono'].agg(['mean','std']).reset_index()

# three errorbar (black, blue, red)
plt.figure(figsize=(8,6))

# Black:True value
plt.errorbar(grouped_true['age'], grouped_true['mean'], 
             yerr=grouped_true['std'], 
             fmt='o', color='black', 
             ecolor='black', capsize=3, alpha=0.9,
             label='Mean ± Std True')

## Blue:baseline value(without constraints)

plt.errorbar(grouped_base['age'], grouped_base['mean'], 
             yerr=grouped_base['std'], 
             fmt='o', color='blue', 
             ecolor='blue', capsize=3, alpha=0.7,
             label='Mean ± Std Baseline')

## Red: Monotonic constraints
plt.errorbar(grouped_mono['age'], grouped_mono['mean'], 
             yerr=grouped_mono['std'], 
             fmt='o', color='red', 
             ecolor='red', capsize=3, alpha=0.7,
             label='Mean ± Std Monotonic')

plt.xlabel('Age (days)')
plt.ylabel('Concrete Strength (MPa)')
plt.title('Group-by-Age Comparison: True vs. Baseline vs. Monotonic')
plt.legend()
plt.show()


# -------------------------------
# 8.Levene testing for explain the better fit of XGBOOST at short aging 
#and also boxplot comparing the fluctuation of data at short anf long aging
# -------------------------------



#  Define Age Groups: 0-90 days vs. 91+ days

bins = [0, 90, df['age'].max()]
labels = ['Short-age (0-90 days)', 'High-age (91+ days)']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)

# Check the distribution
print("\nValue counts in each age group:")
print(df['age_group'].value_counts(dropna=False).sort_index())


#  Boxplot for Each Age Group

plt.figure(figsize=(8, 6))
sns.boxplot(x='age_group', y='strength', data=df, palette=['skyblue', 'salmon'])
plt.xlabel('Age Group')
plt.ylabel('Concrete Strength (MPa)')
plt.title('Strength Distribution by Age Group (Cutoff = 90 days)')
plt.show()

# Group Statistics
grouped_stats = df.groupby('age_group')['strength'].agg(['mean', 'std', 'count']).reset_index()
print("\nGroup Statistics (Mean, Std, Count):")
#print(grouped_stats)

# Perform the Levene test to compare the variances between the short-age and high-age groups.

strength_short = df[df['age_group'] == 'Short-age (0-90 days)']['strength']
strength_high = df[df['age_group'] == 'High-age (91+ days)']['strength']

# Conduct the Levene test (the null hypothesis is that both groups have equal variances)
# 'stat' stores the Levene test statistic, which quantifies the difference in variance between the short-age and high-age groups.
# If p-value is less than 0.05 indicates a significant difference in variance between the two groups.
stat, p_value = levene(strength_short, strength_high)
print(f"Levene test statistic: {stat}, p-value: {p_value}")





