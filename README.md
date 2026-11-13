# Task 35 – Regression Model Evaluation (Salary vs Experience)

**Veda Technology · AI & ML Track · Level 1 · Day 35**

## Objective
Evaluate a Linear Regression model using Mean Absolute Error (MAE) and Mean Squared Error (MSE) and explain the results, to learn the basic evaluation metrics for regression models.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

## Dataset
A small **sample (synthetic) Salary vs Experience dataset** created in code for this exercise. It is not real data.

- 20 employees, 2 columns: `experience` (years, 1 to 10.5) and `salary` (₹31,000 to ₹85,000)
- Mean salary: ₹57,200

## Files in this Repository
| File | Description |
|------|-------------|
| `regression_evaluation.ipynb` | Notebook with training, predictions and metrics |
| `salary_experience.csv` | Dataset used |
| `README.md` | Approach, results and interpretation |

## 1. Features and Target
- **Feature (X):** `experience`
- **Target (y):** `salary`

## 2. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
80% of the data (16 rows) was used for training and 20% (4 rows) for testing. `random_state=42` makes the result reproducible. The model is evaluated only on rows it has not seen during training.

## 3. Model and Predictions
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

The fitted model is approximately: **salary = 23,227 + 5,856 × experience**. The model predicts a salary increase of about ₹5,856 for each additional year of experience.

**Predictions on the test set:**

| Experience | Actual salary | Predicted salary | Error |
|------------|---------------|------------------|-------|
| 1.0 | 31,000 | 29,082 | 1,918 |
| 9.5 | 80,000 | 78,857 | 1,143 |
| 8.5 | 74,000 | 73,001 | 999 |
| 1.5 | 34,000 | 32,010 | 1,990 |

## 4. Evaluation Metrics
```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
```

| Metric | Value |
|--------|-------|
| **MAE** | 1,512.37 |
| **MSE** | 2,485,276.71 |
| RMSE | 1,576.48 |

## 5. Interpretation of the Results
- **MAE ₹1,512:** On average, the model's prediction differs from the actual salary by about ₹1,512. Against a mean salary of ₹57,200, that is roughly 2.6%, so the model fits this data well.
- **MSE 2,485,277:** Errors are squared, so the unit becomes ₹², which is hard to read directly. The square root (RMSE, ₹1,576) is back in rupees and easier to interpret.
- **MAE and RMSE are close:** All four errors are of similar size (₹1,000 to ₹2,000), so there is no single very large mistake. A very large error would push the RMSE much higher than the MAE.
- **Limitation:** The test set has only 4 rows, so these metrics are not very stable, and a different split would give different numbers. The dataset is synthetic, so the results demonstrate the method, not real-world salary behaviour.

## 6. MAE vs MSE

| | MAE | MSE |
|---|-----|-----|
| Calculation | Average of absolute errors | Average of squared errors |
| Unit | Same as target (₹) | Squared unit (₹²) |
| Large errors | Treated linearly | Penalized much more |
| Outlier sensitivity | Lower | Higher |

## Key Learnings
- Always evaluate on data the model has not seen.
- MAE is easy to interpret; MSE punishes big mistakes; RMSE brings MSE back to the original unit.
- A small test set gives unreliable metrics, so the sample size must be mentioned with the results.

## Interview Questions – Key Answers
1. **What is MAE?** The average of the absolute differences between predicted and actual values. It is in the same unit as the target.
2. **What is MSE?** The average of the squared differences between predicted and actual values. It penalizes large errors more and is more sensitive to outliers.
3. **Why are evaluation metrics important?** They measure model performance in numbers, allow different models to be compared, and show whether a model is good enough to use.

## How to Run
```bash
pip install pandas scikit-learn jupyter
jupyter notebook regression_evaluation.ipynb
```
