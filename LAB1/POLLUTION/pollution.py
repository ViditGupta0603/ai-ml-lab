import pandas as pd
import numpy as np

# =============================================================================
# 1. LOAD AND DESCRIBE DATASET
# =============================================================================
df = pd.read_csv(r"C:/Users/vidit/Downloads/ai-ml-lab-main/ai-ml-lab-main/LAB1/POLLUTION/pollution.csv")

print("="*80)
print("DATASET DESCRIPTION")
print("="*80)

print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nColumn Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nStatistical Summary (Numeric Columns):\n{df.describe()}")

# =============================================================================
# 2. NULL VALUE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("NULL VALUE ANALYSIS")
print("="*80)

null_counts = df.isnull().sum()
print("\nColumns with Null Values:")
print(null_counts[null_counts > 0])

rows_with_nulls = df.isnull().any(axis=1).sum()
print(f"\nTotal Rows with at least one NULL value: {rows_with_nulls}")

# =============================================================================
# 3. CORRELATION ANALYSIS
# =============================================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

print("\n" + "="*80)
print("HIGH CORRELATIONS (> 0.7)")
print("="*80)

high_corr = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr.append({
                'Variable_1': numeric_cols[i],
                'Variable_2': numeric_cols[j],
                'Correlation': corr_val
            })

high_corr_df = pd.DataFrame(high_corr).sort_values(
    'Correlation', ascending=False
)

print(high_corr_df)

corr_matrix.to_csv('dataset_correlation_matrix.csv')

# =============================================================================
# 4. OUTLIER DETECTION (IQR METHOD)
# =============================================================================
print("\n" + "="*80)
print("OUTLIER DETECTION (IQR METHOD)")
print("="*80)

outlier_summary = []

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    if len(outliers) > 0:
        outlier_summary.append({
            'Column': col,
            'Outlier_Count': len(outliers),
            'Percentage': f"{(len(outliers) / len(df)) * 100:.2f}%",
            'Lower_Bound': lower,
            'Upper_Bound': upper,
            'Min_Value': df[col].min(),
            'Max_Value': df[col].max()
        })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)

outlier_df.to_csv('dataset_outlier_analysis.csv', index=False)

# =============================================================================
# 5. CHANGE DATA TYPES
# =============================================================================
df_modified = df.copy()

# Convert date-like columns
for col in df_modified.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        df_modified[col] = pd.to_datetime(df_modified[col], errors='coerce')

# Convert object columns with low unique values to category
for col in df_modified.select_dtypes(include='object').columns:
    if df_modified[col].nunique() < 50:
        df_modified[col] = df_modified[col].astype('category')

print("\n" + "="*80)
print("DATA TYPE CHANGES")
print("="*80)
print(df_modified.dtypes)

# =============================================================================
# 6. REGRESSION PROBLEM SETUP (GENERIC)
# =============================================================================
print("\n" + "="*80)
print("REGRESSION PROBLEM FORMULATION")
print("="*80)

if len(numeric_cols) >= 2:
    target = numeric_cols[0]
    features = numeric_cols[1:]

    print(f"Target Variable: {target}")
    print("Feature Variables:")
    for f in features:
        print(f"- {f}")

    print(f"\nTarget Mean: {df[target].mean():.2f}")
    print(f"Target Std Dev: {df[target].std():.2f}")

    reg_df = df[[target] + list(features)].dropna()
    reg_df.to_csv('dataset_regression_data.csv', index=False)
else:
    print("Not enough numeric columns to define a regression problem.")

# =============================================================================
# 7. JOIN WITH ANOTHER DATASET (STRUCTURE ONLY)
# =============================================================================
print("\n" + "="*80)
print("DATASET JOIN (CONCEPTUAL)")
print("="*80)

print("""
This dataset can be joined with another dataset using:
- Location columns (State / District / City)
- Date or Time columns

Example join keys:
- district_name
- city
- date
""")

# =============================================================================
# 8. FILES CREATED
# =============================================================================
print("\n" + "="*80)
print("FILES CREATED")
print("="*80)

print("1. dataset_correlation_matrix.csv")
print("2. dataset_outlier_analysis.csv")
print("3. dataset_regression_data.csv")
