import pandas as pd

def validate_data(df):
    try:
        # Check if empty
        if df.empty:
            return {"error": "Dataset is empty"}

        # Keep only numeric columns
        df = df.select_dtypes(include=['number'])

        if df.shape[1] == 0:
            return {"error": "No numeric columns found"}

        # Handle missing values (simple version)
        df = df.dropna()

        if df.empty:
            return {"error": "All data removed after cleaning (missing values)"}

        return {
            "dataframe": df,
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    except Exception as e:
        return {"error": str(e)}