import pandas as pd
import io
import json


def export_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """Export dataframe to Excel format."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def export_to_json(df: pd.DataFrame, orient: str = "records") -> str:
    """Export dataframe to JSON format."""
    return df.to_json(orient=orient, indent=2)


def export_to_csv(df: pd.DataFrame) -> str:
    """Export dataframe to CSV format."""
    return df.to_csv(index=False)


def export_to_parquet(df: pd.DataFrame) -> bytes:
    """Export dataframe to Parquet format (more efficient)."""
    output = io.BytesIO()
    df.to_parquet(output, index=False)
    return output.getvalue()


def export_to_html(df: pd.DataFrame, title: str = "Data Export") -> str:
    """Export dataframe to HTML format."""
    return df.to_html(index=False, border=0, classes='dataframe')


def get_export_options():
    """Get available export formats."""
    return {
        "CSV": {"extension": "csv", "mime": "text/csv", "func": export_to_csv},
        "Excel": {"extension": "xlsx", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "func": export_to_excel},
        "JSON": {"extension": "json", "mime": "application/json", "func": export_to_json},
        "HTML": {"extension": "html", "mime": "text/html", "func": export_to_html}
    }