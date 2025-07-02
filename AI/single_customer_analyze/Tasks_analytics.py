import pandas as pd
import numpy as np
from datetime import datetime

def convert_to_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            original = df[col].copy()
            
            # Enhanced preprocessing
            cleaned = (
                original
                .astype(str)
                .str.split(r'\s*\(.*', n=1).str[0]  # Remove anything after (
                .str.strip()
                .str.replace(r'([+-]\d{2}):(\d{2})$', r'\1\2', regex=True)  # Fix tz format
                .str.replace(r'\b(UTC|GMT)\b', '', regex=True)  # Remove UTC/GMT prefix
                .str.replace(r'\s+', ' ', regex=True)  # Normalize spaces
            )
            
            # List of formats to try (order matters!)
            formats = [
                '%Y-%m-%d %H:%M:%S%z',          # Case: "2025-03-06 13:24:40+0000"
                '%a %b %d %Y %H:%M:%S %z',      # Case: "Fri Jul 26 2024 18:53:00 +0000"
                '%Y-%m-%d %H:%M:%S',            # Fallback for tz-naive
                '%a %b %d %Y %H:%M:%S',         # Fallback for tz-naive
            ]
            
            parsed = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns, UTC]')
            
            # Try each format sequentially
            for fmt in formats:
                mask = parsed.isna()
                if not mask.any():
                    break
                
                # Attempt parsing with current format
                temp = pd.to_datetime(
                    cleaned[mask],
                    format=fmt,
                    errors='coerce',
                    utc=True
                )
                
                # Only keep successful parses
                parsed[mask] = temp.dropna()
            
            df[col] = parsed
            
            # Report failures
            failed = original[parsed.isna()]

                
    return df

def load_and_clean_data(filepath):
    """Load and clean data from CSV file with error handling."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Check required columns
    required_columns = {
        'dueDate', 'createdAt', 'isOriginal', 'repeated', 'alreadyRepeated',
        'status', 'priority', 'representative_name', 'assignedDistributor_name',
        'title', 'description'
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None

    # Convert date columns
    try:
        df = convert_to_datetime(df, ["createdAt"])
        df = convert_to_datetime(df, ["dueDate"])
        #df['dueDate'] = pd.to_datetime(df['dueDate'], errors='coerce', utc=True)
        #df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce', utc=True)
    except Exception as e:
        print(f"Error converting date columns: {e}")
        return None

    # Convert boolean columns
    bool_cols = ['isOriginal', 'repeated', 'alreadyRepeated']
    for col in bool_cols:
        try:
            df[col] = df[col].apply(lambda x: str(x).lower().strip() == 'true')
        except Exception as e:
            print(f"Error converting boolean column {col}: {e}")
            return None

    return df

def calculate_metrics(df, now_utc):
    """Calculate all metrics from cleaned data."""
    metrics = {}
    
    try:
        # Key metrics
        metrics['total_tasks'] = len(df)
        metrics['pending_tasks'] = df[df['status'] == 'PENDING'].shape[0]
        metrics['completed_tasks'] = df[df['status'] == 'COMPLETED'].shape[0]
        metrics['overdue_tasks'] = df[(df['status'] == 'PENDING') & 
                                    (df['dueDate'] < now_utc)].shape[0]
        metrics['tasks_due_today'] = df[(df['status'] == 'PENDING') & 
                                       (df['dueDate'].dt.date == now_utc.date())].shape[0]
        metrics['upcoming_tasks'] = df[(df['status'] == 'PENDING') & 
                                      (df['dueDate'].dt.date > now_utc.date())].shape[0]
        metrics['tasks_without_due'] = df[(df['status'] == 'PENDING') & 
                                        df['dueDate'].isna()].shape[0]

        # Status breakdown
        metrics['status_counts'] = df['status'].value_counts().to_dict()

        # Priority analysis
        pending_df = df[df['status'] == 'PENDING']
        metrics['priority_counts'] = pending_df['priority'].value_counts().to_dict()

        # Assignment analysis
        metrics['rep_counts'] = df['representative_name'].fillna('Unassigned').value_counts().to_dict()
        metrics['dist_counts'] = df['assignedDistributor_name'].fillna('Unassigned').value_counts().to_dict()

        # Repetition analysis
        metrics['original_tasks'] = df['isOriginal'].sum()
        metrics['repeated_tasks'] = df['repeated'].sum()

    except KeyError as e:
        print(f"Missing expected column: {e}")
        return None
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

    return metrics

def generate_pending_tasks_table(df, now_utc):
    """Generate formatted pending tasks table with due status."""
    try:
        pending_df = df[df['status'] == 'PENDING'].copy()
        
        def get_due_status(row):
            if pd.isna(row['dueDate']):
                return 'No Due Date'
            elif row['dueDate'] < now_utc:
                return 'Overdue'
            elif row['dueDate'].date() == now_utc.date():
                return 'Due Today'
            else:
                return 'Upcoming'

        pending_df['Due Status'] = pending_df.apply(get_due_status, axis=1)
        status_order = {'Overdue': 0, 'Due Today': 1, 'Upcoming': 2, 'No Due Date': 3}
        pending_df['status_order'] = pending_df['Due Status'].map(status_order)
        pending_df = pending_df.sort_values(by=['status_order', 'dueDate'])

        # Formatting functions
        def format_task_details(row):
            title = row['title']
            description = row['description'] if pd.notna(row['description']) else ''
            return f"**{title}**: {description}" if description else f"**{title}**"

        def format_assignee(row):
            rep = row['representative_name'] if pd.notna(row['representative_name']) else None
            dist = row['assignedDistributor_name'] if pd.notna(row['assignedDistributor_name']) else None
            return " | ".join(filter(None, [rep, f"Dist: {dist}" if dist else None])) or 'Unassigned'

        # Generate table lines
        table = [
            "## Pending Tasks Details",
            "| Due Status | Task Details | Due Date | Priority | Assignee |",
            "|------------|--------------|----------|----------|----------|"
        ]
        
        for _, row in pending_df.iterrows():
            table.append(
                f"| {row['Due Status']} | "
                f"{format_task_details(row)} | "
                f"{row['dueDate'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['dueDate']) else 'N/A'} | "
                f"{row['priority']} | "
                f"{format_assignee(row)} |"
            )
        return table
        
    except Exception as e:
        print(f"Error generating pending tasks table: {e}")
        return []

def generate_report(metrics, pending_table):
    """Generate markdown report from calculated metrics."""
    try:
        report = ["---\n## Key Metrics"]
        report.extend([
            f"- **Total Tasks:** {metrics['total_tasks']}",
            f"- **Pending Tasks:** {metrics['pending_tasks']}",
            f"- **Completed Tasks:** {metrics['completed_tasks']}",
            f"- **Overdue Tasks:** {metrics['overdue_tasks']}",
            f"- **Tasks Due Today:** {metrics['tasks_due_today']}",
            f"- **Upcoming Tasks:** {metrics['upcoming_tasks']}",
            f"- **Tasks without Due Date:** {metrics['tasks_without_due']}",
            "---\n## Task Status Breakdown"
        ])
        
        for status, count in metrics['status_counts'].items():
            report.append(f"- **{status}:** {count}")
            
        report.extend([
            "---\n## Priority Analysis (Pending Tasks)",
            *[f"- **{priority}:** {count}" for priority, count in metrics['priority_counts'].items()],
            "---\n## Assignment Analysis\n### Tasks by Representative",
            *[f"- **{rep}:** {count}" for rep, count in metrics['rep_counts'].items()],
            "### Tasks by Distributor",
            *[f"- **{dist}:** {count}" for dist, count in metrics['dist_counts'].items()],
            "---\n## Repetition Analysis",
            f"- **Original Tasks:** {metrics['original_tasks']}",
            f"- **Repeated Tasks:** {metrics['repeated_tasks']}",
            "---"
        ])
        
        report.extend(pending_table)
        return "\n".join(report)
        
    except KeyError as e:
        print(f"Missing expected metric: {e}")
        return "Error generating report"
    except Exception as e:
        print(f"Error generating report: {e}")
        report = "# Business Activities Analysis Report\n\nSorry, we were unable to analyze the activity data due to an issue with calculating the metrics."
        return report

def tasks_report(file_path_tasks):
    """Main function to execute the reporting workflow."""
    try:
        df = load_and_clean_data(file_path_tasks)
        if df is None or df.empty:
            print("No data available for reporting tasks")
            report = "# Business Activities Analysis Report\n\nSorry, we were unable to analyze the activity data due to an issue with calculating the metrics."
            return report

        now_utc = pd.Timestamp.now(tz='UTC')
        metrics = calculate_metrics(df, now_utc)
        pending_table = generate_pending_tasks_table(df, now_utc)
        
        if metrics is None:
            print("Failed to calculate metrics")
            return

        return generate_report(metrics, pending_table)
        
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")

