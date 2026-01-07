import pandas as pd
from datetime import datetime


def convert_to_datetime(df, columns):
    """Custom datetime conversion function with enhanced parsing"""
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

def load_and_clean_notes_data(filepath):
    """Load and clean notes data with error handling"""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None

    # Check required columns
    required_cols = {'representativeDuplicate_name', 'distributor_name', 
                    'text', 'createdAt', 'updatedAt'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns: {missing}")
        return None, None

    try:
        # Convert datetime columns using custom function
        df = convert_to_datetime(df, ['createdAt', 'updatedAt'])
        
        # Prepare notes table
        notes_table = df[['representativeDuplicate_name', 'distributor_name', 
                        'text', 'createdAt', 'updatedAt']].copy()
        
        # Handle missing values
        notes_table['representativeDuplicate_name'] = notes_table['representativeDuplicate_name'].fillna('Unassigned')
        notes_table['distributor_name'] = notes_table['distributor_name'].fillna('N/A')
        notes_table['text'] = notes_table['text'].fillna('No note')
        
        # Sort by creation date
        notes_table = notes_table.sort_values(by='createdAt', ascending=False)
        
        # Format dates
        notes_table['Created At'] = notes_table['createdAt'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A'
        )
        notes_table['Updated At'] = notes_table['updatedAt'].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else 'N/A'
        )

        # Create final table
        final_table = notes_table.rename(columns={
            'representativeDuplicate_name': 'Representative',
            'distributor_name': 'Distributor',
            'text': 'Note Text'
        })[['Representative', 'Distributor', 'Note Text', 'Created At', 'Updated At']]

        return df, final_table

    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

def calculate_notes_metrics(df, now_utc):
    """Calculate notes metrics with error handling"""
    metrics = {}
    try:
        metrics['total_notes'] = len(df)
        
        # Representative/distributor counts
        metrics['rep_counts'] = df['representativeDuplicate_name'].fillna('Unassigned').value_counts().to_dict()
        metrics['dist_counts'] = df['distributor_name'].fillna('Unassigned').value_counts().to_dict()
        
        # Time-based metrics
        seven_days_ago = now_utc - pd.Timedelta(days=7)
        thirty_days_ago = now_utc - pd.Timedelta(days=30)
        
        metrics['notes_last_7_days'] = df[df['createdAt'].between(seven_days_ago, now_utc)].shape[0]
        metrics['notes_last_30_days'] = df[df['createdAt'].between(thirty_days_ago, now_utc)].shape[0]
        
        return metrics
    
    except KeyError as e:
        print(f"Missing column for metrics calculation: {e}")
        return None
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def generate_notes_report(metrics, final_table):
    """Generate markdown report with error handling"""
    try:
        md = ["---\n## Key Metrics"]
        
        # Key metrics section
        md.append(f"- **Total Notes:** {metrics.get('total_notes', 'N/A')}")
        
        md.append("- **Notes per Representative:**")
        for rep, count in metrics.get('rep_counts', {}).items():
            md.append(f"  - **{rep}:** {count}")
            
        md.append("- **Notes per Distributor:**")
        for dist, count in metrics.get('dist_counts', {}).items():
            md.append(f"  - **{dist}:** {count}")
            
        md.extend([
            f"- **Notes Created in Last 7 Days:** {metrics.get('notes_last_7_days', 'N/A')}",
            f"- **Notes Created in Last 30 Days:** {metrics.get('notes_last_30_days', 'N/A')}",
            "---\n## Notes Details",
            "| Representative | Distributor | Note Text | Created At | Updated At |",
            "|---------------|-------------|-----------|------------|------------|"
        ])
        
        # Add table rows
        if final_table is not None:
            for _, row in final_table.iterrows():
                md.append(
                    f"| {row['Representative']} | {row['Distributor']} | "
                    f"{row['Note Text']} | {row['Created At']} | {row['Updated At']} |"
                )
        else:
            md.append("| Error generating table |")
            
        md.append("---\n")
        
        return '\n'.join(md)
    
    except Exception as e:
        print(f"Error generating report: {e}")
        report = "# Business Activities Analysis Report\n\nSorry, we were unable to analyze the activity data due to an issue with calculating the metrics."
        return report

def notes_report(file_path_notes):
    """Main execution flow"""
    try:
        # Load and process data
        df, final_table = load_and_clean_notes_data(file_path_notes)
        if df is None or df.empty:
            print("No data available for reporting notes")
            report = "# Business Activities Analysis Report\n\nSorry, we were unable to analyze the activity data due to an issue with calculating the metrics."
            return report
        
        # Calculate metrics
        now_utc = pd.Timestamp.now(tz='UTC')
        metrics = calculate_notes_metrics(df, now_utc)
        
        # Generate and print report
        if metrics:
            return generate_notes_report(metrics, final_table)
        else:
            return "no data" 
    except Exception as e:
        print(f"Unexpected error: {e}")
