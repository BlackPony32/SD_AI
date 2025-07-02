import pandas as pd
from collections import defaultdict
import asyncio
import aiofiles
import logging
import warnings
# Configure logging to write errors to a file
logging.basicConfig(
    level=logging.ERROR,
    filename='error.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define a custom exception for critical data loading errors
class DataLoadError(Exception):
    pass

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

async def load_data(filepath):
    """Load and preprocess the data asynchronously"""
    loop = asyncio.get_running_loop()
    try:
        df = await loop.run_in_executor(None, _load_data_sync, filepath)
        
        # Check for critical conditions
        if df.empty:
            raise DataLoadError("Data is empty")
        
        # Required columns for analysis
        required_columns = ['createdAt', 'type', 'createdBy', 'representativeDuplicate_name']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise DataLoadError(f"Missing required columns: {', '.join(missing)}")
        

        df = convert_to_datetime(df, ["createdAt"])
        df = convert_to_datetime(df, ["updatedAt"])

        activity_names = {
            'CHECKED_IN': 'Check-ins',
            'ORDER_ADDED': 'Order Activities',
            'PHOTO_GROUP_ADDED': 'Photo Groups Added',
            'TASK_ADDED': 'Tasks Added',
            'NOTE_ADDED': 'Notes Added'
        }
        df['activity_type'] = df['type'].map(activity_names).fillna('Unknown')
        df.to_csv('test.csv',index=False)
        return df
    
    except Exception as e:
        logging.error(f"Error in load_data: {e}")
        raise DataLoadError(str(e)) from e

def _load_data_sync(filepath):
    """Synchronous helper to load CSV"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        raise DataLoadError(f"CSV file not found at {filepath}")
    except Exception as e:
        raise DataLoadError(f"Error reading CSV file: {e}")

async def calculate_metrics(df):
    """Calculate metrics asynchronously"""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _calculate_metrics_sync, df)
    except Exception as e:
        logging.error(f"Error in calculate_metrics: {e}")
        raise  # Propagate the error to main for critical handling

def _calculate_metrics_sync(df):
    """Synchronous helper to calculate metrics"""
    metrics = {}
    
    try:
        # 1. Key Metrics
        try:
            metrics['total_activities'] = len(df)
            reps = df['representativeDuplicate_name'].dropna().unique()
            metrics['unique_reps'] = {
                'count': len(reps),
                'names': reps.tolist()
            }
            metrics['total_orders'] = df[df['type'] == 'ORDER_ADDED'].shape[0]
            metrics['rep_activities'] = df[df['createdBy'] == 'REPRESENTATIVE'].shape[0]
            metrics['distributor_activities'] = df[df['createdBy'] == 'DISTRIBUTOR'].shape[0]
        except Exception as e:
            logging.error(f"Error calculating Key Metrics: {e}")
        
        # 2. Activity Type Distribution
        try:
            activity_dist = df['activity_type'].value_counts(normalize=True).mul(100).round(1)
            metrics['activity_distribution'] = {}
            for activity in activity_dist.index:
                activity_df = df[df['activity_type'] == activity]
                top_performers = activity_df['representativeDuplicate_name'].value_counts().head(3)
                top_performers = top_performers if not top_performers.empty else pd.Series({'No representatives': 0})
                metrics['activity_distribution'][activity] = {
                    'percentage': activity_dist[activity],
                    'top_performers': top_performers.to_dict() 
                }
        except Exception as e:
            logging.error(f"Error calculating Activity Type Distribution: {e}")
        
        # 3. Monthly Activity Trends
        try:
            monthly = df.groupby([df['createdAt'].dt.strftime('%Y-%m'), 'activity_type']).size().unstack(fill_value=0)
            metrics['monthly_trends'] = monthly.reset_index().rename(columns={'createdAt': 'month'}).to_dict('records')

            # 4. Activity Source Breakdown
            metrics['source_breakdown'] = df.groupby(['activity_type', 'createdBy']).size().unstack(fill_value=0).to_dict('index')
        except Exception as e:
            logging.error(f"Error calculating Monthly Activity Trends or Activity Source Breakdown: {e}")
            
        # 5. Hourly Pattern (filtered by ORDER_ADDED)
        try:
            # Filter rows where 'type' is ORDER_ADDED
            order_added_df = df[df['type'] == 'ORDER_ADDED']
            
            if not order_added_df.empty:
                # Calculate hourly distribution for filtered data
                hourly = order_added_df['createdAt'].dt.hour.value_counts().sort_index()
                metrics['peak_hours'] = hourly.idxmax()
                metrics['hourly_distribution'] = hourly.to_dict()
            else:
                # Handle case with no ORDER_ADDED entries
                metrics['peak_hours'] = None
                metrics['hourly_distribution'] = {}
                logging.warning("No 'ORDER_ADDED' entries found for hourly pattern analysis.")
                
        except Exception as e:
            logging.error(f"Error calculating Hourly Pattern: {e}")
            metrics['peak_hours'] = None
            metrics['hourly_distribution'] = {}
            
        
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        raise  # Let main handle critical errors

def generate_report(metrics):
    """Generate markdown report (synchronous)"""
    md = []
    
    try:
        try:
            # 1. Key Metrics
            md.append("# Business Activities Analysis Report\n")
            md.append("## Key Metrics")
            md.append(f"- Total Activities: {metrics['total_activities']}")
            md.append(f"- Unique Representatives: {metrics['unique_reps']['count']} : ({', '.join(metrics['unique_reps']['names'])})")
            md.append(f"- Total Orders Created: {metrics['total_orders']}")
            rep_percent = (metrics['rep_activities'] / metrics['total_activities']) * 100
            dist_percent = (metrics['distributor_activities'] / metrics['total_activities']) * 100
            md.append(f"- Representative Activities: {metrics['rep_activities']} ({rep_percent:.1f}%)")
            md.append(f"- Distributor Activities: {metrics['distributor_activities']} ({dist_percent:.1f}%)\n")
        except Exception as e:
            logging.error(f"Error generating Key Metrics: {e}")
        
        # 2. Activity Type Distribution
        try:
            md.append("## Activity Type Distribution")
            md.append("| Activity Type | Percentage | Top Performers |")
            md.append("|---------------|------------|----------------|")
            for activity, data in metrics['activity_distribution'].items():
                performers = [f"{k} ({v})" for k, v in data['top_performers'].items()]
                md.append(f"| {activity} | {data['percentage']}% | {', '.join(performers)} |")
            md.append("\n")
        except Exception as e:
            logging.error(f"Error generating Activity Type Distribution: {e}")
        
        # 3. Monthly Activity Trends
        try:
            md.append("## Monthly Activity Trends")
            headers = ["Month"] + list({col for entry in metrics['monthly_trends'] for col in entry.keys() if col != 'month'})
            md.append("| " + " | ".join(headers) + " |")
            md.append("|" + "|".join(["---"] * len(headers)) + "|")
            for month_data in metrics['monthly_trends']:
                row = [month_data['month']]
                row += [str(month_data.get(col, 0)) for col in headers[1:]]
                md.append("| " + " | ".join(row) + " |")
            md.append("\n")
        except Exception as e:
            logging.error(f"Error generating Monthly Activity Trends: {e}")
        
        # 4. Activity Source Breakdown
        try:
            md.append("## Activity Source Breakdown")
            md.append("| Activity Type | Representative | Distributor |")
            md.append("|---------------|----------------|-------------|")
            for activity in metrics['activity_distribution'].keys():
                rep = metrics['source_breakdown'].get(activity, {}).get('REPRESENTATIVE', 0)
                dist = metrics['source_breakdown'].get(activity, {}).get('DISTRIBUTOR', 0)
                md.append(f"| {activity} | {rep} | {dist} |")
            md.append("\n")
        except Exception as e:
            logging.error(f"Error generating Activity Source Breakdown: {e}")
        
        # 5. Hourly Activity Pattern
        try:
            md.append("## Hourly Activity Pattern")
            peak_hour = f"{metrics['peak_hours']}:00-{metrics['peak_hours'] + 1}:00"
            md.append(f"Most activities occur between {peak_hour} UTC. Hourly distribution shows "
                      f"peak productivity hours align with typical business operations schedule.")

        except Exception as e:
            logging.error(f"Error generating Hourly Activity Pattern: {e}")

        return "\n".join(md)
    except Exception as e:
        logging.error(f"Error generating report: {e}")
        return "# Business Activities Analysis Report\n\nSorry, we were unable to generate the full report due to an unexpected error."

async def analyze_activities(file_path_notes: str, file_path_tasks: str, file_path_activities: str):
    """Main function to orchestrate the analysis and report generation"""
    try:
        # Attempt to load data
        df = await load_data(file_path_activities)
    except DataLoadError as e:
        logging.error(f"Data load error: {e}")
        report = "# Business Activities Analysis Report\n\nSorry, we were unable to analyze the activity data due to an issue with loading the data."
        return report
    else:
        try:
            # Attempt to calculate metrics
            metrics = await calculate_metrics(df)
            report = generate_report(metrics)
            #print(report)
        except Exception as e:
            logging.error(f"Metrics calculation error: {e}")
            report = "# Business Activities Analysis Report\n\nSorry, we were unable to analyze the activity data due to an issue with calculating the metrics."
            return report
         
    # Write the report to file, handling potential write errors
    try:
        async with aiofiles.open('business_activities_report.md', 'w') as f:
            await f.write(report)
        return report
    except Exception as e:
        logging.error(f"Error writing report: {e}")
        print("Failed to write the report to file.")

#if __name__ == "__main__":
#    asyncio.run(main())