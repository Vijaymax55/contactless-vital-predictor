import boto3
import os
import datetime
from conf.config import app_cfg, sid

data = r"C:\Users\Amarjith CK\Documents\Blusim\Git-repos\data"

def download_data(data, date):
    start_time = 21
    end_time = 9
    
    # Set your AWS credentials
    os.environ['AWS_DEFAULT_REGION'] = app_cfg["aws_config_region"]
    os.environ['AWS_ACCESS_KEY_ID'] = app_cfg["aws_config_key"]
    os.environ['AWS_SECRET_ACCESS_KEY'] = app_cfg["aws_config_pass"]

    # Define your S3 bucket
    s3_bucket_name = "data-blusim-care"

    # Convert the date string to a datetime object
    date = datetime.datetime.strptime(date, "%Y%m%d")
    date_p = date.strftime("%Y-%m-%d")
    print(date_p)
    # Calculate the previous day by subtracting one day using a timedelta
    previous_day = date - datetime.timedelta(days=1)
    previous_day_prefix=previous_day.strftime("%Y-%m-%d")
    # Create a Boto3 S3 client
    s3 = boto3.client('s3')

    for date_offset in range((date - previous_day).days + 1):
        target_datetime = previous_day + datetime.timedelta(days=date_offset)
        date_prefix = target_datetime.strftime("%Y-%m-%d")
        print(date_prefix)
        while True:
            # Check if the hour is within the time range (21:00 to 09:00, spanning two days)
            if date_prefix == previous_day_prefix:
                for hour in range(start_time,24):
                    objects = s3.list_objects(Bucket=s3_bucket_name, Prefix=f"{sid}")
                
                    for obj in objects.get("Contents", []):
                        key = obj["Key"]
                        if os.path.basename(key).startswith(f"sleep_data_{date_prefix}_{hour}.csv"):
                            file_name = key.split("/")[-1]
                            local_file_path = os.path.join(data, file_name)
                    
                            # Download the file to the local directory
                            s3.download_file(s3_bucket_name, key, local_file_path)
                            print(f"Downloaded file: {file_name}")
                break
            if date_prefix == date_p:
                for hour in range(00,end_time):
                    hour=str(hour).zfill(2)
                    objects = s3.list_objects(Bucket=s3_bucket_name, Prefix=f"{sid}")
                
                    for obj in objects.get("Contents", []):
                        key = obj["Key"]
                        if os.path.basename(key).startswith(f"sleep_data_{date_prefix}_{hour}.csv"):
                            file_name = key.split("/")[-1]
                            local_file_path = os.path.join(data, file_name)
                    
                            # Download the file to the local directory
                            s3.download_file(s3_bucket_name, key, local_file_path)
                            print(f"Downloaded file: {file_name}")
                break
    print("Download completed.")

if __name__ == '__main__':
    # Specify the date and data folder
    date = "20231028"
    download_data(data, date)
