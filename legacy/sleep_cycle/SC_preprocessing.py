import pandas as pd
import os

# Define the path to the folder containing the CSV files
mat_path = "/Users/vijayprakash/Downloads/Codes/Sleep_cycle/sleep_mat_data"
ring_csv = "SleepCycle_18-10-2023_16-02_1697625170.csv"
data_path = "/Users/vijayprakash/Documents/sleep_cycle"
def pre_process_sleep_data(ring_csv,mat_path,data_path):
# Initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Iterate through the CSV files in the folder and concatenate them horizontally
    for file in os.listdir(mat_path):
        if file.endswith('.csv'):
            file_path = os.path.join(mat_path, file)
            df = pd.read_csv(file_path)  # Read each CSV file into a DataFrame
            merged_data = pd.concat([merged_data, df], axis=0)  # Concatenate along rows

    # Convert the "Timestamp" column to datetime format
    merged_data['Time'] = pd.to_datetime(merged_data['Time'], format='%H:%M:%S')

    # Add 12 hours to each value in the "Timestamp" column
    #merged_data['Time'] = merged_data['Time'] + pd.to_timedelta('12 hours')

    # Format the "Timestamp" column as HH:mm:ss
    merged_data['Time'] = merged_data['Time'].dt.strftime('%H:%M:%S')
    for file in os.listdir(ring_csv):
        if file.endswith('.csv'):
            file_path = os.path.join(ring_csv, file)
            #df = pd.read_csv(file_path)
    # Merge df1 and merged_data on "Time" column
    #df1_path = ring_csv
    df1 = pd.read_csv(file_path)

    master_data = pd.merge(df1, merged_data, on="Time", how="inner")

    # Save the merged data to a CSV file
    master_data.to_csv(f"{data_path}/SC_master_data.csv", index=False)


if __name__ == "__main__":
    pre_process_sleep_data(ring_csv,mat_path,data_path)