import csv
import json
import datetime

def add_accuracy(date,data,git_name,ac):
    csv_file = f'{data}/accuracy_prediction.csv'
    with open("conf/variables.json", "r") as j_file:
        loaded_data = json.load(j_file)
    ver = loaded_data.get("v_no")
    def create_csv_file(file_path,ver):
        row_names = ["sleep_pose", "sleep_cycle", "presence"]
        col_names = [f"V{ver}"]
        # Create a CSV file and write the header
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write the header with column names
            writer.writerow([''] + col_names)

            # Write the data rows with row names
            for row_name in row_names:
                writer.writerow([row_name] + [''] * len(col_names))

        print(f"CSV file '{file_path}' has been created with predefined row and column names.")

    create_csv_file(csv_file,ver)
    
    def write_to_csv(file_path, row_name, data):
        col_name = f"V{ver}"
        # Read the existing data from the CSV file
        rows = []
        with open(file_path, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                rows.append(row)

        # Find the row index with the specified row name
        row_index = -1
        for i, row in enumerate(rows):
            if row[0] == row_name:
                row_index = i

        if row_index == -1:
            print(f"Row name '{row_name}' not found in the CSV file.")
            return

        # Find the column index with the specified column name
        col_index = -1
        if col_name in rows[0]:
            col_index = rows[0].index(col_name)
        else:
            print(f"Column name '{col_name}' not found in the CSV file.")
            return

        # Update the data in the specified cell
        rows[row_index][col_index] = str(data)

        # Write the updated data back to the CSV file
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)

        print(f"Data '{data}' has been written to the column '{col_name}' for row '{row_name}'.")
    
    if git_name=="sleep_pose":
        value=f'{ac}'
        write_to_csv(csv_file,git_name,value)
    elif git_name=="sleep_cycle":
        value=f'{ac}'
        write_to_csv(csv_file,git_name,value)
    elif git_name=="presence":
        value=f'{ac}'
        write_to_csv(csv_file,git_name,value)