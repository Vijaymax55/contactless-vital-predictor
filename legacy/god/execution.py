#System imports
import os
import datetime
from dashboard_script import app
import glob
import shutil
import json
#import from files
from conf.paths import data_path,video_path,repo_path

#import for repo codes
import sys
repository_paths = [f"{repo_path}/{path}" for path in ["pr", "sleep_pose", "sleep_cycle"]]
sys.path.extend(repository_paths)

import sleep_pose_cam as spc
import SC_preprocessing as scp
import SC_model as scm
import sleep_poses as sp
import presence_decision_tree as pd
import SC_to_csv as scc

import mat_data_get as mdg
import accuracy_csv as ac
from files_git import git_files

with open("conf/variables.json", "r") as j_file:
    values = json.load(j_file)

def create_date_folder(date):
    folder_path = os.path.join(data_path, date)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        for i in ['mat_csv','sleep_pose','sleep_cycle']:
            data_folder = os.path.join(folder_path,i)
            os.makedirs(data_folder)
        exe(date)

def exe(date):
    if "-" in date:
        date = date.replace("-", "")
    if os.path.exists(data_path) and os.path.isdir(data_path):
        date_folder_path = os.path.join(data_path, date)
        if os.path.exists(date_folder_path) and os.path.isdir(date_folder_path):
            print(f"Folder '{date}' exists in '{data_path}'.")
            git_name_fil(repository_paths)
            for i in ['mat_csv','sleep_pose','sleep_cycle']:
                data_folder_path = os.path.join(date_folder_path,i)
                if not os.path.exists(data_folder_path) or not any(f.endswith('.csv') for f in os.listdir(data_folder_path)):
                    if not os.path.exists(data_folder_path):    
                        os.makedirs(data_folder_path)
                    if not any(f.endswith('.csv') for f in os.listdir(data_folder_path)):
                        if i == "mat_csv" :
                            mat_csv(date)
                        if i == "sleep_pose":
                            cam(date)
                        if i =="accuracy_prediction":
                            accuracy(date)
                        if i == "sleep_cycle":
                            s_cycle(date)
                    else:
                        print(f"Folder '{date_folder_path}/{i}' does have a CSV file.")
                else:
                    print("folder and file not present")
            return True
        else:
            create_date_folder(date)
    else:
        print("Data Folder not available")
        return False
        

def git_name_fil(repository_paths):
    changes = git_files(repository_paths)
    for change in changes:
        if change=="sleep_pose" or change=="sleep_cycle" or change=="pr":
            with open("conf/variables.json", "r") as j_file:
                loaded_data = json.load(j_file)
            ver = loaded_data.get("v_no")
            ver = ver+1
            j_file.close()
            loaded_data["v_no"]=ver
            with open("conf/variables.json", "w") as file:
                json.dump(loaded_data, file, indent=4)
    print(changes)

def custom_ignore(src, names):
    existing_files = set(os.listdir(src))
    return set(names) & existing_files

def is_sleep_pose_large_enough(folder_path):
    file_path = os.path.join(folder_path, "sleep_pose.csv")
    if os.path.exists(file_path) and os.path.getsize(file_path) > 5120:  # 5KB
        return True
    return False

def cam(date):
    matching_folders = []
    process = []
    date_obj = datetime.datetime.strptime(date, "%Y%m%d")
    previous_day = date_obj - datetime.timedelta(days=1)
    
    # Generate date strings for the previous day and current date
    previous_day_str = previous_day.strftime("%Y%m%d")
    current_date_str = date
    
    # Convert 21:00 to an integer
    start_time = int(21)
    end_time = int(9)
    
    # Iterate through subfolders in video_paths
    for folder_name in os.listdir(video_path):
        folder_path = os.path.join(video_path, folder_name)

        # Check if the subfolder name starts with the specified date
        if folder_name.startswith(previous_day_str) and os.path.isdir(folder_path):
            timestamp_str = folder_name.split('_')[-1][-2:]
            # Convert the extracted timestamp to an integer
            folder_hour = int(timestamp_str)
            if folder_hour >= end_time:
                matching_folders.append(folder_path)
        elif folder_name.startswith(current_date_str) and os.path.isdir(folder_path):
            timestamp_str = folder_name.split('_')[-1][-2:]
            # Convert the extracted timestamp to an integer
            folder_hour = int(timestamp_str)
            if folder_hour <= start_time:
                matching_folders.append(folder_path)

    # Copy matching folders to the destination path
    for folder in matching_folders:
        destination_folder = os.path.join(data_path, date, "sleep_pose", os.path.basename(folder))
        
        # Check if the destination folder already exists, and if it does, skip the copying
        if not os.path.exists(destination_folder):
            shutil.copytree(folder, destination_folder, ignore=custom_ignore)
        else:
            print(f"Folder '{destination_folder}' already exists. Skipping copy.")
    
    for folder in matching_folders:
        if is_sleep_pose_large_enough(folder):
            process.append(folder)
        else:
            cam_result = spc.process_videos_in_directory(process, os.path.join(data_path, date), date)
            print(cam_result)

def accuracy(date):
    for git_name in ["sleep_pose","sleep_cycle","presence"]:
        if git_name == "sleep_cycle":
            sc_mat = os.path.join(data_path,date, "sc_mat")
            scp.pre_process_sleep_data(f'{data_path}\{date}',f'{data_path}\{date}\mat_csv',sc_mat)
            csv_files = glob.glob(os.path.join(sc_mat, '*.csv'))            
            if not csv_files:
                print("No CSV files found in the directory.")
            else:
                # Sort the CSV files by modification time (most recent first)
                csv_files.sort(key=os.path.getmtime, reverse=True)
                # Get the path of the latest CSV file
                latest_csv_file = csv_files[0]
                ac_sc = scm.sleep_model(latest_csv_file)
                acc_csv=ac.add_accuracy(date,f"{data_path}/{date}",git_name,ac_sc)
                print(acc_csv)
        if git_name == "sleep_pose":
            ac_sp=sp.sleep_model(f'{data_path}\{date}',f'{data_path}\{date}\mat_csv')
            
            ac.add_accuracy(date,f"{data_path}/{date}",git_name,ac_sp)
        if git_name == "presence":
            ac_pr=pd.train(f'{data_path}\{date}',f'{data_path}\{date}\mat_csv')
            ac.add_accuracy(date,f"{data_path}/{date}",git_name,ac_pr)
            
def s_cycle(date):
    data = data_path
    def get_latest_file_in_folder(path,date):
        sc_path = f"{path}/{date}/sleep_cycle"
        # The 'sc' subfolder does not exist or is not a directory

        current_date = date

        # Filter files to find the one with a matching date in the filename
        matching_files = [
            os.path.join(sc_path, file) for file in os.listdir(sc_path)
            if file.endswith('.png') and current_date in file
        
        ]
        if not matching_files:
            return None  # No PNG files with a matching date in the 'sc' subfolder

        # Sort the matching files by modification time (most recent first)
        matching_files.sort(key=os.path.getmtime, reverse=True)

        # Get the path of the latest matching file
        latest_file = matching_files[0]
        return latest_file

    latest_file = get_latest_file_in_folder(data,date)

    if latest_file:
        print("Latest file:", latest_file)
        scc.process_sleep_data(latest_file, f"{data_path}/{date}",date)
    else:
        print("No PNG files found in the 'sc' subfolder.")

def mat_csv(date):
    mat=mdg.download_data(f"{data_path}/{date}/mat_csv",date)
    print(mat)

if __name__ == '__main__':
    #create_date_folder("20231102")
    cam("20231028")
