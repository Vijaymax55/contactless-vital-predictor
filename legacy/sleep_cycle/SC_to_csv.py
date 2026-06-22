import pandas as pd
from PIL import Image
import numpy as np
import math
import pytesseract
from datetime import datetime, timedelta
import matplotlib.image as mpimg
import tempfile
import datetime
import matplotlib.pyplot as plt

# Define pixel colors for sleep stages
NotInBed = [255, 255, 255]
Awake = [250,188,5]
REM = [142,194,251]
LightSleep1 = [85,139,247]
LightSleep2 = [85, 139, 247]
DeepSleep = [38,74,160]

file = rf"C:\Users\Amarjith CK\Documents\Blusim\Git-repos\data\sc\Screenshot_20231020-115704.png"
csv_path = rf"C:\Users\Amarjith CK\Documents\Blusim\Git-repos\data\20231020\sleep_ring"

# Function to determine sleep stage based on pixel color
def isAwake(pixel):
    pixel = pixel[:3]  # Keep the first three elements to match the shape of Awake
    return math.sqrt(np.square(pixel - np.array(Awake)).mean()) < 15

def isREM(pixel):
    pixel = pixel[:3]  # Keep the first three elements to match the shape of Awake
    return math.sqrt(np.square(pixel - np.array(REM)).mean()) < 15

def isLightSleep1(pixel, x, y):
    pixel = pixel[:3]  # Keep the first three elements to match the shape of LightSleep1
    return (
        30 <= x < 568 and 97 <= y < 143 and  # Adjusted coordinates for Light Sleep 1
        math.sqrt(np.square(pixel - np.array(LightSleep1)).mean()) < 15
    )

def isLightSleep2(pixel, x, y):
    pixel = pixel[:3]  # Keep the first three elements to match the shape of LightSleep2
    return (
        30 <= x < 568 and 148 <= y < 192 and  # Adjusted coordinates for Light Sleep 2
        math.sqrt(np.square(pixel - np.array(LightSleep2)).mean()) < 15
    )

def isDeepSleep(pixel):
    pixel = pixel[:3]  # Keep the first three elements to match the shape of Awake
    return math.sqrt(np.square(pixel - np.array(DeepSleep)).mean()) < 15

from typing import Any, Union, Literal

def assign_sleep_mode(pixel: Any, current_mode, x, y):
    if current_mode == 0:
        if isAwake(pixel):
            return 1
        if isREM(pixel):
            return 2
        if isLightSleep1(pixel,x,y):
            return 3  # Light Sleep 1
        if isLightSleep2(pixel,x,y):
            return 4  # Light Sleep 2
        if isDeepSleep(pixel):
            return 5
    elif current_mode in (1, 2, 3, 4, 5):
        if isAwake(pixel) or isREM(pixel) or isLightSleep1(pixel,x,y) or isLightSleep2(pixel,x,y) or isDeepSleep(pixel):
            return 0
    return current_mode

# Load the screenshot
#screenshot = mpimg.imread('Screenshot_20231017-143742.png')

# Create a temporary image file from the screenshot
def process_sleep_data(image_path,csv_path,date):
    file = mpimg.imread(image_path)

    temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_image_name = temp_image.name
    mpimg.imsave(temp_image_name, file)
    temp_image.close()


    # Open the image from the temporary file using PIL
    im = Image.open(temp_image_name)
    im = im.resize((700, 1600))

    # Values to specify the cropping area
    left_data = 50
    top_data = 604
    right_data = 650
    bottom_data = im.height // 2  +46# Use integer division

    # Crop the image based on the specified coordinates
    cropped_im = im.crop((left_data, top_data, right_data, bottom_data))

    # Values to get time from image (region of interest)
    left = int(im.width / 14)
    top = im.height / 2 + 60
    right = im.width - 50
    bottom = im.height / 2 + 95

    # Crop the image to extract time
    im2 = im.crop((left, top, right, bottom))

    # Initialize data storage
    timing_data = [[0, 0]]  # Time and state
    cm, pm = 0, 0

    # Main function: Loops through breadth and width of the data to get the starting point of stages
    for x in range(cropped_im.width):
        for y in range(cropped_im.height):
            cm = assign_sleep_mode(cropped_im.getpixel((x, y)), cm, x, y)
            if cm != pm:
                if cm != 0:
                    # Appending the data to the list
                    timing_data.append([x, cm])
                pm = cm

    # Gets the end point of the last stage
    flag = 0
    for x in reversed(range(cropped_im.width)): # Rightmost edge of the image, moving towards the left
        for y in range(cropped_im.height):
            cm = assign_sleep_mode(cropped_im.getpixel((x, y)), cm, x, y)
            if cm != pm:
                if cm != 0:
                    last_pval = x
                    flag = 1
                    break
                pm = cm
        if flag == 1:
            break

    # Create a DataFrame and save the data
    df = pd.DataFrame(timing_data, columns=['pixel_numer', 'label'])
    df = df.iloc[1:]

    # Config to get time
    custom_oem = r'digits --oem 1 --psm 6 -c tessedit_char_whitelist=0123456789:'

    # Extract time from image
    time_str = pytesseract.image_to_string(im2, config=custom_oem)

    # Split the extracted text into lines
    lines = time_str.split('\n')

    # Split start and end time pixel to hour and minute
    ST, ET = time_str.split(" ")
    print(ET)
    ET, _ = ET.split("\n")

    from datetime import datetime
    # Getting start and end time
    st_time = datetime.strptime(ST, '%H:%M')
    et_time = datetime.strptime(ET, '%H:%M')

    # Calculate the total time interval in minutes
    if et_time < st_time:
        # If end time is earlier than start time (crossing midnight), add 24 hours to end time
        total_time = ((et_time.hour + 24) * 60 + et_time.minute) - ((st_time.hour * 60) + st_time.minute)
    else:
        total_time = ((et_time.hour * 60) + et_time.minute) - ((st_time.hour * 60) + st_time.minute)

    # Finding the starting point of the image
    min_pval = df['pixel_numer'].min()
    p_time = total_time / (last_pval - min_pval)

    # Function to get time for each row in the DataFrame
    def get_time(row):
        time_minut = (row['pixel_numer'] - min_pval) * p_time
        time_hr = int(time_minut / 60)
        time_minut = time_minut % 60
        time_stamp = st_time + timedelta(hours=time_hr, minutes=time_minut)
        # Format time_stamp as HH:MM string
        time_str = time_stamp.strftime('%H:%M')
        return time_str

    # Append start time and get end time for each stage
    df['start_time'] = df.apply(get_time, axis=1)
    end_time = df['start_time'].iloc[1:].tolist()
    end_time.append(et_time.strftime('%H:%M'))
    df["end_time"] = end_time
    df=df.drop_duplicates()


    rearranged_data = []

    for _, row in df.iterrows():
        current_time = datetime.strptime(row['start_time'], '%H:%M')
        end_time = datetime.strptime(row['end_time'], '%H:%M')
        label = row['label']

        while current_time <= end_time:
            #unix_time = int(current_time.timestamp())
            rearranged_data.append({ 'Time': current_time.strftime('%H:%M:%S'), 'label': label})
            current_time += timedelta(seconds=1)

    # Create a new DataFrame from the rearranged data
    new_df = pd.DataFrame(rearranged_data)

    # Replace label with actual state
    # Define a mapping from numerical values to actual values
    value_mapping = {
        0: 'NotInBed',
        1: 'Awake',
        2: 'REM',
        3: 'LightSleep1',
        4: 'LightSleep2',
        5: 'DeepSleep'
    }

    # Replace numerical values in the 'label' column with actual values
    new_df['label'] = new_df['label'].replace(value_mapping)

    date = datetime.datetime.strptime(date, "%Y%m%d")
    date = date.strftime("%Y-%m-%d")
    file_name = f"{csv_path}/sleep_cycle_{date}.csv"
    # Save the DataFrame to the file
    new_df.to_csv(file_name, index=False)

if __name__ == "__main__":
    process_sleep_data(file,csv_path)