import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os
# Load the data


def sleep_model(sleep_cam,mat_csv):
    merged_data = pd.DataFrame()
    for file in os.listdir(mat_csv):
        if file.endswith('.csv'):
            file_path = os.path.join(mat_csv, file)
            df = pd.read_csv(file_path)  # Read each CSV file into a DataFrame
            merged_data = pd.concat([merged_data, df], axis=0)  # Concatenate along rows
    target_file = "sleep_cycle.csv"
    if target_file in os.listdir(sleep_cam):
        file_path = os.path.join(sleep_cam, target_file)
            
    df1 = pd.read_csv(file_path)

    master_data = pd.merge(df1, merged_data, on="Time", how="inner")
    #master_data = pd.read_csv(master_data)

    # Define the feature columns and label column
    X = master_data[['Chn_1', 'Chn_2', 'Chn_3', 'Chn_4']]
    y = master_data['Pose']

    # Encode the labels (assuming 'label' contains categorical values)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets (75% training, 25% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardize the feature data (mean = 0, standard deviation = 1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100
    print(f"Accuracy: {accuracy_percentage:.2f}%")
    return f"{accuracy_percentage:.2f}%"

