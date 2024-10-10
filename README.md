
# Entropy Suite

Entropy Suite is a Streamlit-based web application designed to analyze the complexity of time-series data, particularly EEG data, using various entropy measures like Sample Entropy and Fuzzy Entropy. The app provides users with interactive plots, multiple entropy measures, and the ability to download the processed data.

## Features
- Upload preprocessed EEG data in CSV format or use the sample dataset(s) provided.
- Choose from multiple entropy measures
- Perform single-channel and multichannel entropy analysis
- Visualize entropy across multiple scales with interactive plots
- Download processed entropy data in CSV format

## Table of Contents
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Sample Data](#sample-data)
- [Updating the Repository](#updating-the-repository)
- [License](#license)


## Installation

### Step 1: Install Python (if not already installed)
Before starting, make sure you have Python installed. You can download it from [python.org](https://www.python.org/).

### Step 2: Create an empty folder
Create an empty folder where you want to store the Entropy Suite files, then open terminal (Mac/Linux) or command prompt (Windows) and navigate to the folder:
```bash
cd /path/to/your/folder
```

### Step 3: Clone the repository
Clone the repository into your chosen folder:
```bash
git clone https://github.com/Tcgoalie29/Entropy_Suite_App.git .
```

### Step 4: Install the required dependencies
Install the necessary Python packages by running:
```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit app
Start the application with the following command:
```bash
streamlit run entropy_suite_app_12.0.py
```

Note: After installation, the application can simply be run by navigating to the folder:
```bash
cd /path/to/your/folder
```
and starting the application with:
```bash
streamlit run entropy_suite_app_12.0.py
```

## How to Use

1. Upload your preprocessed EEG CSV files via the sidebar or select one of the sample datasets.
2. Customize the entropy analysis by selecting the analysis type, entropy measure, and other parameters like scales and sampling frequency.
3. Click "Run Analysis" to start processing the data.
4. Visualize the results through interactive plots, and download the results as CSV for further analysis.

## Project Structure

```bash
/Entropy-Suite-App
  - entropy_suite_app_12.0.py            # Main Python script for running the Streamlit app
  - requirements.txt                     # List of required Python packages
  - README_Entropy_Suite.md              # Documentation for the project
  /data
    - Rest_EC_00831_01_FilteredData.csv  # Example EEG data for testing
    - Rest_EC_860_01_FilteredData.csv    # Example EEG data for testing
  /assets
    - Entropy_Updated_Background.png     # Image file for the sidebar logo
```

## Sample Data

The `data` directory contains sample EEG files you can use to test the application. These files are in CSV format and represent preprocessed EEG data.


## Updating the Repository

To update your local copy of the repository after changes are made in the GitHub repository, use the following commands:

1. Navigate to the folder containing your local repository:
```bash
cd /path/to/your/folder
```

2. Pull the latest changes from the remote repository:
```bash
git pull origin main
```

## License

This project is licensed under the MIT License.