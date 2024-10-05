
# Entropy Suite

Entropy Suite is a Streamlit-based web application designed to analyze the complexity of time-series data, particularly EEG data, using various entropy measures like Sample Entropy and Fuzzy Entropy. The app provides users with interactive plots, multiple entropy measures, and the ability to download the processed data.

## Features
- Upload preprocessed EEG data in CSV format
- Choose from multiple entropy measures
- Perform single-channel and multichannel entropy analysis
- Visualize entropy across multiple scales with interactive plots
- Download processed entropy data in CSV format

## Table of Contents
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Sample Data](#sample-data)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Tcgoalie29/Entropy-Suite-App.git
   cd entropy-suite
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How to Use

1. Upload your preprocessed EEG CSV files via the sidebar or select one of the sample datasets.
2. Customize the entropy analysis by selecting the analysis type, entropy measure, and other parameters like scales and sampling frequency.
3. Click "Run Analysis" to start processing the data.
4. Visualize the results through interactive plots, and download the results as CSV for further analysis.

## Project Structure

```
/entropy-suite
  - app.py                    # Main Python script for running the Streamlit app
  - requirements.txt           # List of required Python packages
  - README.md                  # Documentation for the project
  /data
    - Rest_EC_00831_01_FilteredData.csv  # Example EEG data for testing
    - Rest_EC_860_01_FilteredData.csv    # Example EEG data for testing
  /assets
    - Entropy_Updated_Background.png     # Image file for the sidebar logo
```

## Sample Data

The `data` directory contains sample EEG files you can use to test the application. These files are in CSV format and represent preprocessed EEG data.

## License

This project is licensed under the MIT License.
