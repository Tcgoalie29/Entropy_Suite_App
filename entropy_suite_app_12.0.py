# entropy_suite_app.py

import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import EntropyHub as eh
import time

# Initialize st.session_state variables if not already set
if 'entropy_df' not in st.session_state:
    st.session_state['entropy_df'] = None
if 'ci_df' not in st.session_state:
    st.session_state['ci_df'] = None
if 'fig' not in st.session_state:
    st.session_state['fig'] = None
if 'formatted_time' not in st.session_state:
    st.session_state['formatted_time'] = None

# Set up the Streamlit page configuration
st.set_page_config(page_title="Entropy Suite", layout="wide")

# Function to format time
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} minutes {seconds:.2f} seconds"

# Inject custom CSS styles
st.markdown(
    f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: #0E1117;
        color: #4BDCFF;
        font-family: 'Roboto', sans-serif;
    }}
    /* Sidebar background */
    .css-1d391kg {{
        background-color: #263026;
    }}
    /* Text color for various elements */
    body, .css-17eq0hr {{
        color: #4BDCFF;
    }}
    /* Input fields */
    .css-1n543e5, .css-1fv8s86, .css-1d1c3wd, .css-1cpxqw2, .st-bs, .st-bu, .st-bx {{
        background-color: #0E1117;
        color: #4BDCFF !important;
        border: 1px solid #004A9C;
    }}
    /* Checkbox */
    .stCheckbox .st-bq {{
        color: #4BDCFF !important;
    }}
    .stCheckbox .st-bd {{
        color: #4BDCFF !important;
    }}
    /* Buttons */
    .stButton>button {{
        background-color: #004A9C;
        color: #4BDCFF;
        border: None;
    }}
    /* Progress bar */
    .stProgress > div > div > div {{
        background-color: #4BDCFF;
    }}
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: #4BDCFF;
    }}
    /* Axes labels and tick labels */
    .plotly .xtick text, .plotly .ytick text {{
        fill: #FFFFFF !important;
    }}
    /* Adjust x-axis labels */
    .plotly .xaxis .tick text {{
        fill: #FFFFFF !important;
    }}
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: #263026;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #004A9C;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("Entropy Suite")

# Sidebar logo (optional)
st.sidebar.image("Entropy_Updated_Background.png", use_column_width=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Documentation", "Analysis", "Entropy Education"])

# ---------------------- Tab 1: Documentation ----------------------
with tab1:
    st.header("How to Use the Entropy Suite")

    st.markdown("""
    ## Introduction
    Welcome to the **Entropy Suite**, a comprehensive tool designed for analyzing the complexity of time-series data using various entropy measures. This application is particularly suited for EEG data but can be applied to other types of time-series data, such as ECG or heart rate data.                
    
    **Features:**
    - Upload your preprocessed EEG, ECG, or heart rate data in CSV format.
    - Choose the type of analysis and entropy measure.
    - Customize parameters such as the number of scales, data trim length, and sampling frequency.
    - Visualize the results with interactive plots.
    - Download the entropy data for further analysis.

    **Getting Started:**
    1. **Upload Data or Select Sample Data:** Use the sidebar to upload your EEG, ECG, or heart rate data CSV files or select sample datasets. Ensure that your data is preprocessed and formatted correctly.
    2. **Set Parameters:** Select the analysis type, entropy measure, and adjust the parameters as needed.
    3. **Run Analysis:** Click the 'Run Analysis' button to start processing.
    4. **View Results:** The results will be displayed under the 'Analysis' tab with interactive plots.
    5. **Download Data:** You can download the entropy data as a CSV file for further analysis.

    **Data Requirements:**
    - EEG, ECG, or heart rate data should be in CSV format with channels or derived metrics as columns.
    - Ensure that the sampling frequency and data trim length are set correctly to match your data.

    **Support:**
    - For questions, issues, or feedback please contact the developer at wcreel@students.llu.edu

    **Disclaimer:**
    - This tool is intended for research purposes. Please ensure that you understand the methods and interpretations of the entropy measures used.
    """)

# ---------------------- Tab 2: Analysis ----------------------
with tab2:

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload your preprocessed EEG, ECG, or heart rate .CSV files",
        accept_multiple_files=True,
        type=["csv"],
        help="Upload one or more preprocessed EEG, ECG, or heart rate data files in CSV format. Each file should contain time-series data from the corresponding channels or derived heart rate measurements."
    )

    # Preload sample datasets
    sample_files = {
        "Sample EEG 1": "Rest_EC_00831_01_FilteredData.csv",
        "Sample EEG 2": "Rest_EC_860_01_FilteredData.csv"
    }
    selected_samples = st.sidebar.multiselect(
        "Or select sample EEG dataset(s)",
        list(sample_files.keys()),
        help="Select one or more sample datasets to load and preview."
    )

    # Load and preview selected sample datasets
    sample_data_list = []
    if selected_samples:
        for sample_name in selected_samples:
            try:
                sample_data = pd.read_csv(sample_files[sample_name])
                sample_data_list.append((sample_name, sample_data))
            except FileNotFoundError:
                st.sidebar.error(f"Sample file {sample_files[sample_name]} not found in the app directory.")

    # Combine uploaded files and sample files
    all_files = []
    if uploaded_files:
        all_files.extend(uploaded_files)
    if sample_data_list:
        # For each sample_data, create a file-like object
        from io import StringIO
        for sample_name, sample_data in sample_data_list:
            sample_buffer = StringIO()
            sample_data.to_csv(sample_buffer, index=False)
            sample_buffer.seek(0)
            sample_file = sample_buffer
            sample_file.name = sample_name  # Set the name attribute
            all_files.append(sample_file)

    # Parameter selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        (
            "Single-Subject Single-Channel",
            "Single-Subject Multichannel",
            "Group-Level Multichannel",
            "Group-Level Single-Channel",
        ),
        help="""Choose the type of analysis:
- **Single-Subject Single-Channel**: Analyze a single channel from one subject.
- **Single-Subject Multichannel**: Analyze multiple channels from one subject.
- **Group-Level Multichannel**: Analyze multiple subjects across multiple channels.
- **Group-Level Single-Channel**: Analyze a single channel across multiple subjects.
"""
    )

    entropy_type_selection = st.sidebar.selectbox(
        "Select Entropy Measure", ("Sample Entropy (SampEn)", "Fuzzy Entropy (FuzzEn)"),
        help="""Select the entropy measure to use:
- **Sample Entropy (SampEn)**: A measure of complexity that quantifies the regularity and unpredictability of fluctuations over time-series data.
- **Fuzzy Entropy (FuzzEn)**: An entropy measure that incorporates fuzzy logic to improve robustness against noise and data length."""
    )

    # Map entropy_type_selection to actual entropy function names
    entropy_type_mapping = {
        "Sample Entropy (SampEn)": "SampEn",
        "Fuzzy Entropy (FuzzEn)": "FuzzEn"
    }
    entropy_type = entropy_type_mapping[entropy_type_selection]

    # Change min_value to 2
    num_scales = st.sidebar.slider(
        "Number of Scales", min_value=2, max_value=50, value=20,
        help="Determine the number of scales for multiscale entropy analysis. Higher scales analyze coarser-grained time-series."
    )
    trim_length = st.sidebar.number_input(
        "Data Trim Length (seconds)", min_value=1, value=30,
        help="Specify the length of data (in seconds) to be used from the start of the recording."
    )
    sfreq = st.sidebar.number_input(
        "Sampling Frequency (Hz)", min_value=1, value=1000,
        help="Enter the sampling frequency (in Hertz) of your EEG data."
    )

    # Channel selection
    target_channel = None
    exclude_channels = []

    if analysis_type in [
        "Single-Subject Single-Channel",
        "Group-Level Single-Channel",
    ]:
        target_channel = st.sidebar.text_input(
            "Enter Target Channel", value="FP1",
            help="Specify the EEG channel to analyze (e.g., 'FP1')."
        )

    if analysis_type in [
        "Single-Subject Multichannel",
        "Group-Level Multichannel",
    ]:
        exclude_channels = st.sidebar.text_input(
            "Channels to Exclude (comma-separated)", value="",
            help="List any channels to exclude from analysis, separated by commas (e.g., 'EOG,EMG')."
        )
        exclude_channels = [ch.strip() for ch in exclude_channels.split(",") if ch.strip()]

    # Option to toggle legends
    show_legends = st.sidebar.checkbox(
        "Show Legends", value=True,
        help="Toggle to show or hide legends in the plots."
    )

    # Add a Run Analysis button
    run_analysis = st.sidebar.button(
        "Run Analysis",
        help="Click to start the entropy analysis with the selected parameters."
    )

    # Process data when files are uploaded or sample data is selected and run_analysis is clicked
    if all_files and run_analysis:
        # Error handling for number of uploaded files
        if analysis_type.startswith("Single-Subject") and len(all_files) > 1:
            st.error("Please upload only one file or select one sample dataset for Single-Subject analysis.")
        elif analysis_type.startswith("Group-Level") and len(all_files) == 1:
            st.error("Please upload multiple files or select multiple sample datasets for Group-Level analysis.")
        else:
            st.header("Processing Data...")
            scales_list = np.arange(1, num_scales + 1)

            # Determine entropy function
            Mobj = eh.MSobject(entropy_type)

            # Initialize DataFrames to store results
            entropy_df = pd.DataFrame()
            ci_df = pd.DataFrame()

            # Main processing based on analysis type
            if analysis_type == "Single-Subject Single-Channel":
                file = all_files[0]
                try:
                    data = pd.read_csv(file)
                    if data.empty:
                        st.error("Uploaded CSV is empty.")
                    else:
                        data = data.iloc[: int(trim_length * sfreq), 1:]  # Trim data
                        ch_names = data.columns.tolist()

                        if target_channel not in ch_names:
                            st.error(f"Channel {target_channel} not found in data.")
                        else:
                            ch_data = data[target_channel].values

                            # Start timing
                            start_time = time.time()

                            # Show spinner while processing
                            with st.spinner('Calculating entropy...'):
                                Msx, CI = eh.MSEn(ch_data, Mobj, Scales=num_scales)

                            end_time = time.time()
                            total_time = end_time - start_time
                            formatted_time = format_time(total_time)

                            entropy_df = pd.DataFrame(
                                {"Scale": scales_list, "Channel": target_channel, entropy_type: Msx}
                            )
                            ci_df = pd.DataFrame({"Channel": [target_channel], "CI": [CI]})

                            st.header("Analysis Complete")
                            st.write(f"Total processing time: {formatted_time}")

                            # Plotting
                            fig = make_subplots(rows=2, cols=1, subplot_titles=(f"MSE ({entropy_type})", "Complexity Index"), vertical_spacing=0.2)
                            fig.add_trace(
                                go.Scatter(
                                    x=scales_list,
                                    y=Msx,
                                    mode="lines+markers",
                                    line=dict(color="#4BDCFF"),
                                    marker=dict(color="#4BDCFF", size=8),
                                    name="MSE",
                                ),
                                row=1,
                                col=1,
                            )
                            fig.add_trace(
                                go.Bar(
                                    x=[target_channel],
                                    y=[CI],
                                    marker=dict(color="#004A9C"),
                                    name="Complexity Index",
                                ),
                                row=2,
                                col=1,
                            )
                            fig.update_layout(
                                template="none",
                                width=1200,
                                height=800,
                                font=dict(size=20, color="#4BDCFF"),
                                xaxis=dict(
                                    title="Scales",
                                    tickfont=dict(color="#FFFFFF"),
                                    titlefont=dict(color="#FFFFFF")
                                ),
                                yaxis=dict(
                                    tickfont=dict(color="#FFFFFF"),
                                    titlefont=dict(color="#FFFFFF")
                                ),
                                xaxis2=dict(
                                    tickfont=dict(color="#FFFFFF"),
                                    titlefont=dict(color="#FFFFFF")
                                ),
                                yaxis2=dict(
                                    tickfont=dict(color="#FFFFFF"),
                                    titlefont=dict(color="#FFFFFF")
                                ),
                                title_text=f"MSE Analysis for {target_channel}",
                                paper_bgcolor="#0E1117",
                                plot_bgcolor="#0E1117",
                                margin=dict(t=100, b=100),
                            )
                            fig.update_yaxes(title_text=f"{entropy_type}", row=1, col=1)
                            fig.update_yaxes(title_text="Total Entropy", row=2, col=1)
                            # Store results in st.session_state
                            st.session_state['entropy_df'] = entropy_df
                            st.session_state['ci_df'] = ci_df
                            st.session_state['fig'] = fig
                            st.session_state['formatted_time'] = formatted_time
                            st.session_state['csv_file_name'] = "entropy_data.csv"

                except Exception as e:
                    st.error(f"An error occurred while processing the file: {e}")

            elif analysis_type == "Single-Subject Multichannel":
                file = all_files[0]
                try:
                    data = pd.read_csv(file)
                    if data.empty:
                        st.error("Uploaded CSV is empty.")
                    else:
                        data = data.iloc[: int(trim_length * sfreq), 1:]  # Trim data
                        data = data.drop(columns=exclude_channels, errors="ignore")
                        ch_names = data.columns.tolist()

                        entropy_list = []
                        ci_df = pd.DataFrame()

                        # Initialize progress bar and status text
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        total_channels = len(ch_names)

                        # Start timing
                        start_time = time.time()

                        for idx, ch in enumerate(ch_names):
                            ch_data = data[ch].values

                            # Compute entropy
                            Msx, CI = eh.MSEn(ch_data, Mobj, Scales=num_scales)
                            ci_row = pd.DataFrame({"Channel": [ch], "CI": [CI]})
                            ci_df = pd.concat([ci_df, ci_row], ignore_index=True)

                            # Create DataFrame for this channel
                            df_ch = pd.DataFrame({"Scale": scales_list, "Channel": ch, entropy_type: Msx})
                            entropy_list.append(df_ch)

                            # Update progress bar and status text
                            progress = (idx + 1) / total_channels
                            percent_complete = int(progress * 100)
                            status_text.text(f"Processing channel {ch} ({idx + 1}/{total_channels}) {percent_complete}%")
                            progress_bar.progress(progress)

                        # Concatenate all channel DataFrames
                        entropy_df = pd.concat(entropy_list, ignore_index=True)

                        # End timing
                        end_time = time.time()
                        total_time = end_time - start_time
                        formatted_time = format_time(total_time)

                        st.header("Analysis Complete")
                        st.write(f"Total processing time: {formatted_time}")

                        # Visualization
                        fig = make_subplots(rows=2, cols=1, subplot_titles=(f"MSE Across Channels ({entropy_type})", "Complexity Index"), vertical_spacing=0.2)
                        # Use shades of blue for consistency
                        base_colors = ['#4BDCFF', '#00A6FF', '#005DFF', '#1E90FF', '#6495ED', '#7B68EE', '#483D8B', '#191970']
                        colors = base_colors * 10  # Extend color palette if needed

                        for i, ch in enumerate(ch_names):
                            ch_data = entropy_df[entropy_df['Channel'] == ch]
                            fig.add_trace(
                                go.Scatter(
                                    x=ch_data["Scale"],
                                    y=ch_data[entropy_type],
                                    mode="lines+markers",
                                    line=dict(color=colors[i % len(colors)]),
                                    marker=dict(color=colors[i % len(colors)], size=6),
                                    name=ch,
                                    showlegend=show_legends,
                                ),
                                row=1,
                                col=1,
                            )

                        # Compute average entropy across channels
                        mean_mse = entropy_df.groupby('Scale')[entropy_type].mean().reset_index()
                        sem_mse = entropy_df.groupby('Scale')[entropy_type].sem().reset_index()
                        fig.add_trace(
                            go.Scatter(
                                x=mean_mse["Scale"],
                                y=mean_mse[entropy_type],
                                error_y=dict(type="data", array=sem_mse[entropy_type], visible=True),
                                mode="lines+markers",
                                line=dict(color="#FFFFFF", width=4),
                                marker=dict(color="#FFFFFF", size=8),
                                name="Average",
                            ),
                            row=1,
                            col=1,
                        )

                        fig.add_trace(
                            go.Bar(
                                x=ci_df["Channel"],
                                y=ci_df["CI"],
                                marker=dict(color="#004A9C", line=dict(color="black", width=1)),
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )

                        fig.update_layout(
                            template="none",
                            width=1200,
                            height=800,
                            font=dict(size=20, color="#4BDCFF"),
                            xaxis=dict(
                                title="Scales",
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            yaxis=dict(
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            xaxis2=dict(
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            yaxis2=dict(
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            title_text="Multichannel Analysis",
                            paper_bgcolor="#0E1117",
                            plot_bgcolor="#0E1117",
                            margin=dict(t=100, b=100),
                        )
                        fig.update_yaxes(title_text=f"{entropy_type}", row=1, col=1)
                        fig.update_yaxes(title_text="Total Entropy", row=2, col=1)
                        # Store results in st.session_state
                        st.session_state['entropy_df'] = entropy_df
                        st.session_state['ci_df'] = ci_df
                        st.session_state['fig'] = fig
                        st.session_state['formatted_time'] = formatted_time
                        st.session_state['csv_file_name'] = "entropy_data.csv"

                except Exception as e:
                    st.error(f"An error occurred while processing the file: {e}")

            elif analysis_type == "Group-Level Multichannel":
                try:
                    # Processing multiple files
                    all_data_dict = {}
                    for file in all_files:
                        data = pd.read_csv(file)
                        if data.empty:
                            st.error(f"Uploaded CSV {file.name} is empty.")
                            continue
                        data = data.iloc[: int(trim_length * sfreq), 1:]
                        data = data.drop(columns=exclude_channels, errors="ignore")
                        subject_name = os.path.splitext(file.name)[0]
                        all_data_dict[subject_name] = data

                    entropy_list = []
                    ci_across_subjects_df = pd.DataFrame()

                    # Initialize progress bar and status text
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_subjects = len(all_data_dict)

                    # Start timing
                    start_time = time.time()

                    for idx, (subject_name, data) in enumerate(all_data_dict.items()):
                        ch_names = data.columns.tolist()
                        # Initialize arrays to accumulate entropy per scale
                        entropy_accumulator = np.zeros(num_scales)
                        ci_list = []

                        for ch in ch_names:
                            ch_data = data[ch].values

                            # Compute entropy
                            Msx, CI = eh.MSEn(ch_data, Mobj, Scales=num_scales)

                            # Accumulate entropy values
                            entropy_accumulator += Msx
                            ci_list.append(CI)

                        # Compute average entropy across channels for this subject
                        avg_entropy_values = entropy_accumulator / len(ch_names)
                        avg_entropy = pd.DataFrame({'Subject': subject_name, 'Scale': scales_list, entropy_type: avg_entropy_values, 'Channel': 'Average'})

                        entropy_list.append(avg_entropy)

                        # Compute average CI across channels for this subject
                        avg_ci = np.mean(ci_list)
                        avg_ci_df = pd.DataFrame({'Subject': [subject_name], 'Channel': ['Average'], 'CI': [avg_ci]})
                        ci_across_subjects_df = pd.concat([ci_across_subjects_df, avg_ci_df], ignore_index=True)

                        # Update progress bar and status text
                        progress = (idx + 1) / total_subjects
                        percent_complete = int(progress * 100)
                        status_text.text(f"Processing subject {subject_name} ({idx + 1}/{total_subjects}) {percent_complete}%")
                        progress_bar.progress(progress)

                    # Concatenate all entropy data
                    entropy_df = pd.concat(entropy_list, ignore_index=True)

                    # End timing
                    end_time = time.time()
                    total_time = end_time - start_time
                    formatted_time = format_time(total_time)

                    st.header("Analysis Complete")
                    st.write(f"Total processing time: {formatted_time}")

                    # Visualization
                    fig = make_subplots(rows=2, cols=1, subplot_titles=(f"MSE Across Subjects ({entropy_type})", "Complexity Index"), vertical_spacing=0.2)
                    # Use shades of blue
                    base_colors = ['#4BDCFF', '#00A6FF', '#005DFF', '#1E90FF', '#6495ED', '#7B68EE', '#483D8B', '#191970']
                    colors = base_colors * 10  # Extend color palette if needed

                    # Use the average entropy data for plotting
                    avg_data = entropy_df[entropy_df['Channel'] == 'Average']

                    group_avg_mse = avg_data.groupby("Scale")[entropy_type].mean().reset_index()
                    group_sem_mse = avg_data.groupby("Scale")[entropy_type].sem().reset_index()

                    for i, subject_name in enumerate(avg_data["Subject"].unique()):
                        subject_data = avg_data[avg_data["Subject"] == subject_name]
                        fig.add_trace(
                            go.Scatter(
                                x=subject_data["Scale"],
                                y=subject_data[entropy_type],
                                mode="lines",
                                line=dict(color=colors[i % len(colors)], width=2),
                                name=subject_name,
                                showlegend=show_legends,
                            ),
                            row=1,
                            col=1,
                        )

                    # Plot collective average
                    fig.add_trace(
                        go.Scatter(
                            x=group_avg_mse["Scale"],
                            y=group_avg_mse[entropy_type],
                            error_y=dict(type="data", array=group_sem_mse[entropy_type], visible=True),
                            mode="lines+markers",
                            line=dict(color="#FFFFFF", width=4),
                            marker=dict(color="#FFFFFF", size=8),
                            name="Group Average",
                        ),
                        row=1,
                        col=1,
                    )

                    # Complexity Index bar plot
                    ci_average = ci_across_subjects_df[ci_across_subjects_df['Channel'] == 'Average']
                    fig.add_trace(
                        go.Bar(
                            x=ci_average["Subject"],
                            y=ci_average["CI"],
                            marker=dict(color="#004A9C", line=dict(color="black", width=1)),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(
                        template="none",
                        width=1200,
                        height=800,
                        font=dict(size=20, color="#4BDCFF"),
                        xaxis=dict(
                            title="Scales",
                            tickfont=dict(color="#FFFFFF"),
                            titlefont=dict(color="#FFFFFF")
                        ),
                        yaxis=dict(
                            tickfont=dict(color="#FFFFFF"),
                            titlefont=dict(color="#FFFFFF")
                        ),
                        xaxis2=dict(
                            tickfont=dict(color="#FFFFFF"),
                            titlefont=dict(color="#FFFFFF")
                        ),
                        yaxis2=dict(
                            tickfont=dict(color="#FFFFFF"),
                            titlefont=dict(color="#FFFFFF")
                        ),
                        title="Group-Level Multichannel Analysis",
                        paper_bgcolor="#0E1117",
                        plot_bgcolor="#0E1117",
                        margin=dict(t=100, b=100),
                    )
                    fig.update_yaxes(title_text=f"{entropy_type}", row=1, col=1)
                    fig.update_yaxes(title_text="Total Entropy", row=2, col=1)
                    # Store results in st.session_state
                    st.session_state['entropy_df'] = entropy_df
                    st.session_state['ci_df'] = ci_across_subjects_df
                    st.session_state['fig'] = fig
                    st.session_state['formatted_time'] = formatted_time
                    st.session_state['csv_file_name'] = "group_entropy_data.csv"

                except Exception as e:
                    st.error(f"An error occurred during group-level analysis: {e}")

            elif analysis_type == "Group-Level Single-Channel":
                try:
                    # Processing multiple files
                    all_data_dict = {}
                    for file in all_files:
                        data = pd.read_csv(file)
                        if data.empty:
                            st.error(f"Uploaded CSV {file.name} is empty.")
                            continue
                        data = data.iloc[: int(trim_length * sfreq), 1:]
                        subject_name = os.path.splitext(file.name)[0]
                        all_data_dict[subject_name] = data

                    mse_single_channel_df = pd.DataFrame()
                    ci_single_channel_df = pd.DataFrame()

                    # Initialize progress bar and status text
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_subjects = len(all_data_dict)
                    processed_subjects = 0

                    # Start timing
                    start_time = time.time()

                    for idx, (subject_name, data) in enumerate(all_data_dict.items()):
                        ch_names = data.columns.tolist()
                        if target_channel not in ch_names:
                            st.error(f"Channel {target_channel} not found in {subject_name}.")
                            continue
                        ch_data = data[target_channel].values

                        # Compute entropy
                        Msx, CI = eh.MSEn(ch_data, Mobj, Scales=num_scales)
                        subject_mse_df = pd.DataFrame(
                            {"Subject": subject_name, "Scale": scales_list, "Channel": target_channel, entropy_type: Msx}
                        )
                        mse_single_channel_df = pd.concat([mse_single_channel_df, subject_mse_df], ignore_index=True)
                        ci_row = pd.DataFrame({"Subject": [subject_name], "Channel": [target_channel], "CI": [CI]})
                        ci_single_channel_df = pd.concat([ci_single_channel_df, ci_row], ignore_index=True)
                        processed_subjects += 1

                        # Update progress bar and status text
                        progress = processed_subjects / total_subjects
                        percent_complete = int(progress * 100)
                        status_text.text(f"Processing subject {subject_name} ({processed_subjects}/{total_subjects}) {percent_complete}%")
                        progress_bar.progress(progress)

                    # End timing
                    end_time = time.time()
                    total_time = end_time - start_time
                    formatted_time = format_time(total_time)

                    st.header("Analysis Complete")
                    st.write(f"Total processing time: {formatted_time}")

                    # Visualization
                    fig = make_subplots(rows=2, cols=1, subplot_titles=(f"MSE Across Subjects ({entropy_type})", "Complexity Index"), vertical_spacing=0.2)

                    if not mse_single_channel_df.empty:
                        avg_mse = (
                            mse_single_channel_df.groupby("Scale")[entropy_type].mean().reset_index()
                        )
                        sem_mse = (
                            mse_single_channel_df.groupby("Scale")[entropy_type].sem().reset_index()
                        )

                        # Define color palette
                        base_colors = ['#4BDCFF', '#00A6FF', '#005DFF', '#1E90FF', '#6495ED', '#7B68EE', '#483D8B', '#191970']
                        colors = base_colors * 10  # Extend color palette if needed

                        # Plot each subject
                        for i, subject_name in enumerate(mse_single_channel_df["Subject"].unique()):
                            subject_data = mse_single_channel_df[
                                mse_single_channel_df["Subject"] == subject_name
                            ]
                            fig.add_trace(
                                go.Scatter(
                                    x=subject_data["Scale"],
                                    y=subject_data[entropy_type],
                                    mode="lines",
                                    line=dict(color=colors[i % len(colors)], width=2),
                                    name=subject_name,
                                    showlegend=show_legends,
                                ),
                                row=1,
                                col=1,
                            )

                        # Plot average
                        fig.add_trace(
                            go.Scatter(
                                x=avg_mse["Scale"],
                                y=avg_mse[entropy_type],
                                error_y=dict(type="data", array=sem_mse[entropy_type], visible=True),
                                mode="lines+markers",
                                line=dict(color="#FFFFFF", width=4),
                                marker=dict(color="#FFFFFF", size=8),
                                name="Average",
                            ),
                            row=1,
                            col=1,
                        )

                        # Complexity Index bar plot
                        fig.add_trace(
                            go.Bar(
                                x=ci_single_channel_df["Subject"],
                                y=ci_single_channel_df["CI"],
                                marker=dict(color="#004A9C", line=dict(color="black", width=1)),
                                showlegend=False,
                            ),
                            row=2,
                            col=1,
                        )

                        fig.update_layout(
                            template="none",
                            width=1200,
                            height=800,
                            font=dict(size=20, color="#4BDCFF"),
                            xaxis=dict(
                                title="Scales",
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            yaxis=dict(
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            xaxis2=dict(
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            yaxis2=dict(
                                tickfont=dict(color="#FFFFFF"),
                                titlefont=dict(color="#FFFFFF")
                            ),
                            title=f"Group-Level Single-Channel Analysis ({target_channel})",
                            paper_bgcolor="#0E1117",
                            plot_bgcolor="#0E1117",
                            margin=dict(t=100, b=100),
                        )
                        fig.update_yaxes(title_text=f"{entropy_type}", row=1, col=1)
                        fig.update_yaxes(title_text="Total Entropy", row=2, col=1)
                        # Store results in st.session_state
                        st.session_state['entropy_df'] = mse_single_channel_df
                        st.session_state['ci_df'] = ci_single_channel_df
                        st.session_state['fig'] = fig
                        st.session_state['formatted_time'] = formatted_time
                        st.session_state['csv_file_name'] = "group_single_channel_entropy_data.csv"

                    else:
                        st.warning("No data available to plot.")
                except Exception as e:
                    st.error(f"An error occurred during group-level analysis: {e}")

            else:
                st.error("Invalid analysis type selected.")

    # Display the results if they exist in st.session_state
    if st.session_state.get('fig') is not None:
        st.header("Analysis Complete")
        st.write(f"Total processing time: {st.session_state['formatted_time']}")
        st.plotly_chart(st.session_state['fig'])

        # Download results
        csv = st.session_state['entropy_df'].to_csv(index=False).encode()
        st.download_button(
            label="Download Entropy Data as CSV",
            data=csv,
            file_name=st.session_state.get('csv_file_name', 'entropy_data.csv'),
            mime="text/csv",
        )
    else:
        st.info("Upload data files or select a sample dataset, set parameters, then click 'Run Analysis'.")

# # ---------------------- Tab 3: Entropy Education ----------------------
with tab3:
    st.header("Entropy Education")

    st.write("""
    Welcome to the Entropy Suite Education section. Here, we explore various entropy measures available in this application. These entropy measures are sourced from **EntropyHub**, an open-source toolkit that integrates numerous established entropy methods into a single package. EntropyHub provides robust and efficient implementations of entropy algorithms, making it easier for researchers and practitioners to analyze the complexity of time-series data. Documentation for EntropyHub can be retrieved at https://www.entropyhub.xyz/Home.html.

    Below, you will find detailed explanations of each entropy measure, including their theoretical foundations, parameter settings, and applications.
    """)

    entropy_measures = [
        "Sample Entropy (SampEn)",
        "Fuzzy Entropy (FuzzEn)"
    ]

    for measure in entropy_measures:
        with st.expander(measure):
            st.subheader("Overview")
            if measure == "Sample Entropy (SampEn)":
                st.write("""
                **Sample Entropy** quantifies the unpredictability or complexity of a time-series by measuring the likelihood that similar patterns in the data will remain similar on the next incremental comparison. It is less biased than Approximate Entropy and is widely used in physiological signal analysis.
                """)
            elif measure == "Fuzzy Entropy (FuzzEn)":
                st.write("""
                **Fuzzy Entropy** extends Sample Entropy by incorporating fuzzy membership functions, allowing for a smoother transition between similarity and dissimilarity of patterns. This makes it more robust to noise and short data lengths.
                """)

            st.subheader("Parameters Explanation")
            if measure == "Sample Entropy (SampEn)":
                st.markdown("""
                - **Embedding Dimension (`m`)**: Length of sequences to be compared.
                - **Time Delay (`tau`)**: Time delay between points in sequences. 
                - **Tolerance (`r`)**: Distance threshold for considering sequences similar, typically a percentage of the standard deviation.
                """)
            elif measure == "Fuzzy Entropy (FuzzEn)":
                st.markdown("""
                - **Embedding Dimension (`m`)**: Length of sequences to be compared.
                - **Time Delay (`tau`)**: Time delay between points in sequences.
                - **Tolerance (`r`)**: Distance threshold for considering sequences similar, typically a percentage of the standard deviation.
                - **Fuzzy Function (`Fx`)**: Type of fuzzy function for distance transformation. 
                - **Parameters (`r`)**: Parameters of the fuzzy function specified by `Fx`.
                """)

            st.subheader("Default Values and Recommendations")
            st.write("""
            - **Embedding Dimension (`m`)**: Typically set to 2 or 3. **Default: 2**
            - **Time Delay (`tau`)**: Usually set to 1 for consecutive points. **Default: 1**
            - **Tolerance (`r`)**: Commonly set to 0.1 to 0.25 times the standard deviation of the data. **Default: 0.2**
            """)
            if measure == "Fuzzy Entropy (FuzzEn)":
                st.write("""
                - **Fuzzy Function (`Fx`)**: **Default: 'default'**
                - **Parameters (`r`)**: Should be adjusted based on the specific fuzzy function used. **Default: [0.2, 2]**
                """)

            st.subheader("References")
            if measure == "Sample Entropy (SampEn)":
                st.write("""
                - Richman, J. S., & Moorman, J. R. (2000). [Physiological time-series analysis using approximate entropy and sample entropy](https://doi.org/10.1152/ajheart.2000.278.6.H2039). *American Journal of Physiology-Heart and Circulatory Physiology*, 278(6), H2039-H2049.
                """)
            elif measure == "Fuzzy Entropy (FuzzEn)":
                st.write("""
                - Chen, W., Zhuang, J., Yu, W., & Wang, Z. (2009). [Measuring complexity using FuzzyEn, ApEn, and SampEn](https://doi.org/10.1109/TSMCC.2008.2007124). *IEEE Transactions on Systems, Man, and Cybernetics*, 39(6), 642-652.
                """)

# Footer
st.markdown("---")
st.markdown("Developed by Tanner Creel")
