import streamlit as st
import os
import pandas as pd
import numpy as np
import aerosandbox as asb
import tempfile
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# Function to process individual DAT file
def process_dat_file(file_path, original_filename, alpha_range, Re, mach, selected_columns, progress_bar):
    af = asb.Airfoil(file_path)
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], int(alpha_range[2]))
    data = af.get_aero_from_neuralfoil(alpha=alpha_values, Re=Re, mach=mach)
    df = pd.DataFrame(data)
    df['alpha'] = alpha_values
    df['filename'] = original_filename

    mandatory_columns = ['filename', 'analysis_confidence', 'alpha', 'CL', 'CD', 'CM']
    if not selected_columns:
        columns_to_include = mandatory_columns
    else:
        columns_to_include = list(set(mandatory_columns + selected_columns))

    ind_data = df[columns_to_include]

    step_size = 100 / int(alpha_range[2])
    for i in tqdm(range(int(alpha_range[2])), desc=f"Processing  {original_filename}", leave=False):
        progress_value = (i + 1) * step_size / 100
        progress_bar.progress(progress_value)
    
    return ind_data


@st.cache_data(ttl=None)
def load_pre_existing_data(directory):
    data_frames = []
    file_list = [filename for filename in os.listdir(directory) if filename.endswith(".dat")]
    with tqdm(total=len(file_list), desc="Loading data from our database") as pbar:
        for filename in file_list:
            file_path = os.path.join(directory, filename)
            try:
                dat_data = process_dat_file(file_path, filename, (-20.0, 25.0, 451), 3460660, 0.0, [], pbar)
                data_frames.append(dat_data)
            except Exception as e:
                st.warning(f"Error processing {filename}: {str(e)}")
            pbar.update(1)
    return data_frames


def process_airfoil_data(data_frames, rule, threshold=None):
    filtered_dfs = []
    for df in data_frames:
        df['CL/CD'] = df['CL'] / df['CD']
        if rule == 'max_cl':
            max_cl_value = df['CL'].max()
            filtered_df = df[df['CL'] == max_cl_value]
        elif rule == 'min_cd':
            min_cd_value = df['CD'].min()
            filtered_df = df[df['CD'] == min_cd_value]
        elif rule == 'cl_0':
            filtered_df = df[df['alpha'] == 0]
        elif rule == 'max_clcd':
            max_clcd_ratio = df['CL/CD'].max()
            filtered_df = df[df['CL/CD'] == max_clcd_ratio]
        elif rule == 'high_alpha':
            filtered_df = df[df['alpha'] == threshold]
        filtered_dfs.append(filtered_df)
    combined_filtered_data = pd.concat(filtered_dfs, ignore_index=True)
    if combined_filtered_data is not None:
        df2 = pd.DataFrame(combined_filtered_data)
        return df2.drop_duplicates()
    else:
        return pd.DataFrame()


def airfoil_score(df2, weights):
    airfoil_scores = {}
    for airfoil, group_df in df2.groupby('filename'):
        score = (
            group_df['CL'] * weights['w1'] +
            group_df['CD'] * weights['w2'] +
            group_df['CM'] * weights['w3'] +
            group_df['alpha'] * weights['w4'] +
            group_df['CL/CD'] * weights['w5']
        ).sum()
        airfoil_scores[airfoil] = score

    scores_df = pd.DataFrame(list(airfoil_scores.items()), columns=['Airfoil', 'Score'])
    return scores_df


def normalize_aerodynamic_data(data):
    numerical_columns = ['alpha', 'CL', 'CD', 'CM', 'Cpmin', 'Top_Xtr', 'Bot_Xtr', 'mach_crit', 'CL/CD']
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data


def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Airfoil Tool", "Project Details"])

    if app_mode == "Airfoil Tool":
        st.title("Airfoil Selection Tool")

        alpha_min = st.number_input("Minimum Alpha:", value=-20)
        alpha_max = st.number_input("Maximum Alpha:", value=25)
        alpha_num = st.number_input("Total Alpha values:", value=451)
        Re = st.number_input("Reynolds Number:", value=3460660)
        mach = st.number_input("Mach Number:", value=0.0)

        mandatory_columns = ['filename', 'analysis_confidence', 'alpha', 'CL', 'CD', 'CM', 'CL/CD']
        available_columns = ['Cpmin', 'Top_Xtr', 'Bot_Xtr', 'mach_crit', 'mach_dd']
        selected_columns = st.multiselect("Select additional data columns to include:", available_columns)

        if 'uploaded_data_frames' not in st.session_state:
            st.session_state.uploaded_data_frames = []

        uploaded_files = st.file_uploader("Upload .dat files", type='dat', accept_multiple_files=True)
        if uploaded_files:
            progress_bar = st.progress(0)
            for uploaded_file in uploaded_files:
                file_contents = uploaded_file.read()
                file_name = uploaded_file.name
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_contents)
                    temp_file_path = temp_file.name

                try:
                    dat_data = process_dat_file(temp_file_path, file_name, (alpha_min, alpha_max, alpha_num), Re, mach, selected_columns, progress_bar)
                    st.session_state.uploaded_data_frames.append(dat_data)
                    st.success(f"Processed data for {file_name}")
                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                
                os.remove(temp_file_path)
            
            progress_bar.empty()

        use_pre_existing_data = st.checkbox("Use Airfoils from our Database")
        if use_pre_existing_data:
            pre_existing_data_directory = r"C:\Users\shrey\OneDrive\Desktop\Phinal year project\production\dat_files"
            pre_existing_data_frames = load_pre_existing_data(pre_existing_data_directory)
            st.success(f"Loaded {len(pre_existing_data_frames)} pre-existing data files.")
        else:
            pre_existing_data_frames = []

        all_data_frames = st.session_state.uploaded_data_frames + pre_existing_data_frames

        if all_data_frames:
            rule_options = ['max_cl', 'min_cd', 'cl_0', 'max_clcd', 'high_alpha']
            rule = st.selectbox("Select filtering rule:", rule_options)

            if rule == 'high_alpha':
                threshold = st.number_input("Enter value of alpha:", value=15.0)
            else:
                threshold = None

            if st.button("Generate Data"):
                combined_filtered_data = process_airfoil_data(all_data_frames, rule, threshold=threshold)
                st.session_state.combined_filtered_data = combined_filtered_data

                if selected_columns:
                    columns_to_include = list(set(mandatory_columns + selected_columns))
                    st.write("Selected Filtered Airfoil Data:")
                    st.dataframe(combined_filtered_data[columns_to_include])
                else:
                    st.write("Combined Filtered Airfoil Data:")
                    st.dataframe(combined_filtered_data)

                st.session_state.filtered_data_ready = True

        if st.session_state.get('filtered_data_ready', False):
            st.header("Scoring Metric")

            weights = {
                'w1': st.number_input("Weight for CL", value=0.2),
                'w2': st.number_input("Weight for CD", value=0.3),
                'w3': st.number_input("Weight for CM", value=0.1),
                'w4': st.number_input("Weight for Alpha", value=0.2),
                'w5': st.number_input("Weight for CL/CD", value=0.2)
            }

            if st.button("Calculate Scores"):
                combined_filtered_data = st.session_state.combined_filtered_data
                scores_df = airfoil_score(combined_filtered_data, weights)
                st.write("Airfoil Scores:")
                st.dataframe(scores_df)
                top_n = st.selectbox("Select number of top airfoils to display:", [5, 10, 20])
                top_scores_df = scores_df.nlargest(top_n, 'Score')
                st.write(f"Top {top_n} Airfoil Scores:")
                st.dataframe(top_scores_df)

    elif app_mode == "Project Details":
        st.title("Project Details")
        st.write("""
            ## Airfoil Selection Tool
            
            This project involves the development of an **airfoil selection tool** that helps users to choose the best airfoil based on customized aerodynamic performance criteria. It supports 
            **1600+ airfoils** in the database which enables user to start from scratch using our own airfoil database as per the requirement.

            ### Features:
            - **Upload airfoil data files (.dat)**: Allows users to upload their own airfoil data files for analysis as well as can use our data base of airfoils.
            - **Specify aerodynamic conditions**: Users can set parameters such as angle of attack, Reynolds number, and Mach number.
            - **Filter airfoils based on criteria**: Options include filtering by maximum lift coefficient, minimum drag coefficient, and other criteria.
            - **Scoring system**: Customizable weights for different aerodynamic metrics to score and rank airfoils.

            ### Future Work:
            - **Integrate advanced aerodynamic models**: Incorporate more sophisticated analysis and airfoil optimization methods for better accuracy.
            - **Enhanced user interface**: Improve the usability and functionality of the tool.
            - **Cloud integration**: Allow saving and sharing of results through cloud storage options.

            ### Team:
            - **Group - 4**, Final year project 
            - **Ankit**, **Ashutosh**, **Prity**, **Shreyas** from Mechanical Engineering (2020-2024)
            - Guided by **Prof. (Dr.) Rajan Kumar**, Department of Mechanical Engineering, BIT Sindri

        """)

if __name__ == '__main__':
    main()
