import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load and cache the dataset
@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

# Function to encode data
def encode_data(data):
    encoded_data = data.apply(lambda x: pd.factorize(x)[0] if x.dtype == "O" else x)
    return encoded_data

# Function to perform t-test and return results
def perform_t_test(column1, column2):
    result = stats.ttest_ind(column1, column2, nan_policy='omit')
    return result

# Function to check if the DataFrame is not None and not empty
def is_valid_dataframe(df):
    return df is not None and not df.empty

# Main Streamlit app
def main():
    st.title("Data Analysis App")

    # Section Zero: Upload and Encode Data
    st.header("Section Zero: Upload and Encode Data")

    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
    df = load_data(uploaded_file)

    if is_valid_dataframe(df):
        # Encode data if necessary
        df_encoded = encode_data(df)
        st.subheader("Encoded Data:")
        st.write(df_encoded)

        # Section One: Data Analysis
        st.header("Section One: Data Analysis")

        st.subheader("Summary Statistics:")
        summary_stats = df_encoded.describe()
        st.write(summary_stats)

        # Pivoting option
        pivot_option = st.checkbox("Enable Pivoting", value=False)
        if pivot_option:
            pivot_columns = st.multiselect("Select columns for pivoting", df.columns)
            if pivot_columns and is_valid_dataframe(df_encoded):
                df_pivot = df_encoded.pivot_table(index=pivot_columns, aggfunc="mean")
                st.write("Pivoted Data:")
                st.write(df_pivot)

        # Box plot
        st.subheader("Box Plot:")
        selected_columns = st.multiselect("Select columns for box plot", df_encoded.columns)
        if selected_columns and is_valid_dataframe(df_encoded):
            fig = px.box(df_encoded[selected_columns])
            st.plotly_chart(fig)

    # Section Two: ANOVA Test
    st.header("Section Two: ANOVA Test")

    if df is not None and not df.empty:
        # Select variable for ANOVA
        anova_variable = st.selectbox("Select variable for ANOVA", df.columns)

        # Allow user to select 'Group' column
        group_column = st.selectbox("Select the 'Group' column", df.columns)

        # List of ANOVA tests
        anova_tests = ["One-Way ANOVA", "Two-Way ANOVA"]  # Add more if needed
        selected_anova_test = st.selectbox("Select ANOVA test", anova_tests)

        # Perform ANOVA
        if st.button("Perform ANOVA"):
            st.subheader("ANOVA Result:")

            # Attempt to convert only numeric values to float
            try:
                numeric_values = pd.to_numeric(df[group_column], errors="coerce")
                result = stats.f_oneway(*[df[anova_variable][numeric_values.notnull()][numeric_values == group].astype(float) for group in numeric_values.unique()])
                st.table(pd.DataFrame({"Statistic": [result.statistic], "P-value": [result.pvalue]}))
            except ValueError:
                st.warning(f"The '{group_column}' column contains non-numeric values and cannot be used for ANOVA.")

            if selected_anova_test == "Two-Way ANOVA":
                formula = f"{anova_variable} ~ {' + '.join(df.columns.difference([group_column]))}"
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.table(anova_table)

    # Section Three: T-Test
    st.header("Section Three: T-Test")

    # Select two columns for t-test
    t_test_column1 = st.selectbox("Select the first column for t-test", df.columns) if df is not None else None
    t_test_column2 = st.selectbox("Select the second column for t-test", df.columns) if df is not None else None

    # Convert selected columns to numeric
    if df is not None and not df.empty:
        df[t_test_column1] = pd.to_numeric(df[t_test_column1], errors="coerce")
        df[t_test_column2] = pd.to_numeric(df[t_test_column2], errors="coerce")

    # Perform t-test
    
    if st.button("Perform t-test") and t_test_column1 is not None and t_test_column2 is not None:
        st.subheader("t-test Result:")
        
        # Extract data for selected columns
        column1_data = df[t_test_column1].dropna()
        column2_data = df[t_test_column2].dropna()
        
        # Perform t-test
        result_t_test = perform_t_test(column1_data, column2_data)

        # Display t-test result
        st.write(f"T-statistic: {result_t_test.statistic}")
        st.write(f"P-value: {result_t_test.pvalue}")

    # Section Four: Regression Analysis
    st.header("Section Four: Regression Analysis")

    def perform_regression_analysis(df, independent_variable, dependent_variable):
        try:
            X = sm.add_constant(df[independent_variable])
            y = df[dependent_variable]
            model = sm.OLS(y, X).fit()
            return model
        except Exception as e:
            return f"Error: {e}"

    # Select independent and dependent variables
    independent_variable = st.selectbox("Select the independent variable", df.columns) if df is not None else None
    dependent_variable = st.selectbox("Select the dependent variable", df.columns) if df is not None else None

    # Perform linear regression analysis
    if st.button("Perform Regression Analysis") and independent_variable is not None and dependent_variable is not None:
        result_regression = perform_regression_analysis(df, independent_variable, dependent_variable)

        # Display regression analysis result
        st.subheader("Regression Analysis Result:")

        if isinstance(result_regression, sm.regression.linear_model.RegressionResultsWrapper):
            result_dict_regression = {
                "Intercept": [result_regression.params['const']],
                f"{independent_variable} Coefficient": [result_regression.params[independent_variable]],
                "R-squared": [result_regression.rsquared],
                "P-value": [result_regression.f_pvalue],
            }
            result_df_regression = pd.DataFrame(result_dict_regression)
            st.table(result_df_regression)
        else:
            st.error(result_regression)

    # Section Five: Correlation Analysis
    st.header("Section Five: Correlation Analysis")

    # Function to perform correlation analysis and return results
    def perform_correlation_analysis(df, selected_columns):
        correlation_matrix = df[selected_columns].corr()
        return correlation_matrix

    # Select columns for correlation analysis
    selected_columns_correlation = st.multiselect("Select columns for correlation analysis", df.columns) if df is not None else None

    # Initialize result_correlation_matrix with an empty DataFrame
    result_correlation_matrix = pd.DataFrame()

    # Perform correlation analysis
    if st.button("Perform Correlation Analysis") and selected_columns_correlation is not None:
        result_correlation_matrix = perform_correlation_analysis(df, selected_columns_correlation)

        # Display correlation analysis result
        st.subheader("Correlation Analysis Result:")
        st.write(result_correlation_matrix)

    # Section Six: T-Test for Two Groups
    st.header("Section Six: T-Test for Two Groups")

    # Allow user to input p-value
    p_value_t_test_two_groups = st.number_input("Enter the desired p-value for T-Test", min_value=0.001, max_value=0.1, value=0.05, step=0.001)

    # Select column for T-Test
    t_test_column_two_groups = st.selectbox("Select column for T-Test between Two Groups", df.columns) if df is not None else None

    # Perform T-Test for Two Groups
    if st.button("Perform T-Test for Two Groups") and t_test_column_two_groups is not None:
        st.subheader("T-Test Result for Two Groups:")

        try:
            group_values = df[t_test_column_two_groups].unique()
            if len(group_values) == 2:
                group1_data = df[df[t_test_column_two_groups] == group_values[0]]
                group2_data = df[df[t_test_column_two_groups] == group_values[1]]

                result_t_test_two_groups = stats.ttest_ind(group1_data, group2_data, nan_policy='omit')

                # Display T-Test result
                st.write(f"T-statistic: {result_t_test_two_groups.statistic}")
                st.write(f"P-value: {result_t_test_two_groups.pvalue}")

            else:
                st.warning("Please select a column with exactly two unique values for T-Test between two groups.")
        except Exception as e:
            st.warning(f"Error: {e}")

    # Section Seven: Data Visualization
    st.header("Section Seven: Data Visualization")

    # Sub-section A: Visualize DataFrame Data
    st.subheader("Sub-section A: Visualize DataFrame Data")

    # Select columns for visualization
    selected_columns_visualization = st.multiselect("Select columns for visualization", df.columns) if df is not None else None

    # Visualization options
    visualization_options = ["Histogram", "Pie Chart", "Bar Graphs with Error Bars", "Line Plot", "Box Plot", "Scatter Plot", "Violin Plot", "Map Plot"]
    selected_visualization_option = st.multiselect("Select visualization options", visualization_options)

    # Visualize selected columns based on user options
    if st.button("Visualize Data") and selected_columns_visualization is not None:
        for option in selected_visualization_option:
            if option == "Histogram":
                st.subheader("Histogram:")
                for column in selected_columns_visualization:
                    fig = px.histogram(df, x=column, title=f'Histogram of {column}')
                    st.plotly_chart(fig)

            elif option == "Pie Chart":
                st.subheader("Pie Chart:")
                for column in selected_columns_visualization:
                    fig = px.pie(df, names=column, title=f'Pie Chart of {column}')
                    st.plotly_chart(fig)

            elif option == "Bar Graphs with Error Bars":
                    st.subheader("Bar Graphs with Error Bars:")
                    for column in selected_columns_visualization:
                        fig, ax = plt.subplots()
                        means = df.groupby(group_column)[column].mean()
                        sems = df.groupby(group_column)[column].sem()
                        means.plot(kind='bar', yerr=sems, ax=ax)
                        ax.set_ylabel(column)
                        ax.set_title(f'Bar Graph with Error Bars of {column}')
                        st.pyplot(fig)

            elif option == "Line Plot":
                st.subheader("Line Plot:")
                for column in selected_columns_visualization:
                    fig = px.line(df, x=df.index, y=column, title=f'Line Plot of {column}')
                    st.plotly_chart(fig)

            elif option == "Box Plot":
                st.subheader("Box Plot:")
                for column in selected_columns_visualization:
                    fig = px.box(df, y=column, title=f'Box Plot of {column}')
                    st.plotly_chart(fig)

            elif option == "Scatter Plot":
                st.subheader("Scatter Plot:")
                if len(selected_columns_visualization) == 2:
                    fig = px.scatter(df, x=selected_columns_visualization[0], y=selected_columns_visualization[1], title=f'Scatter Plot between {selected_columns_visualization[0]} and {selected_columns_visualization[1]}')
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select exactly two columns for Scatter Plot.")

            elif option == "Violin Plot":
                st.subheader("Violin Plot:")
                for column in selected_columns_visualization:
                    fig = px.violin(df, y=column, box=True, points="all", title=f'Violin Plot of {column}')
                    st.plotly_chart(fig)

            elif option == "Map Plot":
                st.subheader("Map Plot:")
                if "latitude" in df.columns and "longitude" in df.columns:
                    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name=df.index, title="Map Plot")
                    
                    # Allow users to customize map style
                    map_style = st.selectbox("Select Map Style", ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner", "stamen-watercolor"])
                    fig.update_layout(mapbox_style=map_style)
                    
                    # Allow users to customize map appearance
                    zoom_level = st.slider("Zoom Level", min_value=1, max_value=15, value=10)
                    fig.update_layout(mapbox_zoom=zoom_level)
                    
                    # Display the map plot
                    st.plotly_chart(fig)
                else:
                    st.warning("Latitude and Longitude columns are required for Map Plot. Make sure your DataFrame has these columns.")

    # Sub-section B: Generate Correlation Matrix
    st.subheader("Sub-section B: Generate Correlation Matrix")

    # Select columns for correlation matrix
    selected_columns_correlation_matrix = st.multiselect("Select columns for correlation matrix", df.columns) if df is not None else None

    # Generate correlation matrix
    correlation_matrix_legend = st.checkbox("Show Legend for Correlation Matrix", value=True)
    if st.button("Generate Correlation Matrix") and selected_columns_correlation_matrix is not None:
        result_correlation_matrix = df[selected_columns_correlation_matrix].corr()

    # Display correlation matrix
    st.subheader("Correlation Matrix:")
    st.write(result_correlation_matrix)

    # Display legend if selected
    if correlation_matrix_legend:
        st.subheader("Correlation Matrix Legend:")
        st.write("Legend values range from -1 (perfect negative correlation) to 1 (perfect positive correlation). 0 indicates no correlation.")
    # Sub-section C: Generate Heatmap
        st.subheader("Sub-section C: Generate Heatmap")

        # Select columns for heatmap
        selected_columns_heatmap = st.multiselect("Select columns for heatmap", df.columns) if is_valid_dataframe(df) else None

        # Generate heatmap
        if st.button("Generate Heatmap") and is_valid_dataframe(df) and selected_columns_heatmap is not None:
            heatmap_data = df[selected_columns_heatmap].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
