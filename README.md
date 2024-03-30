
# DATA ANALYSIS APP 📊

This Streamlit web application performs various data analysis tasks on uploaded CSV or Excel files. It provides features for summarizing statistics, conducting ANOVA tests, performing t-tests, regression analysis, correlation analysis, and data visualization.

## Getting Started 🚀

To run this application locally, follow these steps:

1. Install the necessary libraries by running:

    ```bash
    pip install streamlit pandas numpy scipy statsmodels plotly seaborn matplotlib
    ```

2. Save the provided code into a Python file (e.g., `data_analysis_app.py`).

3. Run the Streamlit app by executing:

    ```bash
    streamlit run data_analysis_app.py
    ```

4. Upload your CSV or Excel file containing the data for analysis.

## Features 🛠️

### Section Zero: Upload and Encode Data

Upload your CSV or Excel file and view the encoded data. The application automatically encodes categorical variables for analysis.

### Section One: Data Analysis

- **Summary Statistics:** View descriptive statistics of the uploaded dataset.
- **Pivoting Data:** Enable pivoting and select columns for a pivoted view of the data.
- **Box Plot:** Visualize the distribution of selected columns using box plots.

### Section Two: ANOVA Test

- Perform one-way or two-way ANOVA tests to analyze variance between groups.

### Section Three: T-Test

- Conduct t-tests between two selected columns to compare means.

### Section Four: Regression Analysis

- Perform linear regression analysis between selected independent and dependent variables.

### Section Five: Correlation Analysis

- Generate correlation matrices and analyze the relationships between selected columns.

### Section Six: T-Test for Two Groups

- Conduct t-tests between two groups within a selected column.

### Section Seven: Data Visualization

- Visualize data using various plots including histograms, pie charts, bar graphs, line plots, box plots, scatter plots, violin plots, and map plots.
- Generate correlation matrices and heatmaps for visualizing correlations between variables.

## Requirements 📋

- Python 3.x
- Streamlit
- Pandas
- NumPy
- SciPy
- Statsmodels
- Plotly
- Seaborn
- Matplotlib

## Contributing 🤝

Contributions to enhance the functionality or fix any issues are welcome. Please feel free to submit a pull request.

## License 📜

This project is licensed under the [MIT License](LICENSE).
