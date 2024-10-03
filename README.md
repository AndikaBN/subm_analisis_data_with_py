## Setup Environment - Anaconda
```
conda create --name main-ds python=3.9
```
conda activate main-ds
```
pip install -r requirements.txt
```

## Installation
1. Clone this repository to your local machine:
```
git clone https://github.com/AndikaBN/subm_analisis_data_with_py.git
```
2. Go to the project directory
```
cd subm_analisis_data_with_py
```
3. Install the required Python packages by running:
```
pip install -r requirements.txt
```

## Usage

1. **Data Wrangling**: Data wrangling scripts are available in the `notebook.ipynb` file to prepare and clean the data.

2. **Exploratory Data Analysis (EDA)**: Explore and analyze the data using the provided Python scripts. EDA insights can guide your understanding of e-commerce public data patterns.

3. **Visualization**: Run the Streamlit dashboard for interactive data exploration:

```
cd subm_analisis_data_with_py/Dashboard
streamlit run dashboard.py