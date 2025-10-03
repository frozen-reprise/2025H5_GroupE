# Welcome to ZZSC9020 GitHub repository for group [2025H5_GroupE]

This GitHub repository is the main point of access for students and lecturers of the ZZSC9020 capstone course. 

In this repository, you will find the data to start developing your project. Also, we will use the repository to share code, documentation, data, models and other resources between the group members and course lecturers.

Complete the information below regarding your group.

## Group and project information

### Group members and zIDs
- Shaun Stephenson (z5608969) - Group leader
- Arturo Ronda (z5342043)
- Kelly Xu(z5214516) 
- Rahel Legesse(z5492537)


### Brief project description

This project investigates forecasting methods for medium-term electricity demand in New South Wales to support the operation of the Snowy 2.0 pumped-hydro scheme. Using 2010–2021 data on demand, weather, and public holidays, we constructed a comprehensive dataset and conducted exploratory analysis to identify long-term declines, seasonal cycles, and non-linear temperature–demand effects. Two models were implemented and compared: a SARIMAX model incorporating seasonal and exogenous drivers, and a Convolutional Neural Network (CNN) designed to capture complex non-linear patterns. Evaluation across 1–28 day horizons showed that SARIMAX performs adequately for short horizons but deteriorates quickly beyond a week, whereas CNN forecasts remain more accurate and stable, with mean errors around 3–4% of peak demand. These findings highlight the value of advanced forecasting for Snowy 2.0’s scheduling decisions and Australia’s broader renewable energy transition.

## Repository structure

The repository has the following folder structure:

- agendas: agendas for each weekly meeting with lecturers (left 24h before the next meeting)
- checklists: teamwork checklist or a link to an account in a project task management tool
- data: datasets for analysis
- gantt_chart: Gantt chart or a link to an account in a project task management tool
- minutes: minutes for each meeting (left not more than 24h after the corresponding meeting)
- report: RMarkdown or Jupyter notebook report in progress
- src: source code


## Instruction on how to run 2025H5_groupE_EDA.ipynb
1) Clone or download this repository.
2) Download the data zip from the latest GitHub Release:
     - Go to the repository?s "Releases" page.
     - Download `data_raw.zip`.
3) Unzip at the repository root so you end up with:
     data/raw/
       totaldemand_nsw.csv
       temperature_nsw.csv
       forecastdemand_nsw.csv
       rn/  <-- Renewables Ninja hourly CSVs (humidity, rn_temp_c, irradiance,
               precipitation, cloud, wind,)
4) Run the EDA pipeline headlessly:
     chmod +x run.sh        # first time only
     ./run.sh 2025H5_groupE_EDA.ipynb
5) Outputs will appear under:
     outputs/
       executed_2025H5_groupE_EDA.ipynb
       Final_master_daily_with_env_v1.csv
       

### What ./run.sh Does
------------------
- Creates/uses a local Python virtual environment at ./.venv
- Installs all dependencies listed in requirements.txt
- Executes the specified notebook headlessly using Papermill:
    ./run.sh <path/to/notebook.ipynb>
  If no argument is provided, the script looks for a notebook in the repo root.

Default usage for this project:
  ./run.sh 2025H5_groupE_EDA.ipynb


### Outputs
-------
- outputs/executed_2025H5_groupE_EDA.ipynb   (notebook with all cells executed)
- outputs/Final_master_daily_with_env_v1.csv (daily modeling master table)
- forecast_30min_clean.csv
- temperature_30min_clean.csv
- demand_30min_clean.csv

Note: The repository?s .gitignore excludes outputs/ and data/raw/ from commits.


### System Requirements
-------------------
- macOS or Linux with bash
- Python 3.9+ available as `python3` (the script falls back to `python` if needed)
- Internet access during first run (to install Python packages)


### Overriding Data Paths (Optional)
--------------------------------
If your files are in different locations, you can override inputs at runtime:

  DEMAND_CSV="/full/path/totaldemand_nsw.csv" \
  TEMP_CSV="/full/path/temperature_nsw.csv" \
  RN_DIR="/full/path/rn" \
  ./run.sh 2025H5_groupE_EDA.ipynb

The notebook will respect these environment variables.
