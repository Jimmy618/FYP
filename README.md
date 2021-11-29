# Introduction
This project intend to perform statistical analysis and apply machine learning technique on SSDs failure data
Due to file size limit imposed by github, the dataset could not be included in this repository. Please follow the instruction in Statistical Analysis setup to obtain the dataset
# Statistical Analysis
## setup
This FYP use the dataset from Alibaba. Please proceed to the following link to obtain location_info_of_ssd.csv, 20191231.csv from smart_log_20191231.csv.zip and ssd_failure_tag.csv. The three .csv files should be put on the folder dataset provided.
https://github.com/alibaba-edu/dcbrain/tree/master/ssd_open_data

If conda is installed in your machine, you may run the following command to install necessary package to your conda. Otherwise you may install the corresponding package to your python
```
conda install pandas matplotlib numpy scikit-learn
```

## Usage
Please change directory to staticalAnalysis. Then you may running the statistical analysis using the following command.
```
cd ./staticalAnalysis
python3 ./staticalAnalysis
```
Relevant plot could be generated and displayed by using the following command. Note that the statistical analysis is not generated in this manner.
```
python3 ./staticalAnalysis graph
```

# Links
Link to dataset: https://github.com/alibaba-edu/dcbrain/tree/master/ssd_open_data

Link to report site (login required): https://mycuhk.sharepoint.com/sites/Faculty.ERG/FYP/2021-22/CSE/PCL2101/SitePages/Home.aspx
