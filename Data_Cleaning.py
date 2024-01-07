##########################################################
# Importing libraries
##########################################################
import pandas as pd, kaggle, requests

#################################################
# Data Download  from "DATA.GOV.AU" through API
#################################################

data_df = requests.get(
    "https://data.gov.au/data/api/3/action/package_show?id=488ef6d4-c763-4b24-b8fb-9c15b67ece19").json()
aihw_data_df = requests.get(
    "https://data.gov.au/data/api/3/action/package_show?id=05696f6f-6ff5-42a2-904f-af5e4d1f56f8").json()

url = data_df['result']['resources'][0]['url']
aihw_url = aihw_data_df['result']['resources']
csv = []
for i in range(0, len(aihw_url)):
    csv.append(aihw_url[i]['url'])

open('Project_Datasets/AIHW_Data/General Record of Incidence of Mortality.csv', 'wb').write(requests.get(url).content)
open('Project_Datasets/AIHW_Data/ACIM Combined Counts.csv', 'wb').write(requests.get(csv[0]).content)
open('Project_Datasets/AIHW_Data/ACIM Combined Rates.csv', 'wb').write(requests.get(csv[1]).content)
open('Project_Datasets/AIHW_Data/ACIM Combined Ratio.csv', 'wb').write(requests.get(csv[2]).content)

ISIC_data = pd.read_json('Project_Datasets/GCO_Dataset.json')
ISIC_data.to_csv('Project_Datasets/WHO_Data/GCO_Dataset.csv', index=False)

##########################################################
# Data Cleaning
##########################################################
projected = pd.read_csv('Project_Datasets/WHO_Data/GCO_Dataset.csv').drop(
    columns=['population', 'risk', 'type', 'sex', 'id', 'cancer', 'APC', 'cases_base'])
projected.rename(
    columns={'year': 'Year', 'id_label': 'Country', 'cancer_label': 'Cancer_Type', 'cases_pred': "Predicted_Cases",
             'pop': 'Population', 'change': 'Predicted_Increase',
             'percent': 'Predicted_Percentage_Change'}).to_csv('Cleaned_data/GCO_Pred_Data.csv', index=False)

WHO_data = pd.read_csv(
    'Project_Datasets/WHO_Data/WHOMortalityDatabase_Map_Melanoma and other skin cancers_19th December 2023 23 15.csv',
    skiprows=6, index_col=False)
WHO_data = WHO_data.drop(columns=['Region Code', 'Region Name', 'Country Code', 'Age group code']).fillna(0).replace(
    "T?rkiye", "Turkey")
WHO_data['Age Group'] = WHO_data['Age Group'].str.strip('[]')
WHO_data.to_csv("Cleaned_data/WHO_Dataset.csv", index=False)

GRIM_data = pd.read_excel('Project_Datasets/AIHW_Data/General Record of Incidence of Mortality.xlsx')
grim_df = GRIM_data.drop(columns=['grim']).fillna(0).replace(["Missing", "Total", "Skin cancer (ICD-10 C43, C44)"],
                                                             ["Unknown", "All", "Skin Cancer"])
grim_df = grim_df[grim_df['cause_of_death'].isin(['Skin Cancer'])].rename(columns={'cause_of_death': 'Cause of '
                                                                                                     'Mortality'})
grim_df.to_excel('Cleaned_data/GRIM_Mortality_Data.xlsx', index=False)

Cancer_data = pd.read_excel('Project_Datasets/AIHW_Data/aihw-can-122-cancer-data-commentaries.xlsx',
                            sheet_name='Table S4', skiprows=2, nrows=79)
Cancer_data = Cancer_data.drop(
    columns=['Age range (years)', 'Unnamed: 5', 'Unnamed: 8', 'ICD10 codes', 'Unnamed: 10']).rename(
    columns={'Background data:\nCumulative number of\n people diagnosed': 'Cumulative number of people diagnosed',
             'Background data:\nCumulative number of\n deaths': 'Cumulative number of deaths'})
Cancer_data.to_excel('Cleaned_data/AIHW_Cancer_Commentaries.xlsx', index=False)

kaggle.api.authenticate()
kaggle.api.dataset_download_files('bhanuprasanna/isic-2019', path='Project_Datasets/', unzip=True)
kaggle.api.dataset_download_files('andrewmvd/isic-2019', path='Project_Datasets/', unzip=True)
