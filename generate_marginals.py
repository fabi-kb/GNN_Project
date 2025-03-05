import pandas as pd
import numpy as np
import os
import glob
import json

"""
Create marginal results for all variables across all data points for a given census tract. Requires populating folder ACS_tract_tables with tables S0101, S0601, S1101, S1602, S1901, B08201, B11007, B11005.
"""


def load_and_clean_table(table_path):
    """Get rid of some annoyances from ACS Formatting"""
    table = pd.read_csv(table_path)

    # Get rid of white space indentations in label column
    table["Label (Grouping)"] = table["Label (Grouping)"].str.strip()

    # Convert second column to proper numbers; two cases: percents and counts.
    col_name = table.columns[1]

    # Remove encoding for non-numerical values (X)
    is_missing = table[col_name].str.contains(r"\(X\)", na=False)
    table.loc[is_missing, col_name] = np.nan

    # Remove thousands separator commas
    table[col_name] = table[col_name].str.replace(",", "")

    # Divide percentages by 100
    is_percent = table[col_name].str.endswith("%", na=False)
    table.loc[is_percent, col_name] = (
        table.loc[is_percent, col_name].str.rstrip("%").astype(float) / 100
    )

    # Convert rest to float
    table[col_name] = table[col_name].astype(float)

    return table


def generate_marginals():
    # Initialise final dictionary of marginals

    marginals = {}

    ############################################
    # Handle home-ownership (tenure): table S1101
    # Load and clean table
    table_path = glob.glob("ACS_tract_tables/*S1101*")[0]
    table = load_and_clean_table(table_path)

    own_perc = table.iloc[23, 1]

    marginals["TEN:owned or mortgaged"] = own_perc
    marginals["TEN:rented"] = 1 - own_perc

    ############################################
    # Handle language (HHL): table S1602
    table_path = glob.glob("ACS_tract_tables/*S1602*")[0]
    table = load_and_clean_table(table_path)

    n_household = table.iloc[0, 1]
    present_languages = [
        "Spanish",
        "Other Indo-European languages",
        "Asian and Pacific Island languages",
        "Other languages",
    ]
    language_counts = (
        table[table["Label (Grouping)"].isin(present_languages)].iloc[:, 1].tolist()
    )

    marginals["HHL:english"] = (n_household - sum(language_counts)) / n_household
    marginals["HHL:spanish"] = language_counts[0] / n_household
    marginals["HHL:other indo-european"] = language_counts[1] / n_household
    marginals["HHL:asian and pacific island languages"] = (
        language_counts[2] / n_household
    )
    marginals["HHL:other"] = language_counts[3] / n_household

    ############################################
    # Handle number of vehicles: table B08201
    table_path = glob.glob("ACS_tract_tables/*B08201*")[0]
    table = load_and_clean_table(table_path)

    vehicle_labels = [
        "No vehicle available",
        "1 vehicle available",
        "2 vehicles available",
        "3 vehicles available",
        "4 or more vehicles available",
    ]
    vehicle_counts = (
        table[table["Label (Grouping)"].isin(vehicle_labels)].iloc[:5, 1].tolist()
    )

    n_vehicle_households = sum(vehicle_counts)

    marginals["VEH:no vehicles"] = vehicle_counts[0] / n_vehicle_households
    marginals["VEH:1 vehicle"] = vehicle_counts[1] / n_vehicle_households
    marginals["VEH:2 vehicles"] = vehicle_counts[2] / n_vehicle_households
    marginals["VEH:3 vehicles"] = vehicle_counts[3] / n_vehicle_households
    marginals["VEH:4 or more vehicles"] = vehicle_counts[4] / n_vehicle_households

    ############################################
    # Handle income: table S1901
    table_path = glob.glob("ACS_tract_tables/*S1901*")[0]
    table = load_and_clean_table(table_path)

    incomes = table.iloc[1:11, 1].tolist()

    marginals["HINCP:under 10k"] = incomes[0]
    marginals["HINCP:10k-15k"] = incomes[1]
    marginals["HINCP:15k-25k"] = incomes[2]
    marginals["HINCP:25k-35k"] = incomes[3]
    marginals["HINCP:35k-50k"] = incomes[4]
    marginals["HINCP:50k-75k"] = incomes[5]
    marginals["HINCP:75k-100k"] = incomes[6]
    marginals["HINCP:100k-150k"] = incomes[7]
    marginals["HINCP:150k+"] = incomes[8] + incomes[9]

    ############################################
    # Handle over 65: table B11007
    table_path = glob.glob("ACS_tract_tables/*B11007*")[0]
    table = load_and_clean_table(table_path)

    n_household = table.iloc[0, 1]

    n_yes = table.iloc[1, 1]
    marginals["R65:no"] = (n_household - n_yes) / n_household
    marginals["R65:yes"] = n_yes / n_household

    ############################################
    # Handle under 18: table B11005
    table_path = glob.glob("ACS_tract_tables/*B11005*")[0]
    table = load_and_clean_table(table_path)

    n_household = table.iloc[0, 1]

    n_yes = table.iloc[1, 1]

    marginals["R18:no"] = (n_household - n_yes) / n_household
    marginals["R18:yes"] = n_yes / n_household

    ############################################
    # Handle age: table S0101
    table_path = glob.glob("ACS_tract_tables/*S0101*")[0]
    table = load_and_clean_table(table_path)

    table_ages = table.iloc[2:20, 1].tolist()
    age_population = sum(table_ages)

    age_vars = [
        "AGEP:under 5",
        "AGEP:5-9",
        "AGEP:10-14",
        "AGEP:15-19",
        "AGEP:20-24",
        "AGEP:25-29",
        "AGEP:30-34",
        "AGEP:35-39",
        "AGEP:40-44",
        "AGEP:45-49",
        "AGEP:50-54",
        "AGEP:55-59",
        "AGEP:60-64",
        "AGEP:65-69",
        "AGEP:70-74",
        "AGEP:75-79",
        "AGEP:80-84",
        "AGEP:85+",
    ]

    for i, age in enumerate(age_vars):
        marginals[age] = table_ages[i] / age_population

    ############################################
    # Handle sex: table s0601
    table_path = glob.glob("ACS_tract_tables/*S0601*")[0]
    table = load_and_clean_table(table_path)

    ## Sex:
    marginals["SEX:female"] = table.iloc[13, 1]
    marginals["SEX:male"] = table.iloc[12, 1]

    ## Education
    education_labels = [
        "SCHL:less than high school",
        "SCHL:high school",
        "SCHL:college or associate",
        "SCHL:bachelor",
        "SCHL:graduate",
    ]

    table_educations = table.iloc[38:43, 1].to_list()

    for i, label in enumerate(education_labels):
        marginals[label] = table_educations[i]

    ## Remove NaN entries

    with open("marginals.json", "w") as f:
        json.dump(marginals, f, indent=4)


if __name__ == "__main__":
    generate_marginals()
