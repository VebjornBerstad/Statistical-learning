import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fuzzywuzzy import process
from tqdm import tqdm

from kaggle.api.kaggle_api_extended import KaggleApi


def download_data(path="data"):
    try:
        api = KaggleApi()
        api.authenticate()

        path_raw = os.path.join(path, "raw")
        os.makedirs(path_raw, exist_ok=True)

        # Add try-except blocks for each dataset download
        try:
            api.dataset_download_files(
                "vivovinco/20222023-football-player-stats",
                path=path_raw,
                unzip=True,
            )
        except Exception as e:
            print(f"Error downloading dataset '20222023-football-player-stats': {e}")

        try:
            api.dataset_download_files(
                "meraxes10/fantasy-premier-league-dataset-2022-2023",
                path=path_raw,
                unzip=True,
            )
        except Exception as e:
            print(
                f"Error downloading dataset 'fantasy-premier-league-dataset-2022-2023': {e}"
            )

    except Exception as e:
        print(f"An error occurred in the download_data function: {e}")


def merge_dataframes(df1, df2, name_column="name"):
    """
    Merge two dataframes

    Args:
        df1: the first dataframe
        df2: the second dataframe

    Returns:
        pd.DataFrame: the merged dataframe
    """

    # Merge the two dataframes
    df = pd.merge(df1, df2, on=name_column, how="left")

    return df


def get_best_match(name, choices, threshold=60):
    """
    Get the best match from a list of choices

    Args:
        name: the name to match
        choices: the list of choices
        threshold: the threshold for the match

    Returns:
        str: the best match
    """

    match = process.extractOne(name, choices, score_cutoff=threshold)

    return match[0] if match else np.nan


def match_and_merge_dataframes(df1, df2, name_column="name"):
    """
    Match the names in two dataframes

    Args:
        df1: the first dataframe
        df2: the second dataframe
        name_column_df1: the name of the column in the first dataframe
        name_column_df2: the name of the column in the second dataframe

    Returns:
        pd.DataFrame: dataframe with matched names
    """

    tqdm.pandas(desc="Matching names")

    df1[name_column] = df1[name_column].progress_apply(
        lambda x: get_best_match(x, df2[name_column])
    )

    df1 = df1.dropna(subset=[name_column])

    df1 = merge_dataframes(df1, df2, name_column=name_column)

    return df1


def save_data(df, file_path="data/processed/data.csv"):
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")


def preprocess_data(path="data"):
    try:
        path_raw = os.path.join(path, "raw")
        path_processed = os.path.join(path, "processed")
        os.makedirs(path_processed, exist_ok=True)

        try:
            df = pd.read_csv(
                os.path.join(path_raw, "2022-2023 Football Player Stats.csv"),
                delimiter=";",
                encoding="ISO-8859-1",
            )
        except FileNotFoundError:
            print(
                "Error: The file '2022-2023 Football Player Stats.csv' was not found."
            )
            return
        except Exception as e:
            print(
                f"An error occurred while reading '2022-2023 Football Player Stats.csv': {e}"
            )
            return

        try:
            df_points = pd.read_csv(os.path.join(path_raw, "players.csv"))
        except FileNotFoundError:
            print("Error: The file 'players.csv' was not found.")
            return
        except Exception as e:
            print(f"An error occurred while reading 'players.csv': {e}")
            return
    except Exception as e:
        print(f"An error occurred in the preprocess_data function: {e}")
        return

    # Keep only premier league players and remove players with less than 90 minutes played
    df = df[df["Comp"] == "Premier League"]
    df = df[df["Min"] > 90]

    # Drop columns that is directly related to the target
    df = df.drop(
        columns=[
            "Goals",
            "G/Sh",
            "G/SoT",
            "ShoPK",
            "Assists",
            "GcaPassLive",
            "GcaPassDead",
            "GcaDrib",
            "GcaSh",
            "GcaFld",
            "GcaDef",
            "CrdY",
            "CrdR",
            "2CrdY",
            "OG",
            "PKwon",
            "PKcon",
        ]
    )

    # Drop columns that are not needed
    df = df.drop(
        columns=[
            "Rk",
            "Nation",
            "Squad",
            "Comp",
            "Age",
            "Born",
            "MP",
            "Starts",
            "90s",
            "Tkl+Int",
            'Pos'
        ]
    )

    # Drop columns that are not needed in the points dataframe
    df_points = df_points[["name", "position", "total_points"]]

    # Rename the Player column in df to match points dataframe
    df.rename(columns={"Player": "name"}, inplace=True)

    # Match the names in the two dataframes using fuzzy matching
    df = match_and_merge_dataframes(df, df_points)

    # Drop duplicate rows with same name and number of points keeping the first instance
    df = df.drop_duplicates(subset=["name", "total_points"], keep="first")

    # Calculate quantiles for total points
    df["points_quantile"] = pd.qcut(
        df["total_points"], 4, labels=["low", "medium", "high", "very high"]
    )

    # Print number of rows and features
    print(f"Number of rows: {df.shape[0]} \nNumber of features: {df.shape[1]}")

    # Save the dataframe to a csv file
    save_data(df, os.path.join(path_processed, "processed_data.csv"))
    print("Data saved to data/processed/processed_data.csv")

    return


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the data directory",
    )
    parser.add_argument("--download_data", type=bool, default=True, help="Download data")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.download_data:
        download_data()
    preprocess_data(path=args.data_path)


if __name__ == "__main__":
    main()
