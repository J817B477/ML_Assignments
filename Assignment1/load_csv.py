import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_csv(path, split="train"):

    data_path = os.path.join(path, split+".csv")

    # Load CSV file into a DataFrame
    df = pd.read_csv(data_path)

    # Display the first 5 rows
    print(df.head())
    print("--------"*20)

    # General info
    print("\n>>> Data info:")
    print(df.info())
    print("--------"*20)

    # Statistical summary
    print("\n>>> Summary statistics:")
    print(df.describe())
    print("--------"*20)

    # Check for missing values
    print("\n>>> Missing values in each column:")
    print(df.isnull().sum())
    print("--------"*20)

    # Switch DataFrame to numpy array
    df_numpy = df.values
    print(df_numpy.shape)
    print("--------"*20)

    # split entire data into features and labels
    feats = df_numpy[:,  0:-1]
    labels = df_numpy[:, -1]
    print(feats.shape)
    print(labels.shape)
    print("--------"*20)

    # reshape the label column
    labels = labels.reshape(-1, 1)
    print(labels.shape)
    print("--------"*20)

    # remove a column
    df = df.drop('clock_speed', axis=1) # same as df.drop('clock_speed', axis=1, inplace=True)
    print(df.head())
    print("--------"*20)

    # remove a row
    df = df.drop(df.index[1])
    print(df.head())
    print("--------"*20)

    # check the shape of ndarray
    df_numpy = df.values
    print(df_numpy.shape)
    print("--------"*20)

    # Value counts for categorical columns
    print("--------"*20)
    print("\n>>> battery_power distribution:")
    print(df['battery_power'].value_counts())
    print("--------"*20)

    # Histogram of Age
    plt.figure(figsize=(8, 4))
    sns.histplot(df['battery_power'], kde=True, bins=30)
    plt.title("Distribution of battery_power")
    plt.xlabel("battery_power")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


    # Bar plot of price_range by battery_power
    plt.figure(figsize=(6, 4))
    sns.countplot(x='battery_power', hue='price_range', data=df)
    plt.title("Survival by battery_power")
    plt.xlabel("battery_power")
    plt.ylabel("Count")
    plt.legend(title='price_range', labels=['0', '1', '2', '3'])
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    mobile_price_classification_dir = "directory that stores the data (.csv)"
    load_csv(mobile_price_classification_dir)