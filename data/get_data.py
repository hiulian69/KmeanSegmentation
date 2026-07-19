"""Re-download the three customer datasets used in this project.

Run from the repo root:  python data/get_data.py

Sources
-------
1. Mall Customers  — the classic 200-customer mall dataset
   (mirror of the Machine Learning A-Z course file).
2. Wholesale Customers — UCI ML Repository, annual spend of 440
   wholesale clients of a Portuguese distributor.
3. Online Retail — UCI ML Repository, 541,909 transactions of a UK
   online gift retailer (Dec 2010 – Dec 2011). Downloaded as .xlsx and
   converted to a compressed CSV so the notebook loads in seconds.
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

MALL_URL = (
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"
)
WHOLESALE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/"
    "Wholesale%20customers%20data.csv"
)
RETAIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/"
    "Online%20Retail.xlsx"
)


def fetch(url, filename):
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        print(f"already here: {filename}")
        return path
    print(f"downloading {filename} ...")
    urllib.request.urlretrieve(url, path)
    print(f"done: {filename}")
    return path


def main():
    fetch(MALL_URL, "Mall_Customers.csv")
    fetch(WHOLESALE_URL, "wholesale_customers.csv")

    csv_gz = os.path.join(HERE, "online_retail.csv.gz")
    if os.path.exists(csv_gz):
        print("already here: online_retail.csv.gz")
    else:
        xlsx = fetch(RETAIL_URL, "Online_Retail.xlsx")
        import pandas as pd  # imported late: only needed for the conversion
        print("converting xlsx -> csv.gz (takes a minute) ...")
        pd.read_excel(xlsx, engine="openpyxl").to_csv(
            csv_gz, index=False, compression="gzip"
        )
        print("done: online_retail.csv.gz")


if __name__ == "__main__":
    main()
