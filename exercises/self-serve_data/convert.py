import pandas as pd

def convert():
    data = pd.read_csv("data/gdp_pc/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_315923.csv", header=2)
    print(data.info())

if __name__ == "__main__":
    convert()