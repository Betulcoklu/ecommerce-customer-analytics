###############################################################
# Customer Segmentation with RFM (Recency, Frequency, Monetary)
###############################################################

# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analysing RFM Segments
# 7. Functionizing the Entire Process

###############################################################
# 1. Business Problem
###############################################################

# An e-commerce company wants to segment its customers and determine marketing strategies based on these segments.

# Dataset Story
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The Online Retail II dataset contains sales data from an online store based in the UK
# between 01/12/2009 and 09/12/2011.

# Variables
#
# InvoiceNo: Invoice number. Unique number for each transaction. If it starts with C, it means the transaction was canceled.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Quantity of the product. Shows how many units of the products were sold in the invoices.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in pounds)
# CustomerID: Unique customer number
# Country: Country name. The country where the customer lives.


###############################################################
# 2. Data Understanding
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)   # Show all columns
# pd.set_option('display.max_rows', None)  # Show all rows (not preferred due to too much output)
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Show 3 decimal places for numerical values

df_ = pd.read_excel("/Users/betulcoklu/PycharmProjects/pythonProject/datasets/online retail 2. copy.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape  # number of rows and columns
df.isnull().sum()  # check for missing values

# How many unique products are there?
df["Description"].nunique() # result: 4681 unique products

df["Description"].value_counts().head()  # how many times each product appeared in invoices
# This shows how many times unique products appeared in invoices
# For example, the first product appeared 5349 times in invoices, but we don't know how many were sold

df.groupby("Description").agg({"Quantity": "sum"}).head()  # which product was ordered the most (group by description and sum quantity)

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()  # sorted by quantity descending
# This shows the total quantity ordered for each product.

df["Invoice"].nunique()
# total number of unique invoices

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]  # total revenue per product

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()  # total amount spent per invoice

###############################################################
# 3. Data Preparation
###############################################################

df.shape
df.isnull().sum()
df.describe().T
df = df[(df['Quantity'] > 0)]  # show only quantities greater than zero
df.dropna(inplace=True)    # drop missing values
df = df[~df["Invoice"].astype(str).str.contains("C", na=False)]   # remove canceled invoices from dataset


###############################################################
# 4. Calculating RFM Metrics
###############################################################

# Recency (analysis date - last purchase date), Frequency (number of purchases), Monetary (total amount spent)
df.head()

df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)   # analysis date
type(today_date)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]   # remove customers with zero monetary value
rfm.shape


###############################################################
# 5. Calculating RFM Scores
###############################################################

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])  # quantile-based scoring for recency
# qcut divides variable into quantiles with given labels
# qcut sorts from smallest to largest

# 0-100, 0-20, 20-40, 40-60, 60-80, 80-100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
# rank method used to avoid ties and assign ranks
# method "first" assigns ranks in order of appearance

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

rfm[rfm["RFM_SCORE"] == "55"]

rfm[rfm["RFM_SCORE"] == "11"]

###############################################################
# 6. Creating & Analysing RFM Segments
###############################################################
# regex

# RFM naming
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
# add a new variable 'segment' to dataset by replacing RFM_SCORE values according to map

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
# group by segment and calculate mean and count for recency, frequency, monetary


# They might ask for the 'cant loose' class.
rfm[rfm["segment"] == "cant_loose"].head()
rfm[rfm["segment"] == "cant_loose"].index   # access customer IDs

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)  # convert to integer to remove decimals

new_df.to_csv("new_customers.csv")  # export as csv to share with department

rfm.to_csv("rfm.csv")  # export full segment information as csv

###############################################################
# 7. Functionizing the Entire Process
###############################################################

def create_rfm(dataframe, csv=False):   # define a function named create_rfm

    # DATA PREPARATION
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["UnitPrice"]  # calculate total price
    dataframe.dropna(inplace=True)  # drop missing values
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]  # remove canceled invoices

    # CALCULATING RFM METRICS
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # CALCULATING RFM SCORES
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # convert scores to categorical and add to dataframe
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))


    # NAMING SEGMENTS
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)   # name the segments
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]  # keep only relevant columns
    rfm.index = rfm.index.astype(int)   # convert customer IDs to integer

    if csv:   # if csv argument is True
        rfm.to_csv("rfm.csv")  # export rfm as csv

    return rfm

df = df_.copy()   # original dataset copy

rfm_new = create_rfm(df, csv=True)  # now rfm csv file is generated here
