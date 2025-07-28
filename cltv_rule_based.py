###############################################################
# CLTV RULE-BASED (Customer Lifetime Value Calculation)
# This script calculates customer lifetime value using a rule-based approach:
# CLTV = (customer_value / churn_rate) x profit_margin
###############################################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("/Users/betulcoklu/PycharmProjects/pythonProject/datasets/online retail 2. copy.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.isnull().sum()  # Check for missing values
df = df[~df["Invoice"].str.contains("C", na=False)]  # Remove canceled transactions (Invoice contains 'C')
df.describe().T
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),  # Count unique invoices per customer
                                        'Quantity': lambda x: x.sum(),     # Total units purchased per customer
                                        'TotalPrice': lambda x: x.sum()})  # Total revenue per customer

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

##################################################
# 2. Average Order Value (total_price / total_transaction)
###################################################

cltv_c.head()
cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c.head()
cltv_c.shape[0]  # Number of unique customers
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (customers with more than one purchase / total customers)
##################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate  # Proportion of customers likely to churn

##################################################
# 5. Profit Margin (total_price * 0.10)
###################################################

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

##################################################
# 6. Customer Value (average_order_value * purchase_frequency)
###################################################

cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c["purchase_frequency"]

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv", ascending=False).head()  # Top customers by CLTV

##################################################
# 8. Segmentation based on CLTV
###################################################

cltv_c.sort_values(by="cltv", ascending=False).tail()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])  # Segment customers into quartiles

cltv_c.sort_values(by="cltv", ascending=False).head()

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltc_c.csv")

##################################################
# 9. BONUS: Function to calculate CLTV
###################################################

def create_cltv_c(dataframe, profit=0.10):

    # Data preprocessing
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["UnitPrice"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    # Average order value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
    # Purchase frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
    # Repeat rate and churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # Profit margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    # Customer value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])
    # Customer lifetime value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']
    # Segmentation
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df = df_.copy()

clv = create_cltv_c(df)

print("Top customers by CLTV:")
print(clv.sort_values(by="cltv", ascending=False).head())

