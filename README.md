# E-Commerce Customer Analytics with Python

This project includes customer segmentation and CLTV prediction for an e-commerce dataset using Python.

## 📁 Project Structure

- `rfm.py`: Performs RFM-based customer segmentation.
- `cltv_rule_based.py`: Calculates CLTV using a traditional formula.
- `cltv_prediction.py`: Predicts CLTV using BG-NBD and Gamma-Gamma models.

## 📊 Dataset Info

The dataset comes from the UCI Machine Learning Repository:
> [Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II)

It includes transactions from a UK-based online retail store between 2009 and 2011.

You can find the file in this repository as:  
`online retail 2. copy.xlsx`

### Columns:
- `InvoiceNo`: Unique invoice number (if starts with 'C', it was cancelled)
- `StockCode`: Product code
- `Description`: Product name
- `Quantity`: Number of products sold
- `InvoiceDate`: Invoice date
- `UnitPrice`: Price per unit
- `CustomerID`: Unique customer number
- `Country`: Customer's country

## ⚙️ How to Run

1. Clone this repository  
2. Install requirements:  
<pre><code>   ```
   pip install pandas lifetimes matplotlib scikit-learn
   ```</code></pre>


## 🧠 What I Learned

While working on this project, I had the chance to apply everything I learned in the CRM module of the bootcamp.  
I especially understood how important customer segmentation is in real business problems.

Here are a few key things I gained from this experience:
- How to group customers based on Recency, Frequency, and Monetary value
- The logic behind rule-based vs. predictive CLTV calculation
- How to implement probabilistic models like BG-NBD and Gamma-Gamma
- Cleaning messy real-world datasets
- Writing clean and reusable Python code

## 📚 Background

This project was developed during the [Miuul Data Science Bootcamp](https://miuul.com) as part of the CRM analytics module.

