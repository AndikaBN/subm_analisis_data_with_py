import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import streamlit as st
import urllib
import numpy as np
import scipy.stats as stats
from function import DataAnalyzer, BrazilMapPlotter

sns.set(style='dark')

# Dataset
datetime_cols = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date", "order_purchase_timestamp", "shipping_limit_date"]
all_df = pd.read_csv("https://raw.githubusercontent.com/AndikaBN/subm_analisis_data_with_py/refs/heads/main/Dashboard/olist_ecommerce_data.csv")
all_df.sort_values(by="order_approved_at", inplace=True)
all_df.reset_index(inplace=True)

# Geolocation Dataset
geolocation = pd.read_csv('https://raw.githubusercontent.com/AndikaBN/subm_analisis_data_with_py/refs/heads/main/Dashboard/olist_ecommerce_data_silver.csv')
data = geolocation.drop_duplicates(subset='customer_unique_id')

for col in datetime_cols:
    all_df[col] = pd.to_datetime(all_df[col])

min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

# Sidebar
with st.sidebar:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image("https://raw.githubusercontent.com/AndikaBN/subm_analisis_data_with_py/refs/heads/main/Dashboard/WTCf.png", width=100)
    with col3:
        st.write(' ')

    # Date Range
    start_date, end_date = st.date_input(
        label="Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Main
main_df = all_df[(all_df["order_approved_at"] >= str(start_date)) & 
                 (all_df["order_approved_at"] <= str(end_date))]

function = DataAnalyzer(main_df)
map_plot = BrazilMapPlotter(data, plt, mpimg, urllib, st)

daily_orders_df = function.create_daily_orders_df()
sum_spend_df = function.create_sum_spend_df()
sum_order_items_df = function.create_sum_order_items_df()
review_score, common_score = function.review_score_df()
state, most_common_state = function.create_bystate_df()
order_status, common_status = function.create_order_status()

# Define your Streamlit app
st.title("E-Commerce Public Data Analysis")

# Add text or descriptions
st.write("**This is a dashboard for analyzing E-Commerce public data.**")

st.subheader("Product Price vs. Sell Probability")

# Menggunakan all_df untuk menggabungkan items dan products
items_product = all_df[['order_id', 'product_id', 'price', 'order_item_id']].copy()
orders_ip = all_df[['order_id']].merge(items_product, on='order_id', how='inner')

# pivot table aggregating by # of items bought and mean of price
product_revenue = orders_ip.pivot_table(index=['product_id'], aggfunc={'order_item_id': 'sum', 'price': 'mean'})
product_revenue['total'] = product_revenue['order_item_id'] * product_revenue['price']
product_revenue.rename(columns={'order_item_id': 'sell_probability'}, inplace=True)
product_revenue['sell_probability'] = product_revenue['sell_probability'] / len(product_revenue)
product_revenue.sort_values(by='total', ascending=False)

x = np.log(product_revenue.sell_probability)
y = np.log(product_revenue.price)

fig, ax = plt.subplots(figsize=(8, 6))
plt.title('Product Price vs. Sell Probability', fontsize=16)
plt.xlabel('Log Sell Probability', fontsize=12)
plt.ylabel('Log Product Price', fontsize=12)

plt.xlim(-11, -3)
plt.ylim(0, 9)

plt.yticks(range(10), [int(np.exp(x)) for x in range(10)], fontsize=10)
plt.xticks(range(-10, -2), [round(np.exp(x), 4) for x in range(-10, -2)], fontsize=10, rotation=30)

hb = ax.hexbin(x, y, gridsize=14, C=product_revenue.total, reduce_C_function=np.sum, cmap='cividis')

# Added colorbar
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Product Revenue (R$)', rotation=270, labelpad=20, fontsize=12)

plt.tight_layout()
st.pyplot(fig)

# Define orders, payments, and customers from all_df
orders = all_df[['order_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 'customer_id']].copy()
payments = all_df[['order_id', 'payment_value']].copy()
customers = all_df[['customer_id', 'customer_unique_id', 'customer_state']].copy()

# Visualisasi Data Kedua: Mean Transaction by State (95% CI)
st.subheader("Mean Transaction by State (95% CI)")

# Menggabungkan data orders, payments, dan customers
pay_ord_cust = orders.merge(payments, on='order_id', how='outer').merge(customers, on='customer_id', how='outer')
customer_spent = pay_ord_cust.groupby('customer_unique_id').agg({'payment_value': 'sum'}).sort_values(by='payment_value', ascending=False)

# Menghitung rata-rata pengeluaran pelanggan dan standar deviasi
customer_mean = customer_spent['payment_value'].mean()
customer_std = stats.sem(customer_spent['payment_value'])

# Menghitung confidence interval
confidence_interval = stats.t.interval(0.95, loc=customer_mean, scale=customer_std, df=len(customer_spent) - 1)

# Menghitung rata-rata pengeluaran dan CI untuk setiap wilayah
customer_regions = pay_ord_cust.groupby('customer_state').agg({
    'payment_value': ['mean', 'std'], 
    'customer_unique_id': 'count'
})

customer_regions.columns = ['mean_payment_value', 'std_payment_value', 'count_customers']
customer_regions.reset_index(inplace=True)

ci_low = []
ci_hi = []

for index, row in customer_regions.iterrows():
    mean = row['mean_payment_value']
    std = row['std_payment_value']
    count = row['count_customers']
    
    if count > 1:
        ci = stats.t.interval(0.95, loc=mean, scale=std / np.sqrt(count), df=count - 1)
        ci_low.append(ci[0])
        ci_hi.append(ci[1])
    else:
        ci_low.append(np.nan)
        ci_hi.append(np.nan)

customer_regions['ci_low'] = ci_low
customer_regions['ci_hi'] = ci_hi

# Plot Mean Transaction by State
fig, ax = plt.subplots(figsize=(12, 4))
plot = customer_regions.sort_values(by='mean_payment_value')

plt.xticks(rotation=30)
plt.xlabel('State')
plt.ylabel('Mean Transaction (95% CI)')
plt.xlim(-0.5, len(plot) - 0.5)
plt.ylim(125, 325)
plt.scatter(plot['customer_state'], plot['mean_payment_value'], s=100, c=plot['mean_payment_value'])
plt.vlines(plot['customer_state'], plot['ci_low'], plot['ci_hi'], lw=.5)

plt.tight_layout()
st.pyplot(fig)

# Customer Demographic
st.subheader("Customer Demographic")
tab1, tab2 = st.tabs(["State", "Geolocation"])

with tab1:
    most_common_state = state.customer_state.value_counts().index[0]
    st.markdown(f"Most Common State: **{most_common_state}**")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=state.customer_state.value_counts().index, y=state.customer_count.values, palette="viridis", ax=ax)

    plt.title("Number of customers from State", fontsize=15)
    plt.xlabel("State")
    plt.ylabel("Number of Customers")

    st.pyplot(fig)

with tab2:
    map_plot.plot()

    with st.expander("See Explanation"):
        st.write('Menurut grafik yang telah dibuat, terdapat lebih banyak pelanggan di wilayah tenggara dan selatan. Selain itu, sebagian besar pelanggan berada di kota-kota yang merupakan ibu kota, seperti São Paulo, Rio de Janeiro, Porto Alegre, dan lain-lain.')

st.caption('Copyright (C) Andika Bintang Nursalih 2024')
