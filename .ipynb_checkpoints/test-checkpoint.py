import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# 1. Data Wrangling
# Membaca data
customer_df = pd.read_csv('Dataset/customers_dataset.csv')
geolocation_df = pd.read_csv('Dataset/geolocation_dataset.csv')
order_items_df = pd.read_csv('Dataset/order_items_dataset.csv')
order_payments_df = pd.read_csv('Dataset/order_payments_dataset.csv')
order_reviews_df = pd.read_csv('Dataset/order_reviews_dataset.csv')
orders_df = pd.read_csv('Dataset/orders_dataset.csv')
product_category_name_translation_df = pd.read_csv('Dataset/product_category_name_translation.csv')
products_df = pd.read_csv('Dataset/products_dataset.csv')
sellers_df = pd.read_csv('Dataset/sellers_dataset.csv')

# Memeriksa nilai yang hilang
print(customer_df.isnull().sum())
print(geolocation_df.isnull().sum())
print(order_items_df.isnull().sum())
print(order_payments_df.isnull().sum())
print(order_reviews_df.isnull().sum())
print(orders_df.isnull().sum())
print(product_category_name_translation_df.isnull().sum())
print(products_df.isnull().sum())
print(sellers_df.isnull().sum())

# Memeriksa duplikasi
print(geolocation_df.duplicated().sum())

# Menghapus nilai yang hilang pada data yang relevan
orders_df = orders_df.dropna(subset=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
products_df = products_df.dropna()
order_reviews_df = order_reviews_df.dropna(subset=['review_comment_title', 'review_comment_message'])

# Menghapus duplikasi pada geolocation_df
geolocation_df.drop_duplicates(inplace=True)

# 2. Exploratory Data Analysis (EDA)
# Deskripsi statistik dari setiap dataset
print(customer_df.describe(include='all'))
print(geolocation_df.describe(include='all'))
print(order_items_df.describe(include='all'))
print(order_payments_df.describe(include='all'))
print(order_reviews_df.describe(include='all'))
print(orders_df.describe(include='all'))
print(product_category_name_translation_df.describe(include='all'))
print(products_df.describe(include='all'))
print(sellers_df.describe(include='all'))

# 3. Explanatory Data Analysis
# Pertanyaan 1: Produk mana yang paling banyak terjual dan bagaimana pengaruhnya terhadap keuntungan?

# Gabungkan data order_items dengan products berdasarkan product_id
order_items_products = pd.merge(order_items_df, products_df, on='product_id')

# Hitung jumlah penjualan per produk
product_sales = order_items_products.groupby('product_id')['order_item_id'].count().reset_index()
product_sales.columns = ['product_id', 'total_sales']
product_sales = product_sales.sort_values(by='total_sales', ascending=False)

# Hitung total keuntungan per produk
product_profit = order_items_products.groupby('product_id')['price'].sum().reset_index()
product_profit.columns = ['product_id', 'total_profit']
product_profit = product_profit.sort_values(by='total_profit', ascending=False)

# Gabungkan data penjualan dan keuntungan
product_sales_profit = pd.merge(product_sales, product_profit, on='product_id')
product_sales_profit = pd.merge(product_sales_profit, products_df[['product_id', 'product_category_name']], on='product_id')

# Pertanyaan 2: Berapa rata-rata pembelanjaan pelanggan dan bagaimana variasinya berdasarkan lokasi geografis?

# Gabungkan data order_items dengan orders berdasarkan order_id
order_items_orders = pd.merge(order_items_df, orders_df, on='order_id')

# Gabungkan data dengan customers berdasarkan customer_id
order_items_orders_customers = pd.merge(order_items_orders, customer_df, on='customer_id')

# Hitung total pembelanjaan per pelanggan
customer_spending = order_items_orders_customers.groupby('customer_unique_id')['price'].sum().reset_index()
customer_spending.columns = ['customer_unique_id', 'total_spending']

# Gabungkan data pembelanjaan dengan data customer untuk mendapatkan lokasi
customer_spending_location = pd.merge(customer_spending, customer_df[['customer_unique_id', 'customer_city', 'customer_state']], on='customer_unique_id')

# Hitung rata-rata pembelanjaan per kota
city_spending = customer_spending_location.groupby('customer_city')['total_spending'].mean().reset_index()
city_spending.columns = ['customer_city', 'average_spending']
city_spending = city_spending.sort_values(by='average_spending', ascending=False)

# Pertanyaan 3: Lokasi geografis manakah yang memiliki jumlah pelanggan terbanyak?

# Hitung jumlah pelanggan per kota
customer_count_per_city = customer_df['customer_city'].value_counts().reset_index()
customer_count_per_city.columns = ['customer_city', 'customer_count']
customer_count_per_city = customer_count_per_city.sort_values(by='customer_count', ascending=False)

# 4. RFM Analysis

# Merge datasets
order_items_orders = pd.merge(order_items_df, orders_df, on='order_id')
order_items_orders_customers = pd.merge(order_items_orders, customer_df, on='customer_id')

# Hitung Recency, Frequency, dan Monetary
current_date = dt.datetime(2018, 8, 29)  # Asumsi ini adalah tanggal terakhir dalam dataset
order_items_orders_customers['order_purchase_timestamp'] = pd.to_datetime(order_items_orders_customers['order_purchase_timestamp'])
rfm_table = order_items_orders_customers.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (current_date - x.max()).days,
    'order_id': 'count',
    'price': 'sum'
}).reset_index()

rfm_table.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

# Skor RFM
rfm_table['R_Score'] = pd.qcut(rfm_table['Recency'], 4, ['4','3','2','1'])
rfm_table['F_Score'] = pd.qcut(rfm_table['Frequency'], 4, ['1','2','3','4'])
rfm_table['M_Score'] = pd.qcut(rfm_table['Monetary'], 4, ['1','2','3','4'])

# Menggabungkan skor menjadi satu nilai
rfm_table['RFM_Score'] = rfm_table['R_Score'].astype(str) + rfm_table['F_Score'].astype(str) + rfm_table['M_Score'].astype(str)

# Segmentasi Berdasarkan RFM Score
def rfm_segment(rfm):
    if rfm in ['444', '344', '434', '443']:
        return 'Best Customers'
    elif rfm in ['333', '433', '343', '334', '324']:
        return 'Loyal Customers'
    elif rfm in ['111', '211', '121', '112']:
        return 'Lost Customers'
    elif rfm in ['122', '132', '213', '223']:
        return 'Potential Customers'
    else:
        return 'Others'

rfm_table['Segment'] = rfm_table['RFM_Score'].apply(rfm_segment)

# 5. Data Visualization

# Visualisasi Segmen Pelanggan
plt.figure(figsize=(12, 8))
sns.countplot(x='Segment', data=rfm_table, order=rfm_table['Segment'].value_counts().index)
plt.title('Distribusi Segmen Pelanggan Berdasarkan RFM Analysis')
plt.xlabel('Segment')
plt.ylabel('Jumlah Pelanggan')
plt.xticks(rotation=45)
plt.show()

# Menampilkan tabel RFM
print(rfm_table.head())

# Visualisasi Lainnya
# Visualisasi produk terlaris dengan nama produk
top_products = pd.merge(product_sales_profit.head(10), products_df[['product_id', 'product_category_name']], on='product_id')
plt.figure(figsize=(10, 6))
sns.barplot(x='product_category_name', y='total_sales', data=top_products)
plt.title('Top 10 Produk Terlaris')
plt.xlabel('Product Name')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Visualisasi rata-rata pembelanjaan per kota
plt.figure(figsize=(10, 6))
sns.barplot(x='customer_city', y='average_spending', data=city_spending.head(10))
plt.title('Top 10 Kota dengan Rata-rata Pembelanjaan Tertinggi')
plt.xlabel('Customer City')
plt.ylabel('Average Spending')
plt.xticks(rotation=45)
plt.show()

# Visualisasi jumlah pelanggan per kota
plt.figure(figsize=(10, 6))
sns.barplot(x='customer_city', y='customer_count', data=customer_count_per_city.head(10))
plt.title('Top 10 Kota dengan Jumlah Pelanggan Terbanyak')
plt.xlabel('Customer City')
plt.ylabel('Customer Count')
plt.xticks(rotation=45)
plt.show()
