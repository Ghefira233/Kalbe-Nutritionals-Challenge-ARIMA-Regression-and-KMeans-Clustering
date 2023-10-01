#!/usr/bin/env python
# coding: utf-8

# **This Notebook is written by: Ghefira Nur Fatimah Widyasari**

# # IMPORT DATA

# In[2]:


# Import File
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

transaction = pd.read_csv('Case Study - Transaction.csv', sep=';')
customer = pd.read_csv('Case Study - Customer.csv', sep=';')
product = pd.read_csv('Case Study - Product.csv', sep=';')
store = pd.read_csv('Case Study - Store.csv', sep=';')


# # PREPROCESSING & EDA - TRANSACTION

# In[3]:


transaction.head()


# In[4]:


transaction.shape


# In[5]:


# Informasi data
transaction.info()


# Kita perlu untuk mengubah CustomerID, ProductID, dan StoreID menjadi string, serta Date menjadi format datetime

# In[6]:


# Mengubah tipe data kolom CustomerID, ProductID, dan StoreID menjadi string
transaction['CustomerID'] = transaction['CustomerID'].astype(str)
transaction['ProductID'] = transaction['ProductID'].astype(str)
transaction['StoreID'] = transaction['StoreID'].astype(str)

# Mengubah tipe data kolom Date menjadi format datetime
transaction['Date'] = pd.to_datetime(transaction['Date'], format='%d/%m/%Y')

# Menampilkan informasi tipe data setelah perubahan
transaction.info()


# In[7]:


# Statistik deskriptif
transaction.describe()


# In[8]:


# Menampilkan jumlah unik dari setiap kolom string
categorical_columns = ['TransactionID', 'CustomerID', 'ProductID', 'StoreID']
for col in categorical_columns:
    unique_values = transaction[col].nunique()
    print(f'Jumlah unik {col}: {unique_values}')


# In[9]:


# Histogram untuk kolom Price
plt.figure(figsize=(10, 6))
sns.histplot(transaction['Price'], bins=30, kde=True)
plt.xlabel('Harga')
plt.ylabel('Frekuensi')
plt.title('Distribusi Harga')
plt.show()


# In[10]:


# Box plot untuk Total Amount
plt.figure(figsize=(10, 6))
sns.boxplot(transaction['TotalAmount'])
plt.xlabel('Total Amount')
plt.title('Box Plot Total Amount')
plt.show()


# In[11]:


# Visualisasi jumlah transaksi per tanggal
daily_transactions = transaction.groupby('Date').size()
plt.figure(figsize=(12, 6))
daily_transactions.plot(kind='line')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Transaksi')
plt.title('Jumlah Transaksi Harian')
plt.show()


# In[12]:


# Visualisasi jumlah transaksi per store
store_transactions = transaction.groupby('StoreID').size()
plt.figure(figsize=(10, 6))
ax = store_transactions.plot(kind='bar')
plt.xlabel('StoreID')
plt.ylabel('Jumlah Transaksi')
plt.title('Jumlah Transaksi per Toko berdasarkan StoreID')

# Temukan indeks batang dengan jumlah transaksi terbanyak
max_transaction_index = store_transactions.idxmax()

# Temukan posisi indeks di dalam DataFrame
pos_max = store_transactions.index.get_loc(max_transaction_index)

# Temukan indeks batang dengan jumlah transaksi terendah
min_transaction_index = store_transactions.idxmin()

# Temukan posisi indeks di dalam DataFrame
pos_min = store_transactions.index.get_loc(min_transaction_index)

# Tambahkan penanda pada batang terbanyak
max_transaction_value = store_transactions[max_transaction_index]
ax.annotate(f'Max: {max_transaction_value}', (pos_max, max_transaction_value), xytext=(20, 10),
            textcoords='offset points', arrowprops=dict(arrowstyle='->', color='red'))

# Tambahkan penanda pada batang terendah
min_transaction_value = store_transactions[min_transaction_index]
ax.annotate(f'Min: {min_transaction_value}', (pos_min, min_transaction_value), xytext=(20, -30),
            textcoords='offset points', arrowprops=dict(arrowstyle='->', color='blue'))


# In[13]:


# Korelasi antara kolom numerik
correlation_matrix = transaction.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.show()


# In[14]:


Q1 = transaction['TotalAmount'].quantile(0.25)
Q3 = transaction['TotalAmount'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = transaction[(transaction['TotalAmount'] < lower_bound) | (transaction['TotalAmount'] > upper_bound)]

print("Outlier:")
print(outliers)


# In[15]:


Q1_qty = transaction['Qty'].quantile(0.25)
Q3_qty = transaction['Qty'].quantile(0.75)

IQR_qty = Q3_qty - Q1_qty

lower_bound_qty = Q1_qty - 1.5 * IQR_qty
upper_bound_qty = Q3_qty + 1.5 * IQR_qty

outliers_qty = transaction[(transaction['Qty'] < lower_bound_qty) | (transaction['Qty'] > upper_bound_qty)]

plt.figure(figsize=(8, 6))
plt.boxplot(transaction['Qty'], vert=False)
plt.title('Box Plot Qty')
plt.show()

print("Outlier Qty:")
print(outliers_qty)


# In[16]:


# Mengecek duplikat pada kolom 'TransactionID'
duplicate_customers = transaction['TransactionID'].duplicated()

# Menampilkan hasil
print("Terdapat duplikat TransactionID:", duplicate_customers.any())


# In[17]:


# Mengecek duplikat pada kolom 'TransactionID'
duplicate_transaction = transaction[transaction['TransactionID'].duplicated(keep=False)]

# Menampilkan data yang memiliki 'TransactionID' yang sama
duplicate_transaction


# In[18]:


duplicate_transaction[duplicate_transaction['TransactionID'] == 'TR54287']


# Didapat ada 223 transaction id yang berulang, padahal seharusnya unik. Oleh karena itu perlu kita drop dan memilih transaction id yang terbaru berdasarkan tanggal transaksinya.

# In[19]:


# Urutkan tabel berdasarkan 'Date' secara descending
transaction = transaction.sort_values(by='Date', ascending=False)

# Hapus duplikat 'TransactionID' dengan hanya menyimpan yang paling baru
transaction = transaction.drop_duplicates(subset='TransactionID', keep='first')

# Mengecek duplikat pada kolom 'TransactionID'
duplicate_transaction = transaction[transaction['TransactionID'].duplicated(keep=False)]

# Menampilkan data yang memiliki 'TransactionID' yang sama
duplicate_transaction


# Data duplicate sudah berhasil di drop, sehingga sudah tidak ada data yang duplicate pada kolom TransactionID

# In[20]:


transaction.info()


# Setelah duplicate di drop, jumlah row sebanyak 4908 dari yang sebelumnya 5020

# # PREPROCESSING & EDA DATA - CUSTOMER

# In[21]:


customer.head()


# In[22]:


customer.shape


# In[23]:


customer.info()


# Kita perlu untuk mengubah Gender dan CustomerID menjadi string, income menjadi int64. Namun, income perlu diubah terlebih dahulu tanda koma menjadi tanda titik

# In[24]:


# Mengganti koma (,) dengan titik (.) pada kolom 'Income'
customer['Income'] = customer['Income'].str.replace(',', '.')

# Mengubah tipe data kolom 'Income' menjadi float
customer['Income'] = customer['Income'].astype(float)


# In[25]:


# Mengubah tipe data kolom CustomerID dan Gender menjadi string
customer['Gender'] = customer['Gender'].astype(str)
customer['CustomerID'] = customer['CustomerID'].astype(str)

# Menampilkan informasi tipe data setelah perubahan
customer.info()


# In[26]:


# Statistik deskriptif
customer.describe()


# In[27]:


# Menampilkan jumlah unik dari setiap kolom string
string_columns = ['CustomerID', 'Gender', 'Marital Status']
for col in string_columns:
    unique_values = customer[col].nunique()
    print(f'Jumlah unik {col}: {unique_values}')


# In[28]:


# Distribusi Umur (Age)
plt.figure(figsize=(10, 6))
sns.histplot(data=customer, x='Age', bins=20, kde=True)
plt.title('Distribusi Umur (Age)')
plt.xlabel('Umur (Age)')
plt.ylabel('Frekuensi')
plt.show()


# In[29]:


# Distribusi Gender
plt.figure(figsize=(6, 4))
sns.countplot(data=customer, x='Gender')
plt.title('Distribusi Gender')
plt.xlabel('Gender')
plt.ylabel('Jumlah')
plt.show()


# In[30]:


# Distribusi Status Pernikahan (Marital Status)
plt.figure(figsize=(8, 4))
sns.countplot(data=customer, x='Marital Status')
plt.title('Distribusi Status Pernikahan (Marital Status)')
plt.xlabel('Status Pernikahan')
plt.ylabel('Jumlah')
plt.show()


# In[31]:


# Distribusi Pendapatan (Income)
plt.figure(figsize=(10, 6))
sns.histplot(data=customer, x='Income', bins=20, kde=True)
plt.title('Distribusi Pendapatan (Income) dalam Jutaan Rupiah')
plt.xlabel('Pendapatan (Income)')
plt.ylabel('Frekuensi')
plt.show()


# In[32]:


# Korelasi antara kolom numerik
correlation_matrix = customer.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriks Korelasi')
plt.show()


# In[33]:


Q1_Income = customer['Income'].quantile(0.25)
Q3_Income = customer['Income'].quantile(0.75)

IQR_Income = Q3_Income - Q1_Income

lower_bound_Income = Q1_Income - 1.5 * IQR_Income
upper_bound_Income = Q3_Income + 1.5 * IQR_Income

outliers_income = customer[(customer['Income'] < lower_bound_Income) | (customer['Income'] > upper_bound_Income)]

plt.figure(figsize=(8, 6))
plt.boxplot(customer['Income'], vert=False)
plt.title('Box Plot Income')
plt.show()

print("Outlier Income:")
print(outliers_income)


# In[34]:


def age_group(Age):
    if Age < 15:
        return '< 15'
    elif Age <= 24:
        return '15 - 24'
    elif Age <= 34:
        return '25 - 34'
    elif Age <= 44:
        return '35 - 44'
    elif Age <= 54:
        return '45 - 54'
    elif Age <= 64:
        return '55 - 64'
    else:
        return '> 65'

# Menerapkan fungsi untuk membuat kolom "Age_Group" baru
customer['Age_Group'] = customer['Age'].apply(age_group)

# Menampilkan DataFrame hasil
customer


# # PREPROCESSING & EDA DATA - PRODUCT

# In[35]:


product.head()


# In[36]:


product.shape


# In[37]:


# Informasi data
product.info()


# Format sudah benar semua, tidak ada yang perlu diubah

# In[38]:


# Statistik deskriptif
product.describe()


# In[39]:


# Menampilkan jumlah unik dari setiap kolom string
string_columns = ['ProductID', 'Product Name']
for col in string_columns:
    unique_values = product[col].nunique()
    print(f'Jumlah unik {col}: {unique_values}')


# In[40]:


# Distribusi Harga (Price)
plt.figure(figsize=(10, 6))
sns.histplot(data=product, x='Price', bins=10, kde=True)
plt.title('Distribusi Harga (Price)')
plt.xlabel('Harga (Price)')
plt.ylabel('Frekuensi')
plt.show()


# In[41]:


Q1_price = product['Price'].quantile(0.25)
Q3_price = product['Price'].quantile(0.75)

IQR_price = Q3_price - Q1_price

lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

outliers_price = product[(product['Price'] < lower_bound_price) | (product['Price'] > upper_bound_price)]

plt.figure(figsize=(8, 6))
plt.boxplot(product['Price'], vert=False)
plt.title('Box Plot Price')
plt.show()

print("Outlier Price:")
print(outliers_price)


# # PREPROCESSING & EDA DATA - STORE

# In[42]:


store.head()


# In[43]:


store.shape


# In[44]:


# Informasi data
store.info()


# Kita perlu untuk mengubah StoreID menjadi string, serta Latitude dan Longitude menjadi float

# In[45]:


#Mengganti "," menjadi "."
store['Latitude'] = store['Latitude'].str.replace(',', '.')
store['Longitude'] = store['Longitude'].str.replace(',', '.')
store


# In[46]:


# Mengubah tipe data kolom StoreID, Latitude, dan Longitude menjadi string
store['StoreID'] = store['StoreID'].astype(str)
store['Latitude'] = store['Latitude'].astype(float)
store['Longitude'] = store['Longitude'].astype(float)

# Menampilkan informasi tipe data setelah perubahan
store.info()


# In[47]:


# Menampilkan jumlah unik dari setiap kolom string
string_columns = ['StoreID', 'StoreName', 'GroupStore', 'Type']
for col in string_columns:
    unique_values = store[col].nunique()
    print(f'Jumlah unik {col}: {unique_values}')


# Terdapat dua toko yang memiliki ID berbeda

# # MERGED DATA

# In[48]:


# Melakukan left join antara transaction dan store berdasarkan kolom StoreID
merged_data = pd.merge(transaction, store, on="StoreID", how="left")

# Melakukan left join antara merged_data dan customer berdasarkan kolom CustomerID
merged_data = pd.merge(merged_data, customer, on="CustomerID", how="left")

# Melakukan left join antara merged_data dan product berdasarkan kolom ProductID
merged_data = pd.merge(merged_data, product, on="ProductID", how="left")

# Menampilkan contoh hasil gabungan
print("Hasil Gabungan dari Transaction, Store, Customer, dan Product:")
merged_data


# # REMOVE OUTLIER FROM MERGED DATA FOR ARIMA REGRESSION

# In[49]:


Q1_mg = merged_data['Qty'].quantile(0.25)
Q3_mg = merged_data['Qty'].quantile(0.75)

IQR_mg = Q3_mg - Q1_mg

lower_bound_mg = Q1_mg - 1.5 * IQR_mg
upper_bound_mg = Q3_mg + 1.5 * IQR_mg

outliers_mg = merged_data[(merged_data['Qty'] < lower_bound_mg) | (merged_data['Qty'] > upper_bound_mg)]

plt.figure(figsize=(8, 6))
plt.boxplot(merged_data['Qty'], vert=False)
plt.title('Box Plot Qty')
plt.show()

print("Outlier mg:")
print(outliers_mg)


# Terdapat 41 outlier, outlier dihapus karena dapat memberikan akurasi yang lebih buruk jika tidak dihapus

# In[50]:


# Identifikasi outlier menggunakan IQR
Q1 = merged_data['Qty'].quantile(0.25)
Q3 = merged_data['Qty'].quantile(0.75)
IQR = Q3 - Q1

# Tentukan batas atas dan bawah untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hapus baris yang mengandung outlier
merged_data2 = merged_data[(merged_data['Qty'] >= lower_bound) & (merged_data['Qty'] <= upper_bound)]


# In[51]:


merged_data2


# In[52]:


merged_data = merged_data2


# # REGRESI ARIMA

# In[53]:


# Mengubah kolom "Date" menjadi tipe data datetime jika belum
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Mengelompokkan data berdasarkan kolom "Date" dan menghitung jumlah "Qty"
data_regresi = merged_data.groupby('Date')['Qty'].sum().reset_index()

# Menampilkan hasil
data_regresi


# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Pisahkan data menjadi dua bagian: data pelatihan (train) dan data pengujian (test). Data pelatihan digunakan untuk melatih model ARIMA, sementara data pengujian digunakan untuk menguji performa model.

# In[55]:


# Memisahkan data menjadi data pelatihan (train) dan data pengujian (test)
train_size = int(len(data_regresi) * 0.8)
train_data = data_regresi[:train_size]
test_data = data_regresi[train_size:]


# In[56]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Melakukan decompose pada data
decomposed = seasonal_decompose(data_regresi.set_index('Date'))

plt.figure(figsize=(12, 8))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')

plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')

plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')

plt.tight_layout()
plt.show()


# In[57]:


train_data


# In[58]:


test_data


# Lakukan analisis awal pada data time series, seperti plotting data, identifikasi tren, dan komponen musiman jika ada. Selain itu, perlu memastikan bahwa data sudah stasioner. Kita bisa melakukan transformasi atau differencing jika diperlukan untuk mencapai stasioneritas.

# In[59]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 5))
sns.lineplot(data=train_data, x='Date', y='Qty', label='Train Data')
sns.lineplot(data=test_data, x='Date', y='Qty', label='Test Data')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.title('Plot Qty over Time')
plt.legend()
plt.show()


# In[60]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(train_data['Qty'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])


# Karena nilai p-value kurang dari tingkat signifikansi yang telah ditentukan (biasanya 0.05), maka dapat dikatakan bahwa data bersifat stasioner. Dari grafik juga terlihat bahwa data stasioner.

# In[61]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Menampilkan plot ACF hingga lag 100
plt.figure(figsize=(12, 6))
plot_acf(train_data['Qty'], lags=70)
plt.title('ACF Plot')
plt.show()

# Menampilkan plot PACF hingga lag 100
plt.figure(figsize=(12, 6))
plot_pacf(train_data['Qty'], lags=70)
plt.title('PACF Plot')
plt.show()


# In[62]:


# Menampilkan plot ACF dan PACF
plot_acf(train_data['Qty'])
plot_pacf(train_data['Qty'])
plt.show()


# In[63]:


from statsmodels.tsa.arima.model import ARIMA

# Membuat model ARIMA
model = ARIMA(train_data['Qty'], order=(37,0,19))
model_fit = model.fit()

# Menampilkan ringkasan hasil
print(model_fit.summary())


# In[64]:


# Melakukan prediksi pada data pengujian
predictions = model_fit.forecast(steps=len(test_data))


# In[65]:


predictions


# In[66]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Menghitung RMSE
rmse = np.sqrt(mean_squared_error(test_data['Qty'], predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Menghitung MAE
mae = mean_absolute_error(test_data['Qty'], predictions)
print(f'Mean Absolute Error (MAE): {mae}')


# In[67]:


# Menampilkan hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(test_data['Date'], test_data['Qty'], label='Data Pengujian', color='blue')
plt.plot(test_data['Date'], predictions, label='Prediksi', color='red')
plt.xlabel('Tanggal')
plt.ylabel('Qty')
plt.title('Prediksi Time Series dengan Model ARIMA')
plt.legend()
plt.show()


# In[68]:


# Menampilkan hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(train_data['Date'], train_data['Qty'], label='Data Pelatihan', color='green')
plt.plot(test_data['Date'], test_data['Qty'], label='Data Pengujian', color='blue')
plt.plot(test_data['Date'], predictions, label='Prediksi', color='red')
plt.xlabel('Tanggal')
plt.ylabel('Qty')
plt.title('Prediksi Time Series dengan Model ARIMA')
plt.legend()
plt.show()


# In[69]:


# Tanggal-tanggal yang ingin Anda prediksi (contoh: 1 Januari 2023 hingga 5 Januari 2023)
tanggal_prediksi = pd.date_range(start='2023-01-01', end='2023-01-05')

# Melakukan prediksi untuk tanggal-tanggal tersebut
prediksi = model_fit.predict(start=len(train_data), end=len(train_data) + len(tanggal_prediksi) - 1, typ='levels')

# Hasil prediksi
hasil_prediksi = pd.DataFrame({'Date': tanggal_prediksi, 'Qty_Prediction': prediksi})

# Tampilkan hasil prediksi
hasil_prediksi


# In[70]:


# Tanggal-tanggal yang ingin Anda prediksi (contoh: 1 Januari 2023 hingga 5 Januari 2023)
tanggal_prediksi = pd.date_range(start='2023-01-01', end='2023-01-05')

# Melakukan prediksi untuk tanggal-tanggal tersebut
prediksi = model_fit.predict(start=len(train_data), end=len(train_data) + len(tanggal_prediksi) - 1, typ='levels')

# Hasil prediksi
hasil_prediksi = pd.DataFrame({'Date': tanggal_prediksi, 'Qty_Prediction': prediksi})

# Plot hasil prediksi
plt.figure(figsize=(12, 6))
plt.plot(train_data['Date'], train_data['Qty'], label='Data Training', color='blue')
plt.plot(test_data['Date'], test_data['Qty'], label='Data Testing', color='green')
plt.plot(hasil_prediksi['Date'], hasil_prediksi['Qty_Prediction'], label='Prediksi', color='red')
plt.xlabel('Tanggal')
plt.ylabel('Qty')
plt.title('Prediksi Time Series dengan Model ARIMA')
plt.legend()
plt.show()


# # DATA FOR CLUSTERING

# In[71]:


# Groupby 'CustomerID' dan lakukan agregasi
cluster_data = transaction.groupby('CustomerID').agg({
    'TransactionID': 'count', 
    'Qty': 'sum',              
    'TotalAmount': 'sum'       
}).reset_index() 

# Ubah nama kolom agar lebih deskriptif
cluster_data.rename(columns={
    'TransactionID': 'TransactionCount',
    'Qty': 'TotalQty',
    'TotalAmount': 'TotalAmountSum'
}, inplace=True)

# Tampilkan hasilnya
cluster_data


# # REMOVE OUTLIER FROM DATA FOR CLUSTERING

# In[72]:


Q1_mg = cluster_data['TotalQty'].quantile(0.25)
Q3_mg = cluster_data['TotalQty'].quantile(0.75)

IQR_mg = Q3_mg - Q1_mg

lower_bound_mg = Q1_mg - 1.5 * IQR_mg
upper_bound_mg = Q3_mg + 1.5 * IQR_mg

outliers_mg = cluster_data[(cluster_data['TotalQty'] < lower_bound_mg) | (cluster_data['TotalQty'] > upper_bound_mg)]

plt.figure(figsize=(8, 6))
plt.boxplot(cluster_data['TotalQty'], vert=False)
plt.title('Box Plot TotalQty')
plt.show()

print("Outlier mg:")
print(outliers_mg)


# In[73]:


import pandas as pd

# Identifikasi outlier menggunakan IQR
Q1 = cluster_data['TotalQty'].quantile(0.25)
Q3 = cluster_data['TotalQty'].quantile(0.75)
IQR = Q3 - Q1

# Tentukan batas atas dan bawah untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hapus baris yang mengandung outlier
cluster_data2 = cluster_data[(cluster_data['TotalQty'] >= lower_bound) & (cluster_data['TotalQty'] <= upper_bound)]


# In[74]:


cluster_data2


# In[75]:


Q1_mg = cluster_data['TotalAmountSum'].quantile(0.25)
Q3_mg = cluster_data['TotalAmountSum'].quantile(0.75)

IQR_mg = Q3_mg - Q1_mg

lower_bound_mg = Q1_mg - 1.5 * IQR_mg
upper_bound_mg = Q3_mg + 1.5 * IQR_mg

outliers_mg = cluster_data[(cluster_data['TotalAmountSum'] < lower_bound_mg) | (cluster_data['TotalAmountSum'] > upper_bound_mg)]

plt.figure(figsize=(8, 6))
plt.boxplot(cluster_data['TotalAmountSum'], vert=False)
plt.title('Box Plot TotalAmountSum')
plt.show()

print("Outlier mg:")
print(outliers_mg)


# Outlier-outlier tersebut akan dihapus agar akurasi model tidak menjadi lebih buruk

# In[76]:


import pandas as pd

# Identifikasi outlier menggunakan IQR
Q1 = cluster_data2['TotalAmountSum'].quantile(0.25)
Q3 = cluster_data2['TotalAmountSum'].quantile(0.75)
IQR = Q3 - Q1

# Tentukan batas atas dan bawah untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hapus baris yang mengandung outlier
cluster_data2 = cluster_data2[(cluster_data2['TotalAmountSum'] >= lower_bound) & (cluster_data2['TotalAmountSum'] <= upper_bound)]


# In[77]:


cluster_data2


# In[78]:


Q1_mg = cluster_data['TransactionCount'].quantile(0.25)
Q3_mg = cluster_data['TransactionCount'].quantile(0.75)

IQR_mg = Q3_mg - Q1_mg

lower_bound_mg = Q1_mg - 1.5 * IQR_mg
upper_bound_mg = Q3_mg + 1.5 * IQR_mg

outliers_mg = cluster_data[(cluster_data['TransactionCount'] < lower_bound_mg) | (cluster_data['TransactionCount'] > upper_bound_mg)]

plt.figure(figsize=(8, 6))
plt.boxplot(cluster_data['TransactionCount'], vert=False)
plt.title('Box Plot TransactionCount')
plt.show()

print("Outlier mg:")
print(outliers_mg)


# In[79]:


import pandas as pd

# Identifikasi outlier menggunakan IQR
Q1 = cluster_data2['TransactionCount'].quantile(0.25)
Q3 = cluster_data2['TransactionCount'].quantile(0.75)
IQR = Q3 - Q1

# Tentukan batas atas dan bawah untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hapus baris yang mengandung outlier
cluster_data2 = cluster_data2[(cluster_data2['TransactionCount'] >= lower_bound) & (cluster_data2['TransactionCount'] <= upper_bound)]


# In[80]:


cluster_data2


# In[81]:


cluster_data = cluster_data2


# In[82]:


selected_columns = ['TransactionCount', 'TotalQty', 'TotalAmountSum']
cluster_data = cluster_data[selected_columns]


# In[83]:


from sklearn.preprocessing import MinMaxScaler

# Inisialisasi MinMaxScaler
scaler = MinMaxScaler()

# Menggunakan scaler untuk melakukan normalisasi pada data
normalized_data = scaler.fit_transform(cluster_data)

# Membuat DataFrame baru dari hasil normalisasi
normalized_df = pd.DataFrame(normalized_data, columns=cluster_data.columns)

# Memeriksa DataFrame yang sudah dinormalisasi
normalized_df


# In[84]:


cluster_data = normalized_df


# # K-MEANS CLUSTERING WITH NORMALIZED DATA

# Untuk menentukan jumlah cluster di awal akan menggunakan Elbow Method, selanjutnya untuk evaluasinya akan digunakan Silhouette Score dan Davies Bouldin Index

# In[98]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Anggaplah Anda memiliki DataFrame cluster_data yang sudah disiapkan

# Menyimpan nilai SSE (Sum of Squared Errors) untuk berbagai nilai k
sse = []
silhouette_scores = []
davies_bouldin_scores = []

# Coba nilai k dari 2 hingga 10
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_data)
    
    # Menambahkan SSE ke dalam list
    sse.append(kmeans.inertia_)
    
    # Menghitung silhouette score dan menambahkannya ke dalam list
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(cluster_data, labels)
    silhouette_scores.append(silhouette_avg)
    
    # Menghitung Davies-Bouldin Index dan menambahkannya ke dalam list
    davies_bouldin_avg = davies_bouldin_score(cluster_data, labels)
    davies_bouldin_scores.append(davies_bouldin_avg)

# Menampilkan plot SSE
plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), sse, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Menampilkan plot Silhouette Score
plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='g')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)
plt.show()

# Menampilkan plot Davies-Bouldin Index
plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), davies_bouldin_scores, marker='o', linestyle='-', color='r')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index for Optimal k')
plt.grid(True)
plt.show()


# In[86]:


from sklearn.cluster import KMeans

# Inisialisasi model K-Means dengan jumlah cluster yang diinginkan (misalnya, 3)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Melakukan clustering pada data
cluster_labels = kmeans.fit_predict(cluster_data)

# Menambahkan kolom cluster_labels ke DataFrame
cluster_data['Cluster'] = cluster_labels


# In[87]:


from mpl_toolkits.mplot3d import Axes3D

# Inisialisasi plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot hasil clustering berdasarkan TransactionCount, TotalQty, dan TotalAmountSum
for cluster_num in range(n_clusters):
    cluster = cluster_data[cluster_data['Cluster'] == cluster_num]
    ax.scatter(cluster['TransactionCount'], cluster['TotalQty'], cluster['TotalAmountSum'], label=f'Cluster {cluster_num}')

# Menambahkan label sumbu x, y, dan z
ax.set_xlabel('TransactionCount')
ax.set_ylabel('TotalQty')
ax.set_zlabel('TotalAmountSum')

# Menambahkan legenda
plt.legend()

# Menampilkan plot
plt.title('Hasil Clustering K-Means (3D Plot)')
plt.show()


# In[88]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import Parallel, delayed

# Extract features for clustering
X = cluster_data

# Number of repetitions for stability analysis
num_repeats = 10

# Number of clusters
num_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Define a function to compute scores for a given cluster number
def compute_scores(num_cluster, X):
    sil_scores_kmeans = []
    sil_scores_gmm = []
    dbi_scores_kmeans_rep = []
    dbi_scores_gmm_rep = []

    for _ in range(num_repeats):
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=num_cluster, random_state=0)
        kmeans_labels = kmeans.fit_predict(X)
        sil_scores_kmeans.append(silhouette_score(X, kmeans_labels))
        dbi_scores_kmeans_rep.append(davies_bouldin_score(X, kmeans_labels))

        # Apply GMM clustering
        gmm = GaussianMixture(n_components=num_cluster, covariance_type='tied', random_state=0)
        gmm_labels = gmm.fit_predict(X)
        sil_scores_gmm.append(silhouette_score(X, gmm_labels))
        dbi_scores_gmm_rep.append(davies_bouldin_score(X, gmm_labels))

    avg_sil_kmeans = np.mean(sil_scores_kmeans)
    avg_sil_gmm = np.mean(sil_scores_gmm)
    avg_dbi_kmeans = np.mean(dbi_scores_kmeans_rep)
    avg_dbi_gmm = np.mean(dbi_scores_gmm_rep)

    return num_cluster, avg_sil_kmeans, avg_sil_gmm, avg_dbi_kmeans, avg_dbi_gmm

# Use parallel processing to compute scores for different cluster numbers
results = Parallel(n_jobs=-1)(delayed(compute_scores)(num_cluster, X) for num_cluster in num_clusters)

# Print the results
for num_cluster, avg_sil_kmeans, avg_sil_gmm, avg_dbi_kmeans, avg_dbi_gmm in results:
    print(f"Number of Clusters: {num_cluster}")
    print(f"K-Means - Silhouette Score: {avg_sil_kmeans}, Davies-Bouldin Index: {avg_dbi_kmeans}")
    print(f"GMM - Silhouette Score: {avg_sil_gmm}, Davies-Bouldin Index: {avg_dbi_gmm}\n")


# Didapat cluster optimalnya yaitu 3, berdasarkan elbow method, lalu evaluasi menggunakan silhouette score dan davies bouldin index (DBI)

# In[89]:


cluster_data


# # CLUSTER'S INTERPRETATION

# In[90]:


import missingno as msno
import seaborn as sns

fig, ax = plt.subplots(nrows= 1, ncols = 3, figsize= (14,6))
tt=sns.boxplot(x='Cluster', y='TransactionCount', data=cluster_data, ax = ax[0], palette='coolwarm_r')
tt.set_title('Clusters based on TransactionCount')
tt.set_ylabel('TransactionCount')
tt.set_xlabel('Clusters')

tr=sns.boxplot(x='Cluster', y='TotalQty', data=cluster_data, ax = ax[1], palette='magma_r')
tr.set_title('Clusters based on TotalQty')
tr.set_ylabel('TotalQty')
tr.set_xlabel('Clusters')

tm=sns.boxplot(x='Cluster', y='TotalAmountSum', data=cluster_data, ax = ax[2], palette='magma_r')
tm.set_title('Clusters based on TotalAmountSum')
tm.set_ylabel('TotalAmountSum')
tm.set_xlabel('Clusters')


# In[91]:


cluster_data2 = cluster_data2.reset_index()


# Selanjutnya akan digunakan data sebelum normalisasi untuk interpretasi, lalu ditambahkan kolom "Cluster" ke data nya

# In[92]:


# Menggabungkan kolom "Cluster" dari cluster_data ke cluster_data2 berdasarkan indeks
cluster_data2 = cluster_data2.join(cluster_data['Cluster'])


# In[93]:


cluster_data2


# In[94]:


# Menghapus kolom "index" dari DataFrame cluster_data2
cluster_data2.drop('index', axis=1, inplace=True)


# In[95]:


cluster_data2


# In[96]:


cluster_data2.info()


# In[97]:


# Grouping data berdasarkan kolom "Cluster" dan menghitung statistik
cluster_stats = cluster_data2.groupby('Cluster')[['TransactionCount', 'TotalQty', 'TotalAmountSum']].agg(
    {
        'TransactionCount': ['min', 'max', 'mean', 'var'],
        'TotalQty': ['min', 'max', 'mean', 'var'],
        'TotalAmountSum': ['min', 'max', 'mean', 'var']
    }
)

# Menampilkan hasil statistik
cluster_stats


# Karena tujuan dari project ini adalah untuk membuat segment customer, dimana customer ini nantinya akan digunakan oleh tim marketing untuk memberikan personalized promotion dan sales treatment.
# 
# Oleh karena itu dapat dilihat bahwa dari hasil menggunakan Elbow Method untuk menentukan jumlah cluster diawal, lalu melakukan evaluasi menggunakan Silhouette Score, dan Davies Bouldin Index, didapat bahwa jumlah k = 3 merupakan jumlah klaster optimal pada kasus ini.
# 
# Berikut merupakan **interpretasinya**:
# 
# **Cluster 0**:
# 
# 1. Terdiri dari customer dengan rentang TransactionCount antara 3 hingga 10, dengan rata-rata sekitar 7.01.
# 2. TotalQty berkisar antara 10 hingga 37, dengan rata-rata sekitar 24.67.
# 3. TotalAmountSum berkisar antara 84,300 hingga 319,200, dengan rata-rata sekitar 207,579.25.
# 4. Variansi dari setiap atribut dalam cluster ini relatif rendah, menunjukkan bahwa data dalam cluster ini cenderung stabil dan tidak terlalu bervariasi.
# 5. Cluster ini mungkin dapat disebut sebagai "Customer Reguler" karena memiliki frekuensi transaksi yang lebih rendah dan total pembelian yang lebih kecil dibandingkan dengan cluster lainnya.
# 
# **Cluster 1**:
# 
# 1. Terdiri dari customer dengan rentang TransactionCount antara 11 hingga 18, dengan rata-rata sekitar 14.59.
# 2. TotalQty berkisar antara 43 hingga 73, dengan rata-rata sekitar 54.93.
# 3. TotalAmountSum berkisar antara 348,200 hingga 676,200, dengan rata-rata sekitar 497,099.19.
# 4. Variansi dari setiap atribut dalam cluster ini juga cukup rendah, menunjukkan stabilitas dalam perilaku pembelian.
# 5. Cluster ini mungkin dapat disebut sebagai "Customer Aktif" karena memiliki frekuensi transaksi yang lebih tinggi dan total pembelian yang lebih besar dibandingkan dengan cluster 0.
# 
# **Cluster 2**:
# 
# 1. Terdiri dari customer dengan rentang TransactionCount antara 7 hingga 15, dengan rata-rata sekitar 10.70.
# 2. TotalQty berkisar antara 26 hingga 53, dengan rata-rata sekitar 38.33.
# 3. TotalAmountSum berkisar antara 215,100 hingga 476,200, dengan rata-rata sekitar 336,869.63.
# 4. Seperti cluster lainnya, variansi dari setiap atribut dalam cluster ini relatif rendah.
# 5. Cluster ini mungkin dapat disebut sebagai "Customer Menengah" karena memiliki tingkat transaksi dan total pembelian yang berada di tengah-tengah antara cluster 0 dan 1.
# 
# Melalui hasil yang didapatkan, tim pemasaran dapat merancang strategi promosi yang lebih sesuai dengan masing-masing jenis customer.

# **SARAN**
# 
# **Saran untuk Cluster 0 ("Customer Reguler"):**
# Cluster ini terdiri dari pengguna dengan frekuensi transaksi yang lebih rendah dan total pembelian yang lebih kecil. Untuk meningkatkan frekuensi transaksi mereka, berikut saran yang dapat dipertimbangkan:
# 
# 1. **Program Loyalitas:**  
# Dengan meluncurkan program loyalitas khusus untuk anggota Cluster 0. Tawarkan imbalan atau insentif seperti diskon eksklusif, cashback, atau poin loyalitas setiap kali mereka berbelanja.
# 
# 2. **Promosi Berkala:** 
# Kirimkan mereka penawaran promosi berkala melalui email atau pesan teks untuk mendorong mereka bertransaksi lebih sering. Selain itu, pastikan promosi ini relevan dengan preferensi dan riwayat pembelian mereka.
# 
# 3. **Bundle Deals:** 
# Tawarkan paket produk dengan harga khusus atau bundel produk yang sering dibeli bersama-sama. Hal ini dapat mendorong pembelian yang lebih besar.
# 
# 4. **Edukasi Produk:** 
# Jika ada produk yang mungkin belum mereka kenal, berikan informasi dan ulasan yang bermanfaat untuk mendorong mereka mencoba produk baru.
# 
# **Saran untuk Cluster 1 ("Customer Aktif"):**
# Cluster ini memiliki frekuensi transaksi yang lebih tinggi dan total pembelian yang lebih besar. Mereka adalah calon yang baik untuk produk unggulan. Berikut saran yang dapat dipertimbangkan:
# 
# 1. **Highlight Produk Unggulan:** 
# Fokuskan promosi pada produk-produk unggulan yang mungkin menarik bagi merek, termasuk produk dengan margin keuntungan tinggi atau produk yang paling sering dibeli oleh kelompok ini.
# 
# 2. **Promosi Cross-Selling:** 
# Tawarkan produk tambahan atau layanan terkait dengan pembelian mereka saat ini. Contohnya, jika mereka membeli oat, maka bisa dilakukan penawaran untuk membeli yoghurt atau susu juga sebagai pilihan sarapan sehat.
# 
# 3. **Program Reward Tier:** 
# Buat program reward dengan tingkat. Semakin banyak mereka berbelanja, semakin banyak hadiah atau keuntungan yang mereka dapatkan.
# 
# 4. **Pelayanan Customer Unggulan:** 
# Berikan pelayanan customer yang paling unggul kepada anggota Cluster 1. Pastikan mereka merasa dihargai dan mendapatkan respon yang cepat terhadap pertanyaan atau masalah mereka.
# 
# 5. **Promosi Eksklusif:** 
# Tawarkan akses eksklusif ke penjualan atau acara khusus yang tidak tersedia untuk umum, karena hal ini dapat membuat mereka merasa istimewa.
# 
# **Saran untuk Cluster 2 ("Customer Menengah"):**
# Cluster ini berada di tengah-tengah dalam hal frekuensi transaksi dan total pembelian. Berikut saran yang dapat dipertimbangkan:
# 
# 1. **Program Diskon Bertahap:** 
# Buat program diskon atau cashback dengan tingkat yang berkembang seiring dengan frekuensi transaksi. Semakin sering mereka berbelanja, semakin banyak diskon yang mereka dapatkan.
# 
# 2. **Bundel Produk:** 
# Tawarkan bundel produk dengan harga khusus yang menggabungkan produk yang sering dibeli bersama-sama oleh anggota Cluster 2.
# 
# 3. **Rekomendasi Produk Personalisasi:** 
# Gunakan data pembelian mereka untuk memberikan rekomendasi produk yang sangat relevan dalam komunikasi pemasaran Anda.
# 
# 4. **Promosi Tematik:** 
# Sesuaikan promosi dengan musim atau acara khusus. Misalnya, promosikan produk musiman atau penawaran liburan.
# 
# 5. **Feedback Loop:** 
# Selalu berkomunikasi dengan anggota Cluster 2 untuk mendengar pendapat mereka tentang produk dan layanan. Umpan balik mereka dapat membantu untuk memperbaiki pengalaman mereka terhadap pembelian produk di toko kita.
# 
# Hal ini merupakan saran umum, dan penting untuk terus memantau dan mengevaluasi respons customer terhadap strategi-promosi yang selanjutnya akan diterapkan. Penggunaan metode seperti Market Basket Analysis akan sangat berguna untuk menganalisis pola pembelian customer dan mengidentifikasi produk yang sering dibeli bersamaan. Dengan menerapkan analisis ini dapat memberikan wawasan yang lebih dalam tentang hubungan antara produk dan mengidentifikasi peluang cross-selling yang lebih efektif.

# **THANK YOU :)**
