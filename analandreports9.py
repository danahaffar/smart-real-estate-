# create dummy data 
import pandas as pd

properties = pd.DataFrame({
    'property_id': range(1, 11),
    'city': ['Damascus','Damascus','Aleppo','Homs','Latakia']*2,
    'price': [180000,200000,220000,150000,170000]*2,
    'area': [100,120,130,90,110]*2,
    'type': ['Apartment','Apartment','House','Apartment','Apartment']*2
})

searches = pd.DataFrame({
    'user_id': range(1, 11),
    'city': ['Damascus','Aleppo','Damascus','Homs','Latakia']*2,
    'min_price': [150000,160000,180000,140000,155000]*2,
    'max_price': [230000,240000,260000,200000,220000]*2
})
# تقرير ( متوسط الأسعار حسب المدينة )
avg_price_city = properties.groupby('city')['price'].mean().reset_index()
print(avg_price_city)
# تقرير (عدد العقارات في كل مدينة )
count_city = properties['city'].value_counts().reset_index()
count_city.columns = ['city', 'number_of_properties']
print(count_city)
# تقرير عن أكثر المدن طلبا 
top_searched_cities = searches['city'].value_counts().reset_index()
top_searched_cities.columns = ['city', 'search_count']
print(top_searched_cities)

import pandas as pd

properties = pd.DataFrame({
    'price': [180000,200000,220000,150000,170000],
    'area': [100,120,130,90,110]
})

correlation = properties['price'].corr(properties['area'])
print("Correlation between price and area:", correlation)
import matplotlib.pyplot as plt
import pandas as pd

# بيانات العقارات (مثال)
properties = pd.DataFrame({
    'area': [100, 120, 130, 90, 110, 140, 160],
    'price': [180000, 200000, 220000, 150000, 170000, 240000, 260000]
})

# رسم مخطط Scatter يوضح العلاقة بين السعر والمساحة 
plt.scatter(properties['area'], properties['price'])
plt.xlabel('Area (m²)')
plt.ylabel('Price')
plt.title('Scatter Plot: Price vs Area')
plt.show()
