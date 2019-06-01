from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]
        }

df = DataFrame(Cars, columns= ['Brand', 'Price'])

export_csv = df.to_csv (r'GIS/export_dataframe.txt', index = None, header=True) #Don't forget to add '.csv' at the end of the path
print(export_csv)
print (df)