import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = 'C:\\Users\\megal\\Desktop\\MSDS696Project\\PlantCV\\Plant'

total = 0
for sub_dir in os.listdir(path):
    target_dir = path + '/' + sub_dir
    files = os.listdir(target_dir)
    count = len(files)
    total += count
    print(f'{sub_dir}: {count}')


plant=[["complex", 1602],
["frog eye leaf spot", 3181],
["frog eye leaf spot complex", 165],
["healthy", 4624],
["powdery mildew", 1184],
["powdery mildew complex", 87],
["rust", 1860],
["rust complex", 97],
["rust frog eye leaf spot", 120],
["scab", 4826],
["scab frog eye leaf spot", 686],
["scab frog eye leaf spot complex", 200]]

plant = pd.DataFrame(plant,columns=['Class','Image Count'])

path = 'C:\\Users\\megal\\Desktop\\MSDS696Project\\PlantCV\\Tomato'

total = 0
for sub_dir in os.listdir(path):
    target_dir = path + '/' + sub_dir
    files = os.listdir(target_dir)
    count = len(files)
    total += count
    print(f'{sub_dir}: {count}')

tomato=[["Bacterial spot",2826],
["Early blight", 2455],
["Healthy", 3051],
["Late blight", 3113],
["Leaf mold", 2754],
["Powdery mildew", 1004],
["Septoria leaf spot", 2882],
["Target Spot", 1827],
["Tomato mosaic virus", 2153],
["Tomato yellow leaf curl virus", 2036],
["Two-spotted spider mite", 1747]]

tomato = pd.DataFrame(tomato,columns=['Class','Image Count'])


x=plant['Class']
y=plant['Image Count']
plt.figure(figsize=(10,5))

plt.title("General Plant: Image Counts by Class")

def addlabels(x,y):
    for i in range(len(plant)):
        plt.text(i, y[i], y[i], ha = 'center')

chart = sns.barplot(
    data=plant,
    x=x,
    y=y,
    palette='Set1',
)
addlabels(x,y)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

def addlabels2(x,y):
    for i in range(len(tomato)):
        plt.text(i, y[i], y[i], ha = 'center')


x=tomato['Class']
y=tomato['Image Count']
plt.figure(figsize=(12,6))

plt.title("Tomato: Image Counts by Class")

chart = sns.barplot(
    data=tomato,
    x=x,
    y=y,
    palette='Set1',
)
addlabels2(x,y)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')