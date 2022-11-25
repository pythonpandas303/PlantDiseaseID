### Final Project
### MSDS696
### Tom Teasdale

# Importing required libraries

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting path

path = 'C:\\Users\\megal\\Desktop\\MSDS696Project\\PlantCV\\Plant'

# Recursive function to count images in subdirs

total = 0
for sub_dir in os.listdir(path):
    target_dir = path + '/' + sub_dir
    files = os.listdir(target_dir)
    count = len(files)
    total += count
    print(f'{sub_dir}: {count}')
    
# From print function, creating dictionary to plot from


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

# Creating data frame from dictionary

plant = pd.DataFrame(plant,columns=['Class','Image Count'])

# Setting path

path = 'C:\\Users\\megal\\Desktop\\MSDS696Project\\PlantCV\\Tomato'

# Recursive function to count images in subdirs

total = 0
for sub_dir in os.listdir(path):
    target_dir = path + '/' + sub_dir
    files = os.listdir(target_dir)
    count = len(files)
    total += count
    print(f'{sub_dir}: {count}')
    
# From print function, creating dictionary to plot from

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

# Creating data frame from dictionary

tomato = pd.DataFrame(tomato,columns=['Class','Image Count'])

# Plotting of image counts, adjust size and titles as necessary

x=plant['Class']
y=plant['Image Count']
plt.figure(figsize=(10,5))

plt.title("General Plant: Image Counts by Class")

# Funtion to add counts as labels. 

def addlabels(x,y):
    for i in range(len(plant)):
        plt.text(i, y[i], y[i], ha = 'center')
        
# Plotting bar chart in Seaborn library

chart = sns.barplot(
    data=plant,
    x=x,
    y=y,
    palette='Set1',
)

# Calling addlabels function and rotation of x-ticks

addlabels(x,y)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')

def addlabels2(x,y):
    for i in range(len(tomato)):
        plt.text(i, y[i], y[i], ha = 'center')

# Plotting of image counts, adjust size and titles as necessary

x=tomato['Class']
y=tomato['Image Count']
plt.figure(figsize=(12,6))

plt.title("Tomato: Image Counts by Class")

# Plotting bar chart in Seaborn library

chart = sns.barplot(
    data=tomato,
    x=x,
    y=y,
    palette='Set1',
)

# Calling addlabels function and rotation of x-ticks

addlabels2(x,y)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')