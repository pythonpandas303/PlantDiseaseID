### MSDS696 Project ###
### Instructor: John Koenig ###
### Tom Teasdale ###
import os
import pandas as pd
import nbformat
import pathlib
import glob
import plotly_express as px

## Plant Pathology data set came in as one large folder containing all images with a seperate .csv file
## with labels and train, test, validation datasets. This chunk seperates images out into 
## their appropriate labels within the Plant folder 

df = pd.read_csv('data.csv')
for _, row in df.iterrows():
  f = row['filepaths']
  l = row['labels']
  os.replace(f'./{f}', f'./{l}/{f}')
  

### Timeline visual for project. Update as necessary. 

df2 = pd.DataFrame([
    dict(Task="Data Preperation", Start='2022-10-17', Finish='2022-10-23', Resource='Complete'),
    dict(Task="EDA/Visualization", Start='2022-10-24', Finish='2022-11-06', Resource='Complete'),
    dict(Task="Modeling", Start='2022-11-07', Finish='2022-11-16', Resource='Complete'),
    dict(Task="Testing/Optimization", Start='2022-11-17', Finish='2022-11-21', Resource='In progress'),
    dict(Task="Interpret Results", Start='2022-11-22', Finish='2022-11-27', Resource='Not Started'),
    dict(Task="Presentation/Final Delivery", Start='2022-11-28', Finish='2022-12-02', Resource='Not Started')
])

fig = px.timeline(df2, x_start="Start", x_end="Finish", y="Task", color="Resource", height=(800), width=(1600))
fig.update_yaxes(autorange="reversed") 
fig.show()
fig.write_image('timeline.png')


## Renaming all files in Tomato folder ###


folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Bacterial_spot/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Bacterial_spot" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Early_blight/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Early_blight" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Late_blight/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Late_blight" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Leaf_Mold/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Leaf_Mold" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Septoria_leaf_spot/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Septoria_leaf_spot" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Spider_mites Two-spotted_spider_mite/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Spider_mites Two-spotted_spider_mite" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Target_Spot/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Target_Spot" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Tomato_Yellow_Leaf_Curl_Virus/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Tomato_Yellow_Leaf_Curl_Virus" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/Tomato_mosaic_virus/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/Tomato_mosaic_virus" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/healthy/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/healthy" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

folder = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/powdery_mildew/"
count = 1
for file_name in os.listdir(folder):
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "/powdery_mildew" + str(count) + ".jpg"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

### Removing punctuation from file names ###


dataset_path = "C:/Users/megal/Desktop/MSDS696Project/PlantCV/Tomato/"

for directname, directnames, files in os.walk(dataset_path):
    for f in files:
        filename, ext = os.path.splitext(f)
        if "." in filename:
            new_name = filename.replace(".", "")
            os.rename(
                os.path.join(directname, f),
                os.path.join(directname, new_name + ext))

### Checking that images are in the appropriate format, removing if not ###


import cv2
import imghdr

def check_images(s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for klass in s_list:
        klass_path=os.path.join (s_dir, klass)
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            for f in file_list:               
                f_path=os.path.join (klass_path,f)
                tip = imghdr.what(f_path)
                if ext_list.count(tip) == 0:
                  bad_images.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img=cv2.imread(f_path)
                        shape=img.shape
                    except:
                        print('file ', f_path, ' is not a valid image file')
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

source_dir=dataset_path
good_exts=['jpg', 'png', 'jpeg', 'gif', 'bmp' ] # list of acceptable extensions
bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
if len(bad_file_list) !=0:
    print('improper image files are listed below')
    for i in range (len(bad_file_list)):
        print (bad_file_list[i])
else:
    print(' no improper image files were found')
    
    
if len(bad_file_list) !=0:
    os.remove(bad_file_list)
else:
    print('No files to remove!')








