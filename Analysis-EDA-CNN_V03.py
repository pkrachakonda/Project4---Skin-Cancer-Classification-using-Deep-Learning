################################################################
##                     EDA - Overview                         ##
################################################################
# ignore warnings
import warnings  # tf needs to learn to stfu
warnings.simplefilter(action="ignore", category=FutureWarning)

# general libraries
from PIL import Image
from glob import glob

# data analysis library
import numpy as np
import pandas as pd

# data visualization
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# To see the value of multiple statements at once.
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

###################################################################

DATA_ROOT_PATH = "Project_Datasets/"
TRAIN_CSV = DATA_ROOT_PATH + "ISIC2019_Metadata.csv"
TEST_CSV = DATA_ROOT_PATH + "ISIC_2019_Test_Metadata.csv"

train_df = pd.read_csv(TRAIN_CSV, na_values=['unknown'])
test_df = pd.read_csv(TEST_CSV)
trn_len_df = len(train_df)
tst_len_df = len(test_df)
print(f"There are {trn_len_df} images in the training set")
print(f"There are {tst_len_df} images in the test set")

print(train_df.head())
print(test_df.head())

################################################################
##                EDA - Meta-features                         ##
################################################################

# NaN values
nan_stats = train_df.isna().sum() / len(train_df) * 100
stats = pd.DataFrame({
    'columns': train_df.columns,
    'NaN statistics (in %)': nan_stats
})

stats = stats.sort_values(by=['NaN statistics (in %)'], ascending=False)

stats = stats.reset_index(drop=True)
print(stats.head(5))

# Patient distribution
print(f"There are {train_df['patient_id'].nunique()} unique patients for {len(train_df)} images in the training set.")

fig_1, ax1 = plt.subplots(1, 1, figsize=(15, 10))
ax1.set_title('Patient distribution in the training set')
sns.countplot(train_df, x =train_df['patient_id'], hue=train_df['patient_id'], dodge = False, ax=ax1, legend= False)
plt.show()
plt.savefig("images/Fig_Patient distribution in the training set.jpg")

trn_patients = set(train_df['isic_id'])
tst_patients = set(test_df['image'])
inter_patients = len(trn_patients.intersection(tst_patients))
print(f'There are {inter_patients} common patients in the training and test sets.')

# Gender distribution
fig_2, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(x=train_df['sex'], hue=train_df['sex'], ax=ax[0])
ax[0].set_title("Sex distribution in the training set")

sns.countplot(x=test_df['sex'], hue=test_df['sex'], ax=ax[1])
ax[1].set_title("Sex distribution in the test set")

plt.show()
plt.savefig("images/Fig_Gender distribution in the training and test sets.jpg")

# Age distribution
fig_3, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.distplot(x=train_df['age_approx'], ax=ax[0])
ax[0].set_title("Age distribution in the training set")

sns.distplot(x=test_df['age_approx'], ax=ax[1])
ax[1].set_title("Age distribution in the test set")

plt.show()
plt.savefig("images/Fig_Age distribution in the training and test sets.jpg")

# Anatomical site Distribution
fig_4, ax = plt.subplots(1, 2, figsize=(5, 5))

chart = sns.countplot(x=train_df['anatom_site_general'], hue=train_df['anatom_site_general'], ax=ax[0])
ax[0].set_title("Anatomical site in the training set")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart2 = sns.countplot(x=test_df['anatom_site_general'], hue=test_df['anatom_site_general'], ax=ax[1])
ax[1].set_title("Anatomical site in the test set")
chart2.set_xticklabels(chart2.get_xticklabels(), rotation=45)

plt.show()
plt.savefig("images/Fig_Anatomical site distribution in the training and test sets.jpg")

## Diagnosis Distribution

chart3 = sns.countplot(x=train_df['diagnosis'], hue=train_df['diagnosis'])
chart3.set_xticklabels(chart3.get_xticklabels(), rotation=45)
chart3.set_title("Distribution of Diagnosis in training set")
plt.show()
plt.savefig('images/Fig_Distribution of Diagnosis in training set.jpg')
################################################################
##              EDA - Target distribution                     ##
################################################################

print(f"There are {len(train_df[train_df['target'] == 0])} negative labels.")
print(f"There are {len(train_df[train_df['target'] == 1])} positive labels.")

sns.countplot(x=train_df['target'], hue=train_df['target'])
plt.title("Type of Diagnosis Bening (0) vs Malignant (1)")
plt.show()
plt.savefig('images/Fig_Diagnosis Type.jpg')

################################################################
##                  EDA - Correlation                         ##
################################################################

fig_5 = plt.figure(figsize=(7, 5))
ax = sns.countplot(x="target", hue="sex", data=train_df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 10, '{:1.2f}%'.format(100 * height / len(train_df)), ha="center")
plt.title('Relation between Diagnosis and Gender')
plt.show()
plt.savefig("images/Fig_Relation between Diagnosis and Gender.jpg")

fig_6 = plt.figure(figsize=(7, 5))
ax = sns.countplot(x="target", hue="anatom_site_general", data=train_df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 15, '{:1.2f}%'.format(100 * height / len(train_df)), ha="center")
plt.title('Relation between Diagnosis and Anatomical site')
plt.show()
plt.savefig("images/Fig_Relation between Diagnosis and Anatomical site.jpg")

fig_7 = plt.figure(figsize=(7, 5))
ax = sns.countplot(x="target", hue="age_approx", data=train_df)

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 15, '{:1.2f}%'.format(100 * height / len(train_df)), ha="center")
plt.title('Relation between Diagnosis and Age')
plt.show()
plt.savefig("images/Fig_Relation between Diagnosis and Age.jpg")

################################################################
##                  EDA - Image visualization                 ##
################################################################

# Training set

img_names = glob('Project_Datasets\ISIC_2019_Training_Input\*.jpg')

fig_8, ax = plt.subplots(4, 4, figsize=(20, 20))

for i in range(16):
    x = i // 4
    y = i % 4

    path = img_names[i]
    image_id = path.split("\\")[2][:-4]

    target = train_df.loc[train_df['image'] == image_id, 'target'].tolist()[0]

    img = Image.open(path)

    ax[x, y].imshow(img)
    ax[x, y].axis('off')
    ax[x, y].set_title(f'ID: {image_id}, Target: {target}')

fig_8.suptitle("Training set samples", fontsize=15)
plt.show()
plt.savefig('images/Training set samples.jpg')

# Test set
img_names = glob('Project_Datasets\ISIC_2019_Test_Input\*.jpg')

fig_9, ax = plt.subplots(4, 4, figsize=(20, 20))

for i in range(16):
    x = i // 4
    y = i % 4

    path = img_names[i]
    image_id = path.split("\\")[2][:-4]

    img = Image.open(path)

    ax[x, y].imshow(img)
    ax[x, y].axis('off')
    ax[x, y].set_title(f'ID: {image_id}')

fig_9.suptitle("Test set samples", fontsize=15)
plt.show()
plt.savefig('images/Test set samples.jpg')

################################################################
##                      Starter Model                         ##
################################################################

# ignore warnings
import warnings  # tf needs to learn to stfu

warnings.simplefilter(action="ignore", category=FutureWarning)

# important libraries
import itertools
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# keras tensorflow sklearn
import tensorflow as tf
from keras import Input
from keras import layers, regularizers
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adamax
from sklearn.metrics import classification_report, confusion_matrix

# To see the value of multiple statements at once.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

##########################################################

DATA_ROOT_PATH = "Project_Datasets/"

lesion_type_dict = {
    'NV': 'Melanocytic nevi (nv)',
    'MEL': 'Melanoma (mel)',
    'AK': 'Actinic keratosis (ak)',
    'BKL': 'Benign keratosis-like lesions (bkl)',
    'BCC': 'Basal cell carcinoma (bcc)',
    'VASC': 'Vascular lesions (vasc)',
    'SCC': 'Squamous cell carcinoma (SCC) ',
    'DF': 'Dermatofibroma (df)',

}
## Label Encoding ##
label_mapping = {
    0: 'NV',
    1: 'MEL',
    2: 'AK',
    3: 'BKL',
    4: 'BCC',
    5: 'VASC',
    6: 'SCC',
    7: 'DF'
}
reverse_label_mapping = dict((value, key) for key, value in label_mapping.items())

training_df = train_df.drop(columns=['attribution', 'copyright_license', 'clin_size_long_diam_mm', 'dermoscopic_type',
                                     'diagnosis_confirm_type', 'family_hx_mm', 'image_type', 'personal_hx_mm',
                                     'benign_malignant',
                                     'melanocytic', 'nevus_type'])

training_df['dx_type'] = training_df['diagnosis'].map(lesion_type_dict.get)

training_df['label'] = training_df['dx_type'].map(reverse_label_mapping.get)

training_df['image_path'] = glob('Project_Datasets//ISIC_2019_Training_Input//*.jpg')


################################################################
##                    Train - Test Splits                     ##
################################################################
## Loading Datasets #####

# for training data
train_file_dir = 'Project_Datasets/isic-2019-skin-lesion-images-for-classification'
image_size = (180, 180)
train_ds = image_dataset_from_directory(
    train_file_dir,
    validation_split=0.2,
    subset='training',
    seed=133,
    shuffle=True,
    batch_size=None,
    image_size=image_size,
)

# for testing data
image_size = (180, 180)
val_ds = image_dataset_from_directory(
    train_file_dir,
    validation_split=0.2,
    subset='validation',
    seed=133,
    shuffle=True,
    batch_size=None,
    image_size=image_size,
)

# convert data to numpy array #

X_train = []
y_train = []

for i, j in train_ds:  # i= images, j=labels of images
    X_train.append(i)
    y_train.append(int(j))

X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)

# for test data
X_test = []
y_test = []
for i, j in val_ds:  # i= images , j=labels of images
    X_test.append(i)
    y_test.append(int(j))
X_test = np.array(X_test)
y_test = np.array(y_test)

print("the shape of test data=", X_test.shape)

####Saving Features and Label dataset values

np.save('Cleaned_Data/X_train_v03', X_train)
np.save('Cleaned_Data/y_train_v03', y_train)
np.save('Cleaned_Data/X_test_v03', X_test)
np.save('Cleaned_Data/y_test_v03', y_test)

################################################################
##                 Model Building (Squential)                ##
################################################################

def create_sequential_model(model):
    model = model
    model.add(Input(shape=[180, 180, 3]))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())

    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(units=256, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(units=32, activation='relu', kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.L1L2()))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(units=8, activation='softmax', kernel_initializer='glorot_uniform', name='classifier'))

    optimizer = Adamax(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.build((None, 180, 180, 3))
    print(model.summary())
    return model

# def plot_model_training_curve(his):
#      fig = make_subplots(rows=1, cols=2, subplot_titles=['Model Accuracy', 'Model Loss'])
#      fig.add_trace(
#          go.Scatter(
#              y=his['accuracy'],
#              name='train_acc'
#          ),
#          row=1, col=1
#      )
#      fig.add_trace(
#          go.Scatter(
#              y=his['val_accuracy'],
#              name='val_acc'
#          ),
#          row=1, col=1
#      )
#      fig.add_trace(
#          go.Scatter(
#              y=his['loss'],
#              name='train_loss'
#          ),
#          row=1, col=2
#      )
#      fig.add_trace(
#          go.Scatter(
#              y=his['val_loss'],
#              name='val_loss'
#          ),
#          row=1, col=2
#      )
#      fig.show(renderer="jpg")
#      fig.write_image("images/train_history_v03_plotly.jpg")

def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()

def test_model(model, X_test, Y_test):
    model_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
    print("Test Accuracy: {:.3f}%".format(model_acc * 100))
    y_true = np.array(Y_test)
    y_pred = model.predict(X_test)
    y_pred = np.array(list(map(lambda x: np.argmax(x), y_pred)))
    clr = classification_report(y_true, y_pred, target_names=label_mapping.values(), output_dict= True)
    print(clr)
    pd.DataFrame(clr).transpose().to_csv('model/Classification_Report_v03.csv', index = True)

    sample_data = X_test[:15]
    plt.figure(figsize=(20, 20))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow((sample_data[i]*255).astype(np.uint8))
        plt.title(label_mapping[y_true[i][0]] + '|' + label_mapping[y_pred[i]])
        plt.axis("off")
    plt.show()
    plt.savefig('images/Comparison of Observation vs Model Prediction_v03.jpg')

################################################################
##                       Setting Annealing                    ##
################################################################
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    mode='auto'
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=0.00001,
    mode='auto'
)

callback = tf.keras.callbacks.ModelCheckpoint(filepath='model/best_model.keras',
                                              monitor='val_acc', mode='max',
                                              save_best_only=True, verbose=1)

################################################################
##                      Training Model                     ##
################################################################

from datetime import datetime

start_time = datetime.now()

m = Sequential()
model = create_sequential_model(m)
history = model.fit(X_train,
                    y_train,
                    validation_split=0.2,
                    batch_size=128,
                    epochs=100,
                    callbacks=[callback, reduce_lr, early_stop])

end_time = datetime.now()
model.save('model/EDA_CNN_Trained_Model_v03.h5')
print('Duration: {}'.format(end_time - start_time))

################################################################
##                      Training History                      ##
################################################################

plot_training(history)
plt.savefig('images/Fig_Training_History_v03.jpg')

# plot_model_training_curve(history)

################################################################
##                      Model Evaluation                      ##
################################################################
train_score = model.evaluate(X_train, y_train, verbose=1)
print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)

test_model(model, X_test, y_test)

################################################################
##                      Confusion Matrix                      ##
################################################################
## Classes Label - creation

classes_labels = []
for key in label_mapping.keys():
    classes_labels.append(key)

print(classes_labels)

## Confusion Matrix
y_true = np.array(y_test)
y_pred = model.predict(X_test)
y_pred = np.array(list(map(lambda x: np.argmax(x), y_pred)))

cm = confusion_matrix(y_true, y_pred, labels=classes_labels)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=matplotlib.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(label_mapping))
plt.xticks(tick_marks, label_mapping, rotation=45)
plt.yticks(tick_marks, label_mapping)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.show()
plt.savefig('images/Confusion Matrix_v03.jpg')

################################################################
##                  Model Save and Conversion                 ##
################################################################
# Save the model

import pickle

training_history = {
    'loss': history.history['loss'],
    'accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy']
}
with open('model/training_history_v03.pkl', 'wb') as file:
    pickle.dump(training_history, file)
model.save('model/EDA_CNN_Trained_Model_v03.h5')
model.save('model/EDA_CNN_Trained_Model_v03.keras')

print("Training history and model saved successfully.")

################################################################
##          Comparing results with external sources           ##
################################################################

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np, requests
from io import BytesIO

## from images folders:

image = PIL.Image.open('Project_Datasets/')
image = image.resize((180, 180))
img = np.array(image)

plt.imshow(img)
plt.axis('off')
plt.show()
img = X_test[1]
img = np.array(image).reshape(-1, 180, 180, 3)
result = model.predict(img)
print(result[0])
result = result.tolist()
max_prob = max(result[0])
class_ind = result[0].index(max_prob)
print(label_mapping[class_ind])

### From Websites:

image_url = "https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/11/15/17/38/ds00190_-ds00439_im01723_r7_skincthu_jpg.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image = image.resize((180, 180))
img = np.array(image)
plt.imshow(img)
plt.axis('off')
plt.show()
img = X_test[1]
img = np.array(image).reshape(-1, 180, 180, 3)
result = model.predict(img)
print(result[0])
result = result.tolist()
max_prob = max(result[0])
class_ind = result[0].index(max_prob)
print(lesion_type_dict[label_mapping[class_ind]])

image_url = "https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/11/15/17/38/ds00190_-ds00439_im01723_r7_skincthu_jpg.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
image = image.resize((180, 180))
img = np.array(image)
plt.imshow(img)
plt.axis('off')
plt.show()
img = X_test[1]
img = np.array(image).reshape(-1, 180, 180, 3)
result = model.predict(img)
print(result[0])
result = result.tolist()
max_prob = max(result[0])
class_ind = result[0].index(max_prob)
print(lesion_type_dict[label_mapping[class_ind]])

################################################################
##             Loading Model File and History                 ##
################################################################

from keras.models import load_model
model=load_model('model/EDA_CNN_Trained_Model_v03.h5')

import pickle
with open('model/training_history_v03.pkl', 'rb') as f:
    history = pickle.load(f)