# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neural_network import MLPClassifier




# Set random seed for reproducibility
torch.manual_seed(1234)

# Define dataset path and levels
path = "/content/drive/MyDrive/COVID-19_Radiography_Dataset"
levels = ['Normal', 'COVID']

# Load image paths and labels
data = []
for id, level in enumerate(levels):
    for file in os.listdir(os.path.join(path, level, 'images')):
        data.append([os.path.join(path, level, 'images', file), level])
data = pd.DataFrame(data, columns=['image_file', 'result'])

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load and preprocess images
images = []
labels = []
for idx, row in data.iterrows():
    image = Image.open(row['image_file'])
    image = transform(image)
    images.append(image)
    labels.append(row['result'])
images = torch.stack(images)
labels = torch.tensor([1 if label == 'COVID' else 0 for label in labels])


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



# Define class labels
classes = ['Normal', 'COVID']

# Define function to show image tensor
def show_image(img_tensor, label):
    plt.imshow(torchvision.transforms.ToPILImage()(img_tensor))
    plt.title(classes[label])
    plt.axis('off')
    plt.show()

# Show a few random images from the training set
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i in range(2):
    for j in range(4):
        idx = torch.randint(len(X_train), size=(1,)).item()
        axs[i, j].imshow(torchvision.transforms.ToPILImage()(X_train[idx]))
        axs[i, j].set_title(classes[y_train[idx]])
        axs[i, j].axis('off')
plt.show()

# Plot class distribution
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(classes, [sum(y_train == i) for i in range(2)])
ax.set_xlabel('Class')
ax.set_ylabel('Count')
ax.set_title('Class Distribution')
plt.show()



# Reshape image data into a 2D matrix
X = images.reshape(images.shape[0], -1)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# Plot PCA visualization
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(2):
    idx = np.where(labels == i)
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label=classes[i])
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_title('PCA Visualization of COVID-19 Dataset')
ax.legend()
plt.show()



# Reshape image data into a 2D matrix

# Perform PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Create plotly figure
fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], color=labels,
                    color_discrete_sequence=['blue', 'red'], opacity=0.7, title='3D Visualization of COVID-19 Dataset')

# Show plotly figure
fig.show()





# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels, random_state=42)

# Train an SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'COVID'])

# Print results
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Classification Report:\n', report)


# Train a random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'COVID'])

# Print results
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Classification Report:\n', report)



# Train a Gaussian Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'COVID'])

# Print results
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Classification Report:\n', report)



# Train an XGBoost model
params = {'max_depth': 5, 'learning_rate': 0.1, 'objective': 'binary:logistic', 'n_estimators': 100}
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'COVID'])

# Print results
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Classification Report:\n', report)



# Train an MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, solver='adam', random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mlp_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'COVID'])

# Print results
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Classification Report:\n', report)