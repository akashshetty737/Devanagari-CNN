# Devanagari-CNN
CNN model for recognition of devanagari.
Shuffling.py can be used to shuffle the dataset.
https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset for the dataset.
each class should have 2000 images which will be split by the shuffling.py code into training, test and validation sets in the ratio of 6:2:2.
Transfer Learning is done by freezing the 5 convloutional layers and only training the fully connected layers (i.e. the dense layers )
