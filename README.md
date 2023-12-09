# sketches-classification

Ce projet a été réalisé dans le cadre du cours **Reconnaissance d'objets et vision artificielle (RecVis)** - Master M2 MVA
Lecturers: Gül Varol(), Jean Ponce, Armand Joulin, Josef Sivic, Ivan Laptev, Cordelia Schmid, and Mathieu Aubry

**Kaggle (rank 4/59)** : https://www.kaggle.com/competitions/mva-recvis-2023/leaderboard

The aim of this work is to develop a model capable of classifying the images of the dataset classifysketch with the best accuracy. It is made up of 250 classes of sketches. We will begin by examining the dataset, then discuss the model selection, data augmentation and model tuning that enabled me to achieve 82.8% accuracy on the test dataset using results from [1] and [2] and a new data augmentation

* Implementation of Cut-mix, Mix-up and Line width augmentation :
* 
![data_augmentation](https://github.com/b-ptiste/sketches-classification/assets/75781257/3a9e659f-5965-4d60-b188-dfb1d8603466)

# Credit
Teacher assistant : Ricardo Garcia Pinel
[1] Alexander Kolesnikov Dirk Weissenborn Xiaohua Zhai Thomas Unterthiner Mostafa Dehghani Matthias Minderer Georg Heigold Sylvain Gelly Jakob Uszkoreit Neil Houlsby Alexey Dosovitskiy, Lucas Beyer. An image is worth 16x16 words: Transformers for image recognition at scale, 2020. Google Research, Brain Team.
[2] Xiaohua Zhai Ross Wightman Jakob Uszkoreit Lucas Beyer Andreas Steiner, Alexander Kolesnikov. How to train your vit? data, augmentation, and regularization in vision trans- formers, 2021. Google Research, Brain Team.
