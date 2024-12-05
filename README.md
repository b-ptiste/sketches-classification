**Kaggle (rank 4/59)** : https://www.kaggle.com/competitions/mva-recvis-2023/leaderboard

The aim of this work was to develop a model capable of classifying the images of the dataset classifysketch with the best accuracy. It is made up of 250 classes of sketches. I fine-tuned model from timm library 🤗. I improved the performance using data augmentation. It enabled me to achieve 82.8% accuracy on the test dataset using results from [[1](https://arxiv.org/abs/2010.11929)], [[2](https://arxiv.org/abs/2106.10270)], and a new data augmentation :

* Implementation of Cut-mix, Mix-up 
* New domain specific augmentation : **Line width augmentation**
![data_augmentation](https://github.com/b-ptiste/sketches-classification/assets/75781257/3a9e659f-5965-4d60-b188-dfb1d8603466)

# Credit

This project was carried out as part of the **Reconnaissance d'objets et vision artificielle (RecVis)** - Master M2 MVA
Lecturers: Gül Varol, Jean Ponce, Armand Joulin, Josef Sivic, Ivan Laptev, Cordelia Schmid, and Mathieu Aubry

Teacher assistant : Ricardo Garcia Pinel
