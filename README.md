# Project4_Gr12
# Skin Cancer Classification using Deep Learning

# Introduction and Background
Skin cancer, out-of-control growth of abnormal cells in the *epidermis*, the outermost skin layer, is one of the most wide-spread cancer found in global population. These mutations lead the skin cells to multiply rapidly and <b>form malignant tumors. The main types of skin cancer are basal cell carcinoma (BCC), squamous cell carcinoma (SCC), *melanoma* and Merkel cell carcinoma (MCC). Among all the skin cancer type, melanoma is the least common skin cancer, but it is responsible for **75%** of death [SIIM-ISIC Melanoma Classification, 2020](https://www.kaggle.com/c/siim-isic-melanoma-classification).
 
<img src="https://www.cancer.gov/sites/g/files/xnrzdm211/files/styles/cgov_article/public/cgov_contextual_image/2019-06/1-how-does-cancer-form.jpg?h=b48714fe&itok=fc2eMUvr" alt="Cancer cell development" width=400/>
    
## Melanoma, in 4 questions ##

<img src="https://www.verywellhealth.com/thmb/GmfzGuCPokTF14Dk06EaWvvROzo=/3125x2084/filters:no_upscale():max_bytes(150000):strip_icc()/what-is-melanoma-514215_final-01-3b091d9a68074ba7b5a1cb6d8287cf92.png" width=500 alt="4 types of melanoma"/><br><br>
    
What is it? *Melanoma is a cancer that develops from melanocytes, the skin cells that produce melanin pigment, which gives skin its color.*

Where is it usually found? *Melanomas often resemble moles and sometimes may arise from them. They can be found on any area of the body, even in areas that are not typically exposed to the sun.*

Nowadays, melanoma has emerged as a prevalent affliction in Australia. The incidence of this disease has been on the rise in recent years, posing a significant threat to public health. 
The number of new cases of melanoma of the skin diagnosed increased from 3,538 (1,736 males and 1,802 females) in 1982 to 15,628 in 2019 and 1,455 people died in 2021. Over the same period, the age-standardised incidence rate increased from 27 cases per 100,000 persons (28 for males and 26 for females) in 1982 to 54 cases per 100,000 in 2019.\
For individuals who seek to gain further insight into the melanoma situation in Australia, recommend consulting the following dashboard. The dashboard provides a wealth of information that can help to gain a better understanding of the current state of melanoma in the region.\
[Tableau Dashboard](https://public.tableau.com/app/profile/lakna.premasinghe/viz/MelanomaAnalysis_17032704663840/melanomainaustralia)

    
# Objective of the Project
The overarching goal is to support the efforts to reduce the death caused by skin cancer. This project aims to develop a predictive using **Convolutional Neural Network (CNN)** to classify nine types of skin cancer from outlier lesions images. The improved accuracy and efficiency of the model can aid to detect melanoma in the early stages and can help to reduce unnecessary deaths

### Diagnosis:

![image](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/37cac8f8-6be6-448e-b8fe-8e00b9f6f0a7)

# Dataset
The [Project Dataset](https://github.com/pkrachakonda/Project4_Gr12/tree/main/Project_Datasets) is openly available on
- Kaggle [ISIC-2019](https://www.kaggle.com/code/bhanuprasanna/skin-cancer-detection-isic-2019/input)
- ISIC 2019 [Test Datasets](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Test_Input.zip)
- ISIC 2019 [Training Datasets](https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip)
- Kaggle [ISIC-2019-Skin-Lesion-Images-For-Classification](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification/data)
- AIHW [Cancer data in Australia](https://www.aihw.gov.au/reports/cancer/cancer-data-in-australia/data)
- WHO [WHO Mortality Database](https://platform.who.int/mortality/themes/theme-details/topics/indicator-groups/indicator-group-details/MDB/melanoma-and-other-skin-cancers)
- GCO [Cancer Today](https://gco.iarc.fr/today/online-analysis-map?v=2020&mode=population&mode_population=continents&population=900&populations=900&key=asr&sex=0&cancer=39&type=0&statistic=5&prevalence=0&population_group=0&ages_group%5B%5D=0&ages_group%5B%5D=17&nb_items=10&group_cancer=1&include_nmsc=0&include_nmsc_other=0&projection=natural-earth&color_palette=default&map_scale=quantile&map_nb_colors=5&continent=0&show_ranking=0&rotate=%255B10%252C0%255D)

# Flowchart

![image](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/e7187979-e98b-4218-941f-ff810253d311)

# Technologies Used

 -[Tensorflow](https://www.tensorflow.org/)

 -[Keras3](https://keras.io/keras_3/)

 -[Scikit-learn](https://scikit-learn.org/stable/)

 -[Pandas](https://pandas.pydata.org/)

 -[Numpy](https://numpy.org/)

 -[Seaborn](https://seaborn.pydata.org/)

 -[Tableau Public](https://www.tableau.com/products/public)


# Model Setup
CNNs have a unique layered architecture consisting of convolutional, pooling, and fully connected layers, which are designed to automatically learn the features and hierarchies of the input data, while Ohter ML algorithms have different architecture.

![image](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/f76ae654-a550-4d0d-b770-50d78d3742e3)
[Source](https://miro.medium.com/v2/resize:fit:4800/format:webp/0*LeK_gmCf3DfO3gj_.jpeg)

The following architecture is used in our model. Four different *Feature and Label* datasets were used to test the performance of the model. In Model Version 04, *RandomOverSampler* is implemented which has increased the input size by approximately four times.

![image](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/a70fa8a8-8e52-4013-be39-679e19ef1843)

# Results

## Accuracy

![image](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/3a5b33d8-c3ee-461b-af35-3f27058589be)


## Training History

![image](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/90d2de4e-1476-4f3c-9285-dfc443cb44a8)

## Confusion Matrix

For Model V04
![Confusion Matrix_v04](https://github.com/pkrachakonda/Project4_Gr12/assets/20739237/b2b9c1f1-57d4-4104-96bf-fc3db9b05759)

Based on the analysis, **Model V04** performance is better in comparison with other Models. 

# Limitations
Even though **Model V04** performance is better in comparison with other models, performance of model in identification of labels for datasets, obtained from external sources in failing. One the reason for failure could be the pixilated images datasets used for training the model. All model require further training and validations on different types of datasets, such as datasets from the *Cancer Imaging Archive*.

Further work
As part of this project, one CNN architecture, *Sequential Model* is implemented. Other model CNN architectures, such as *Efficientnetb1*, *fasiai*, *K-fold methods*. PyTorch libraries could also be used for accelerated processing.
