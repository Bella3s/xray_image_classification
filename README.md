# xray_image_classification

- image here

## Abstract

### Links

- Repo tree here with links to every other part of the project

## Pneumonia, UNICEF, and Nerual Labs Africa

Pneumonia is the cause of death for over 725,000 children under the age of 5 worldwide, of which around 190,000 are newborns ([World Health Organization](https://www.who.int/news-room/fact-sheets/detail/pneumonia) and [UNICEF](https://www.unicef.org/stories/childhood-pneumonia-explained)). The most common type of pneumonia infection is bacterial, thus treatment is often as easy as taking a round of antibiotics. 

One of the most common ways to diagnosis if a patient has contracted pneumonia is to examine a chest x-ray, looking for white spots in the lungs (called infiltrates) ([RadiologyInfo.org](https://www.radiologyinfo.org/en/info/pneumonia#:~:text=When%20interpreting%20the%20x%2Dray,(fluid%20surrounding%20the%20lungs).)).

A vast majority of deaths caused by pneumonia are concentrated in the world's poorest countries (southern Asia and sub-Saharan Africa), where there is a lack of robust health care systems. This includes a shortage of doctors and lack of access to x-rays and labs. This lack of a health care system is one of the main reasons that pneumonia kills so many children even when the treatment is known and easy to administer. 

One of the ways that UNICEF is actively fighting the number of deaths by pneumonia is by investing in companies like [Neural Labs Africa](https://neurallabs.africa/#) through their [Venture Fund](https://www.unicef.org/innovation/venturefund/ai-ds-learning-health-cohort). Neural Labs Africa is working to leverage AI to "democratize access to diagnostic healthcare" and improve patient care. Their product NeuralSight for Chest Imaging is capable of identifying, labeling and highlighting over 20 respiratory diseases, including pneumonia. 

While what Neural Labs Africa is accomplishing is widely outside the scope of this project, it does serve as the inspiration.  The idea of expanding health care to areas that most need it through machine learning by offering real time diagnoses from a simple x-ray image. 

## The Business + Business Problem

For this project, we will posit a non-for-profit organization looking to expand access of health care as the business.  The project is to create a model that will predict whether or not a patient has pneumonia based on their x-ray.  The idea is that patients in rural areas, or areas with limited access to hospitals and doctors (but have access to a mobile x-ray machine more locally) can utilize this to achieve real-time diagnoses, and those patients that have pneumonia can make next steps to get care.    

It is important that we capture as many patients as possible with pneumonia to prevent deaths.  However, placing an undue burden on patients who are not sick must also be avoided whenever possible.  Keeping in mind that we are targeting people who do not have health care resources readily available, the assumption is that taking steps to get care will be quite arduous.  Thus, the goal is to achieve both accurate and precise predictions.  


## The Data Source

The data for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).  The chest x-rays themselves were “selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzou.”  The radiographs were screened for quality control. 

**If reproducing this project from GitHub:**   

The original data source had a different percentage of images in the train and validate folders.  Some prep work has already been done to move a percentage of pictures from the train set into the validation set to create more of an 80/10/10 split between the train/validation/test sets.   

**If reproducing this project via Google CoLab:** 

Please see directions on how to download and prepare the data on the document labeled [google_colab_setup](https://github.com/Bella3s/xray_image_classification/blob/main/google_colab_setup.ipynb).  This document goes through the directions and cells to download the Kaggle data and move a percentage of the train images into the validation set to achieve an approximate 80/10/10 split between the train/validation/test sets. 

## Data Exploration

- visual of first 8 x-rays
- visual of imbalanced data

## Model Iteration

- Baseline Model -- Very Small Neural Network
    - Baseline with more data
- Neural Network
- Convolutional Neural Network
- Convolutional Neural Network with Dropout + Regularlization
- Transfer Learning Model

(image of final Data Frame here)
(image of chosen final model visuals)

## Final Model Evaluation

- Visuals
    - final model metrics + confusion matrix
    - activation layer visual?

## Conclusion

- good model
- recommendations
