# X-Ray Image Classification


## Abstract

This project creates a model that predicts if a patient has pneumonia or not, using pediatric chest x-ray images.  Each year, pneumonia is the cause of death for approximately 725,000 children under the age of 5 worldwide, largely impacting communities that do not have a robust health care system (namely southern Asia and sub-Saharan Africa).  Companies like [Neuro Labs Africa](https://neurallabs.africa/#) are working in conjunction with [UNICEF](https://www.unicefventurefund.org/story/neural-labs-using-ai-accelerate-medical-imaging-diagnosis-respiratory-diseases) to help increase access to healthcare, provide real time diagnoses of pneumonia (and other diseases), and ultimately decrease the number of deaths caused by this very treatable illness.  This project proposes a business with similar goals, one aiming to expand health care access and help identify patients with pneumonia. 

This project goes through a model iteration process, starting with a very basic densely connected neural network, next trying a convolutional neural network – both with and without regularization – and ending with a model that utilizes transfer learning.  The project finds that the CNN with regularization is the most performant model, based on validation metrics, including accuracy, recall, precision and F1 score, as well as considering computational time and the weight of the model.  The project evaluates the chosen final model with a holdout test set, displays a few visuals of the final model (the structure of the model as well as the activation layers), and offers a few recommendations to the proposed business. 

### Repository Structure

```
├── chest_xray (https://github.com/Bella3s/xray_image_classification/tree/main/chest_xray)
├── images (https://github.com/Bella3s/xray_image_classification/tree/d1b37c59c035a1a0b05b24ba8345585b693f13dc/images)
├── pdfs ()
├── README.md (https://github.com/Bella3s/xray_image_classification/blob/main/README.md)
├── google_colab_setup.ipynb (https://github.com/Bella3s/xray_image_classification/blob/main/google_colab_setup.ipynb) 
├── index.ipynb (https://github.com/Bella3s/xray_image_classification/blob/main/index.ipynb)
```

## Pneumonia, UNICEF, and Nerual Labs Africa

<img src=images/bronchi_lungs.jpg>

Pneumonia is the cause of death for over 725,000 children under the age of 5 worldwide, of which around 190,000 are newborns ([World Health Organization](https://www.who.int/news-room/fact-sheets/detail/pneumonia) and [UNICEF](https://www.unicef.org/stories/childhood-pneumonia-explained)). The most common type of pneumonia infection is bacterial, thus treatment is often as easy as taking a round of antibiotics. 

One of the most common ways to diagnosis if a patient has contracted pneumonia is to examine a chest x-ray, looking for white spots in the lungs (called infiltrates) ([RadiologyInfo.org](https://www.radiologyinfo.org/en/info/pneumonia#:~:text=When%20interpreting%20the%20x%2Dray,(fluid%20surrounding%20the%20lungs).)).

<img src=images/pneumonia_map.png width=70%>

A vast majority of deaths caused by pneumonia are concentrated in the world's poorest countries (southern Asia and sub-Saharan Africa), where there is a lack of robust health care systems. This includes a shortage of doctors and lack of access to x-rays and labs. This lack of a health care system is one of the main reasons that pneumonia kills so many children even when the treatment is known and easy to administer. 

One of the ways that UNICEF is actively fighting the number of deaths by pneumonia is by investing in companies like [Neural Labs Africa](https://neurallabs.africa/#) through their [Venture Fund](https://www.unicef.org/innovation/venturefund/ai-ds-learning-health-cohort). Neural Labs Africa is working to leverage AI to "democratize access to diagnostic healthcare" and improve patient care. Their product NeuralSight for Chest Imaging is capable of identifying, labeling and highlighting over 20 respiratory diseases, including pneumonia. 

<img src=images/neuralsight_xray.png width=50%>

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

<img src=images/train_xray_imgs.png width=80%>

As mentioned above, the initial exploration of the data lead to the discovery that the original downloaded data had a split of approximately 79% in the training folder, 1% in the validation folder, and 10% in the test folder. Some work was done to achieve a more standard 80/10/10 split. 

This is a fairly large dataset, with 4,696 images in the train set, 536 in the validation, and 624 in the test, for a total of 5,8856 images

<img src=images/normal_xray_pixel_dist.png width=49% align="left">
<img src=images/pneumonia_xray_pixel_dist.png width=50% align="right">

In the above images we can see the pixel distribution for an x-ray with out pneumonia and an x-ray with pneumonia. We can see that the normal x-ray has more pixels with a 0 value (absolute black) than the x-ray with pneumonia.

<img src=images/target_dist.png width=45%>

We also found that there is quite an imbalance in the target distribution, with only 23% of the training images being in the normal class, and 77% being x-rays with pneumonia present. With image processing, we will be using Neural Networks. The idea we move forward with in this project is that these complex neural network models will be able to overcome this imbalance. However, we can always circle back and either downsample or perform data augmentation to address this imbalance if needed.


## Model Iteration

For our model iteration, we are going to start very small and work our way to more complex.  The models we are going to run through are: 

1. An extremely small, densely connected Neural Network, utilizing only 25% of the training data.  The goal is to see what kind of results we can accomplish with a lightweight Neural Network model.  This will serve as our baseline model.
    - We will also look at how this same model structure performs wit 75% of the data as well. 
2. A Convolutional Neural Network.
3. A Convolutional Neural Network with dropout and regularization. 
4. A model that utilizes transfer learning.  We will be using TensorFlow's VGG19 as the base of our final model. 

Each model is trained on the test set and evaluated with the validation set. A visual of five graphs and the confusion matrix is generated for each model as well.

<img src=images/final_model_graphs.png>

Then all of the models are compared side-by-side before a final model is chosen.  

<img src=images/model_comparison.png>

Recall and precision are both important in this senario, hense the use of the F1 score.  Our main concern is to identify as many children as possible with pneumonia to avoid unessisary deaths.  However, we must also keep in mind that the goal is to service areas that have a lack of resources. Diagnosing a healthy patient with pneumonia could be a great financial and physical burden (simply in terms of proximity) to that family -- a burden that they may not be able to afford unless their child is actually sick.  Furthermore, we would like to keep the model relatively light if possible.  Again, with working in areas with a lack of resources, a lighter-weight model would be more attractive and usable in this senario.  Taking all that into consideration, the final model we will move forward wtih is the ####.


## Final Model Evaluation

<img src=images/final_model_test_eval.png, width=49%, align="left">
<img src=images/cm_xrays.png, width=50%, align="right">

As we can see from the image above, our model performed worse on our test data than on the validation data.  While this is slightly expected, these results are not quite as performant as we would like. 

We can see from the x-ray examples of each confusion matrix category that these images are quite difficult to differentiate without a trained eye.  Let’s take a look at the model structure and one of the model’s activation layers to get a better understanding of how the model is making its predictions. 

<img src=images/model.png>

<img src=images/activation_layers.png>

We can see that, as expected, the model is abstracting the image as we go further along through the activation levels. We can also see that the model is taking into consideration the area outside of the lungs, namely on the left + right side of the image outside of the body cavity as well as the area below the ribcage. These areas have no bearing on diagnosis, and the size and quality vary from x-ray to x-ray.


## Conclusion + Recommendations

In the end, we will say this is a good model, but not as precise as we would have liked. Ultimately, the model is capturing most of the patients with pneumonia – identifying these patients with real time diagnoses means they can more quickly take next steps to receive care and avoid unnecessary deaths.  That said, there is always room for improvement!  It would be good to train these models on a more balanced test set, which can easily be achieved via downsampling.   

Based on the evaluation of our final model, we have three recommendations for the business: 

1. **Improve access to mobile x-ray machines in affected areas.**  Again, the goal of this company and project is to increase access to care, however it is completely reliant on the use of x-ray technology.  Many of the areas in the world that are most affected by pneumonia don’t have access to this technology which means these advancements in machine learning are moot point.  First and foremost, we must work to increase the access to a more robust healthcare system – if not more doctors, then at least a mobile x-ray machine so that models like the one in this project can actually be put to use. 

2. **Use this model as a screening process.** The original goal of this project was to offer real time diagnoses to patients.  However, because the ending model precision is not as high as we would like (the model is still quite often over predicting the patient has pneumonia when they are healthy), we would recommend to first implement this model as a screening process.  Patients can come in, get their x-ray, and if pneumonia is predicted, the image can be sent to a doctor to confirm even if that doctor is not physically present.  This will balance out the tendency for the model to over predict pneumonia and decrease the burden of then following through to receive care for those patients who are actually healthy. 

3. **Investigate how to standardize x-ray images further.**  As seen from the activation layers, the model is basing its ‘diagnosis’ on areas of the image that have no bearing on actual diagnosis.  If the x-ray images can be further standardized, and/or preprocessed so the model has only the most prevalent parts of the image to consider, then it will improve its performance.  


### Next Steps: 

There is always more to accomplish!  No model is perfect and there is plenty of room for improvement on this model and project (running it again with downsampling, trying out larger more complex models, looking at further hyperparameter tuning, etc.).   

Furthermore, we can expand this idea to a multiclassification problem!  Neural Labs Africa is working on a product that can identify up to 30 different diseases all from a single x-ray machine.  This is a natural progression in computer vision for this type of project, and would be much more helpful for expanding access to health care. 
