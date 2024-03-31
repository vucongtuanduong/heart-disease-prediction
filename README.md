# heart-disease-prediction

## Introduction
Here are my files to deploy the simple heart disease classification app on HuggingFace.\
My source code is in this repository: https://github.com/vucongtuanduong/heart-disease-22-classification

In my model:
- Using logistic regression model
- No feature engineering technique
- Using label encoder and one hot encoding for categorical variables
- Using Gradio to deploy the model on HuggingFace

## Dataset that I used:
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

## Huggingface
My model is being deployed on this website: https://huggingface.co/spaces/vucongtuanduong/heart-disease-prediction

## How to enter the box:
Sex: 0 for Female, 1 for Male

GeneralHealth: 0 for Excellent, 1 for Fair , 2 for Good , 3 for Poor, 4 for Very good

Last Checkup Time: 0 for 5 or more years ago, 1 for Within past 2 years (1 year but less than 2 years ago), 2 for Within past 5 years (2 years but less than 5 years ago), 3 for Within past year (anytime less than 12 months ago)

PhysicalActivities: 1 for Yes, 0 for No

RemovedTeeth: 0 for 1 to 5, 1 for 6 or more, but not all , 2 for All , 3 for None of them

HadAngina: 0 for no, 1 for Yes

HadStroke: 0 for no, 1 for yes

HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, ChestScan, HIVTesting, PneumoVaxEver,HighRiskLastYear , AlcoholDrinkers, FluVaxLast12: 0 for no, 1 for yes

HadDiabetes: 0 for no, 1 for No, pre-diabetes or borderline diabetes , 2 for Yes , 3 for Yes, but only during pregnancy (female)

SmokerStatus: 0 for Current smoker - now smokes every day, 1 for Current smoker - now smokes some days, 2 for Former smoker , 3 for Never smoked

ECigaretteUsage: 0 for Never used e-cigarettes in my entire life, 1 for Not at all (right now) , 2 for Use them every day , 3 for Use them some days

RaceEthnicityCategory: 0 for Black only, Non-Hispanic, 1 for Hispanic , 2 for Multiracial, Non-Hispanic , 3 for Other race only, Non-Hispanic, 4 for White only, Non-Hispanic

AgeCategory: 9 for "Age 65 to 69", 8 for "Age 60 to 64" , 10 for "Age 70 to 74" , 7 for" Age 55 to 59" , 6 for "Age 50 to 54" , 11 for "Age 75 to 79", 12 for "Age 80 or older" , 4 for "Age 40 to 44", 5 for Age 45 to 49, 3 for "Age 35 to 39" , 2 for  "Age 30 to 34", 0 for "Age 18 to 24", 1 for "Age 25 to 29"

TetanusLast10Tdap:0 for "No, did not receive any tetanus shot in the past 10 years", 3 for "Yes, received tetanus shot but not sure what type", 1 for "Yes, received Tdap", 2 for  "Yes, received tetanus shot, but not Tdap"

CovidPos: 0 for No, 2 for Yes , 1 for "Tested positive using home test without a health professional"
