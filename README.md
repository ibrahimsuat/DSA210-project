## Introduction

Traffic accidents are a major public health and safety concern in Turkey. This project aims to investigate the potential relationship between education levels and traffic collision rates in different regions of Turkey. By analyzing data on educational levels and traffic accidents (deadly or not), we can identify correlations between education levels and traffic accidents, finally putting an end to the much-repeated question: "Are bad drivers uneducated?" Istanbul is most likely to be excluded from the dataset due to its heavy immigration from other cities, but it will be considered separately.

## Motivation
Curosity is main driver for this project, Personally i heard many people call other people uneducated for various reasons but i heard the phrase most in traffic and i want to end this conversation for good.

## Objectives

- **a)** To determine if there is any relationship between driving responsibly and education.  
- **b)** Try to determine the real cause of traffic accidents   
- **c)** Use machine learning to make predictions

## Methods

- **a)** Traffic data will be extracted from TÜİK.  
- **b)** Educational data will be extracted from MEB, but TÜİK has similar data. In that case, newer or cleaner data will be used.  
- **c)** The quality of traffic infrastructure and the age of cars are factors as well. To determine the factor of car age, we will use how many cars are sold per thousand, and for traffic infrastructure, we will use how much money has been spent on infrastructure over the past * years to account for and correct these factors as well.
- **d)** Machine learning methods will be used to determine the real cause of traffic accidents(if not education).
- ## Tools and Technologies

I’ll rely on the following tools for data analysis and visualization:

- **Python**: For data cleaning and statistical analysis.  
- **Pandas**: To manipulate and preprocess data.
- **Matplotlib and Seaborn**: For creating visualizations.  
- **Scipy**: For hypothesis testing.
- **Regex**: For formatting and removing unwanted values.
- **sklearn**: For machine learning applications

# Analysis Plan

1. **Data Collection**  
   - TUIK provides downloadable csv files from the website, i will download the required data using the webportal

2. **Visualization**  
   - Use scatter plots, barcharts to clearly see diffrence between education levels and traffic accidents among cities    

3. **Hypothesis Testing**  
   - Test hypotheses like:  
     - **H₀**: Amount of education does not impact traffic. 
     - **Hₐ**: Amount of edcuation does impact traffic and leads to more accidents 

## To reproduce this analysis

   -simply download this repository and run the python code with requirements installed beforehand.
