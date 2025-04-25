# 1. Introduction

Traffic accidents are a major public health and safety concern in Turkey. This project aims to investigate the potential relationship between education levels and traffic collision rates in different regions of Turkey. By analyzing data on educational levels and traffic accidents (deadly or not), we can identify correlations between education levels and traffic accidents, finally putting an end to the much-repeated question: "Are bad drivers uneducated?" Istanbul is most likely to be excluded from the dataset due to its heavy immigration from other cities, but it will be considered separately.

# 2. Objectives

- **a)** To determine if there is any relationship between driving responsibly and education.  
- **b)** To analyze any regional variations between the 7 districts of Turkey.  
- **c)** To determine if the quality of infrastructure affects traffic-related accidents.  

# 3. Methods

- **a)** Traffic data will be extracted from TÜİK.  
- **b)** Educational data will be extracted from MEB, but TÜİK has similar data. In that case, newer or cleaner data will be used.  
- **c)** The quality of traffic infrastructure and the age of cars are factors as well. To determine the factor of car age, we will use how many cars are sold per thousand, and for traffic infrastructure, we will use how much money has been spent on infrastructure over the past * years to account for and correct these factors as well.

- ## Tools and Technologies

I’ll rely on the following tools for data analysis and visualization:

- **Python**: For data cleaning and statistical analysis  
- **Pandas**: To manipulate and preprocess data  
- **Matplotlib and Seaborn**: For creating visualizations (scatter plots, heatmaps, time series)  
- **SciPy**: For hypothesis testing and regression analysis    

# Analysis Plan

1. **Data Collection**  
   - TUIK provides downloadable csv files from the website, i will download the required data using the webportal

2. **Visualization**  
   - Use scatter plots,   
   - Examples include:    

3. **Hypothesis Testing**  
   - Test hypotheses like:  
     - **H₀**: Daily habits have no effect on Bench Press performance.  
     - **Hₐ**: One or more daily variables significantly impact Bench Press performance.  
   - Run regression analysis to identify the strongest predictors of progress.

4. **Trend Analysis**  
   - Investigate patterns in performance over time, identifying peaks or plateaus.  
   - Analyze how body weight fluctuations and day-to-day difficulty ratings correlate with performance trends.

