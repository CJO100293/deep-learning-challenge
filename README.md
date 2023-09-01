# deep-learning-challenge
## **Background:**
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as: 
- **EIN** and **NAME**—Identification columns
- **APPLICATION_TYPE**—Alphabet Soup application type
- **AFFILIATION**—Affiliated sector of industry
- **CLASSIFICATION**—Government organization classification
- **USE_CASE**—Use case for funding
- **ORGANIZATION**—Organization type
- **STATUS**—Active status
- **INCOME_AMT**—Income classification
- **SPECIAL_CONSIDERATIONS**—Special considerations for application
- **ASK_AMT**—Funding amount requested
- **IS_SUCCESSFUL**—Was the money used effectively

## **Instructions**
### **Step 1: Preprocess the Data**
1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
	- What variable(s) are the target(s) for your model?
	- What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.
3. Determine the number of unique values for each column.
4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6. Use pd.get_dummies() to encode categorical variables.
7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.  

### **Step 2: Compile, Train, and Evaluate the Model**
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3. Create the first hidden layer and choose an appropriate activation function.
4. If necessary, add a second hidden layer with an appropriate activation function.
5. Create an output layer with an appropriate activation function.
6. Check the structure of the model.
7. Compile and train the model.
8. Create a callback that saves the model's weights every five epochs.
9. Evaluate the model using the test data to determine the loss and accuracy.
10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.  

### **Step 3: Optimize the Model**
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%. Use any or all of the following methods to optimize your model: 
- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
	- Dropping more or fewer columns.
	- Creating more bins for rare occurrences in columns.
	- Increasing or decreasing the number of values for each bin.
	- Add more neurons to a hidden layer.
	- Add more hidden layers.
	- Use different activation functions for the hidden layers.
	- Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.
1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.  

### **Step 4: Write a Report on the Neural Network Model**
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup. The report should contain the following:
1. **Overview** of the analysis: Explain the purpose of this analysis.
2. **Results:** Using bulleted lists and images to support your answers, address the following questions:
- Data Preprocessing
	- What variable(s) are the target(s) for your model?
	- What variable(s) are the features for your model?
	- What variable(s) should be removed from the input data because they are neither targets nor features?
- Compiling, Training, and Evaluating the Model
	- How many neurons, layers, and activation functions did you select for your neural network model, and why?
	- Were you able to achieve the target model performance?
	- What steps did you take in your attempts to increase model performance?
3. **Summary:** Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.  

### **Step 5: Copy Files Into Your Repository**
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.
1. Download your Colab notebooks to your computer.
2. Move them into your Deep Learning Challenge directory in your local repository.
3. Push the added files to GitHub.  \

### **Summary**
- ANSWER: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.  

## **Files:**
- **README.md** - This readme file.  
- **Report.rtf** - This file contains our report from "Step 4: Write a Report on the Neural Network Model".  
- **AlphabetSoupCharity_Optimization.ipynb** - This is the jupyter notebook used to attempt to use deep learning and neural networks to evaluate whether applicants would be funded by the non-profit organization Alphabet Soup. This is doing so while dropping the "EIN" and "NAME" columns from the model. This particular model was only able to achieve a maximum target predictive accuracy of 74.01% across 6 attempts using various combinations of the number of hidden layers, neurons and activation functions.  
- **AlphabetSoupCharity_Optimization_final.ipynb** - This jupyter notebook was used to attempt to use deep learning and neural networks to evaluate whether applicants would be funded by the non-profit organization Alphabet Soup after only dropping the "EIN" column in the model but keeping the "NAME" column. This particular model was able to achieve the target predictive accuracy of above 75% and with a final accuracy of 79.94%.  
- **output_data/AlphabetSoupCharity_Attempt_1.h5** - exported HDF5 file of results from attempt 1 of training the model in the "AlphabetSoupCharity_Optimization.ipynb" jupyter notebook.  
- **output_data/AlphabetSoupCharity_Attempt_2.h5** - exported HDF5 file of results from attempt 2 of training the model in the "AlphabetSoupCharity_Optimization.ipynb" jupyter notebook after removing the "EIN" and "NAME" columns from the model.  
- **output_data/AlphabetSoupCharity_Attempt_3.h5** - exported HDF5 file of results from attempt 3 of training the model in the "AlphabetSoupCharity_Optimization.ipynb" jupyter notebook after removing the "EIN" and "NAME" columns from the model.  
- **output_data/AlphabetSoupCharity_Attempt_4.h5** - exported HDF5 file of results from attempt 4 of training the model in the "AlphabetSoupCharity_Optimization.ipynb" jupyter notebook after removing the "EIN" and "NAME" columns from the model.  
- **output_data/AlphabetSoupCharity_Attempt_5.h5** - exported HDF5 file of results from attempt 5 of training the model in the "AlphabetSoupCharity_Optimization.ipynb" jupyter notebook after removing the "EIN" and "NAME" columns from the model.  
- **output_data/AlphabetSoupCharity_Attempt_6.h5** - exported HDF5 file of results from attempt 6 of training the model in the "AlphabetSoupCharity_Optimization.ipynb" jupyter notebook after removing the "EIN" and "NAME" columns from the model.  
- **output_data/AlphabetSoupCharity_Final.h5** - exported HDF5 file of results of training the model in the "AlphabetSoupCharity_Optimization-Final.ipynb" jupyter notebook after removing the "EIN" column from the model but keeping the "NAME" column. These final results achieved the target predictive accuracy of above 75%.  

## **Sources:**
- The basis for the code used in the "Drop the non-beneficial ID columns, 'EIN' and 'NAME'" section of both jupyter notebooks was found from https://blog.hubspot.com/website/drop-multiple-columns-pandas
- The basis for the code used in the "Choose a cutoff value and create a list of application types to be replaced. use the variable name application_types_to_replace" section of both jupyter notebooks was found from https://github.com/ItsGreyedOut/Deep-Learning-Charity-Funding-Predictor/blob/main/Starter_Code.ipynb
- The basis for the code used in the "Convert categorical data to numeric with pd.get_dummies" section of both jupyter notebooks was found from https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html and https://stackoverflow.com/questions/55333191/does-get-dummies-function-change-the-dtype-of-a-column.
- The basis for the code used in the "Split our preprocessed data into our features and target arrays" section of both jupyter notebooks came from "https://github.com/ItsGreyedOut/Deep-Learning-Charity-Funding-Predictor/blob/main/Starter_Code.ipynb".
- The basis for the code used in the "Export our model to HDF5 file" section came from "https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format.
- per the suggestion of https://github.com/ItsGreyedOut/Deep-Learning-Charity-Funding-Predictor/blob/main/Analysis_Report_Neural_Network_Model.pdf i added back the "NAME" column to try and improve accuracy in the "AlphabetSoupCharity_Optimization_Final.ipynb" file.
- The basis for the code used in the "Look at NAME counts for binning" and "Choose a cutoff value and create a list of application types to be replaced, use the variable name application_types_to_replace" sections of "AlphabetSoupCharity_Optimization_Final.ipynb" was found from https://github.com/ItsGreyedOut/Deep-Learning-Charity-Funding-Predictor/blob/main/AlphabetSoupCharity_Optimzation.ipynb.
- The basis for the code used in the "Importing Dependencies needed for "softmax activation function" section", "softmax activation function", "linear activation function" and "tanh activation function" sections of "AlphabetSoupCharity_Optimization.ipynb" was found from https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/