import numpy as np
import pandas as pd

"""
1.1 Loading Data
@params:
    datapath: the file path that stores the data, which will be stored as a pandas dataframe
    header: if the dataset contains a header, if not, use names to assign column names
    name: a list that has been manually drawn from the .name file
    index_col: whether to set a column as index, if nothing is passed, the rows are automatically indexed
@return: pandas dataframe that stores the desired data
This function assumes the data does not include header and the name is preprocessed
which will be the case for this class's assignments
"""
def load_data(datapath, header=None, names=None, index_col=None, log_columns=None):
    df = pd.read_csv(datapath, header=header, names=names, index_col=index_col)
    if log_columns:
        for col in log_columns:
            # we do not delete the original column to keep the option of using the original data in future modeling stages
            # Define a small constant (epsilon) to avoid taking the logarithm of zero
            epsilon = 1e-10
            df[col+'_log'] = np.log(df[col]+epsilon) 
    return df

"""
1.2 Handling missing values
@params:
    df: the dataframe that we are dealing with
    missing_notation: How is missing value denoted in the dataframe, help us to locate the correct missing value, defulf ?
    numeric: whether the features are numeric. If we have features mixing numeric & categorical, extended implementations and structures are needed, but we are good for now
@return: NA (modifications made in place)   
"""
def handle_missing_values(df, missing_notation='?', numeric=True):
    # iterate through the columns
    for column in df:
        column_val = df[column].to_list()
        # if there is missing value in the column
        if missing_notation in column_val:
            # get the column list without missing_notation for mean calculation
            remove_na = [i for i in column_val if i != missing_notation]
            if numeric:
                # if numeric, transform to numeric and calculate the mean
                mean = pd.to_numeric(remove_na).mean()
                column_val[:] = [x if x != '?' else mean for x in column_val]
                column_val = pd.to_numeric(column_val)
                # replace the original column in the dataframe
                df[column] = column_val
            # we only have numerical missing values among all the datasets
            # TODO: implement for categorical missing values when needed
            else:
                continue
                
"""
1.3 Task 3: Encode ordinal data as integers
@params: 
    df: the dataframe
    columns: map of the columns that need ordinal encoding to the corresponding ordering. 
@return: None (modifications made in place)
"""
def encode_ordinal(df, columns):
    # iterate through the columns need modifications
    for col in columns:
        # get column values to a list
        col_val = df[col].to_list()
        # get the corresponding order
        order = columns[col]
        for i in range(len(order)):
            if order[i] not in col_val: continue # if the ele is not in the column ignore it
            # for each element in the order, switch element with the corresponding order number
            col_val[:] = [x if x != order[i] else i for x in col_val]
        # replace the original column in the dataframe
        df[col] = col_val
            
"""
1.3 Task 4: Encode nominal
@params:
    df: the dataframe
    columns: list of the columns we are getting dummy variables about
@return: None (modifications made in place)
"""
def one_hot_encoding(df, columns):
    for col in columns:
        # get column values to a list
        col_val = df[col].to_list()
        # get unique value
        unique_val = pd.unique(col_val)
        for ele in unique_val:
            # set the column name and get the dummy value for the dummy column
            dummy_name = col + '_' + str(ele)
            dummy_val = [0 if x != ele else 1 for x in col_val]
            # add the column to the dataframe
            df[dummy_name] = dummy_val
            # if there are less than 2 categories, we only need to do the above procedure once
            if len(unique_val) <= 2:
                break
        # delete the original column
        df.drop(columns=[col], inplace=True)
    
"""
1.4 Discretization
@params: 
    df: the dataframe
    columns: the list of columns to be discreitized
    discret_type: the list of discret_types applying to each column
    bins: the list of bin sizes using for discretization
@return: None (modifications made in place)
This function assumes the inputs are in correct corresponding order
"""
def discretization(df, columns, discret_types, bins):
    for i in range(len(columns)):
        col = columns[i]
        col_val = df[col]
        discret_type = discret_types[i]
        b = bins[i]
        if discret_types[i] == 'width':
            df[col] = _discretization_by_width(col_val, b)
        elif discret_types[i] == 'freq':
            df[col] = _discretization_by_freq(col_val, b)
        else:
            # if the discretization type does not have the correct input, raise error
            raise ValueError(f"Invalid discretization type {discret_type}. Choose from: ['width', 'freq']")
        
# Helper function for discretization by width
# return the discretized list
def _discretization_by_width(col_val, b):
    df = pd.DataFrame({
        'index': range(len(col_val)),
        'val': col_val})
    df.sort_values(by=['val'], inplace=True)
    # update the value according to the bin
    for i in range(len(col_val)):
        df['val'].iloc[i] = int(i/b)
    df.sort_values(by=['index'], inplace=True)
    return df.val.to_list()
    
# Helper function for discretization by frequency
# return the discretized list
def _discretization_by_freq(col_val, b):
    sorted_val = np.sort(col_val)
    # Calculate the bin edges (quantiles)
    bin_edges = np.percentile(sorted_val, np.linspace(0, 100, b + 1))
    # Initialize an empty list to store discretized data
    discretized_data = []

    # Assign each data point to a bin
    for value in col_val:
        bin_index = 0
        while bin_index < num_bins and value > bin_edges[bin_index + 1]:
            bin_index += 1
        discretized_data.append(bin_index)

    return discretized_data
    
"""
1.5 Standardization
Perform z-score standardization on training and test data.
@params:
    train_data (numpy array): Training data to be standardized.
    test_data (numpy array): Test data to be standardized.
@returns:
    standardized_train_data (numpy array): Standardized training data.
    standardized_test_data (numpy array): Standardized test data.
"""
def z_standarization(train_data, test_data):
    # Calculate the mean and standard deviation of the training data
    mean = np.mean(train_data, axis=0)
    std_dev = np.std(train_data, axis=0)

    # Apply standardization to both training and test data
    standardized_train_data = (train_data - mean) / std_dev
    standardized_test_data = (test_data - mean) / std_dev

    return standardized_train_data, standardized_test_data

"""
1.6 Cross-Validation 
Perform kx2 cross-validation without using sklearn.model_selection.
@params:
    dataset (numpy array or list): The input dataset.
    target (numpy array or list): The target labels.
    k: number of validation rounds
    classification (bool): True if it's a classification task, False for regression.
                           Default is True.
    hyperparam: the hyperparam setting we are using for the model
@return: (best hyperparameter setting chosen, the average performance of that setting)
"""
def k_2_cv(dataset, target, k, hyperparam, model, classification=True,):

    num_splits = k
    num_inner_splits = 2
    
    # Step 1: Split data into 80% training and 20% validation
    X_train, X_valid, y_train, y_valid = train_test_split(dataset, target, 0.8, classification)
    
    print("X_train size: ", len(X_train))
    print("X_valid size: ", len(X_valid))
    print("y_train size: ", len(y_train))
    print("y_valid size: ", len(y_valid))
    
    performance = []
    # performance is stored as 
    # [1, 2, 3] - fold 1
    # [1, 2, 3] - fold 2
    # ...
    # with 3 hyperparameter setting
    for i in range(num_splits):
        
        # Step 2: Divide the 80% into two equally-sized partitions (stratified for classification)
        train_data_split_1, train_data_split_2, train_labels_split_1, train_labels_split_2 = train_test_split(X_train, y_train, 0.5, classification)
        
        # Step 2(b): Train and evaluate models with different hyperparameters        
        
        # calculate evaluation metrics and store the performance
        print(f"Round {i}.1 of hyperparameter choosing.")
        print(f"train size: {len(train_labels_split_1)}. test size: {len(y_valid)}")
        p1 = evaluate_model_with_hyperparameters(train_data_split_1, train_labels_split_1, X_valid, y_valid, hyperparam, classification, model)
        print(f"Performance of corresponding hyperparams: {p1}")
        
        print(f"Round {i}.2 of hyperparameter choosing.")
        print(f"train size: {len(train_labels_split_2)}. test size: {len(y_valid)}")
        p2 = evaluate_model_with_hyperparameters(train_data_split_2, train_labels_split_2, X_valid, y_valid, hyperparam, classification, model)
        performance.append(p1)
        performance.append(p2)
        print(f"Performance of corresponding hyperparams: {p2}")
   
    # Step 3: Average the results of these ten experiments for each of the parameter settings
    average_performance = np.mean(performance, axis=0)
    print(f"The average performance is: {average_performance}")
    # there will be only one value in the list, but our helper accept list only
    best_hyperparams = []
    # select the best hyperparams setting
    # performance is evaluated by mse for regression, and precision for classification
    if classification:
        # we want the max performance for classification
        best_hyperparams.append(hyperparam[np.argmax(average_performance)])
    else:
        # we want the min performance
        best_hyperparams.append(hyperparam[np.argmin(average_performance)])
        
    
    print(f"The best hyperparams setting is: {best_hyperparams}")
    # Step 4: Do the following five times
    best_hyperparam_performance = []
    
    for i in range(num_splits):
        # Step 4(a): Divide the 80% again into two equally-sized partitions (stratified)
        data_split_1, data_split_2, labels_split_1, labels_split_2 = train_test_split(X_train, y_train, 0.5, classification)
            
        # Step 4(b): Train a model using the tuned hyperparameters on the first half
        # and test on the second half
        print(f"Round {i}.1 performance testing of best hyperparameter setting.")
        print(f"train size: {len(labels_split_1)}. test size: {len(labels_split_2)}")
        p1 = evaluate_model_with_hyperparameters(data_split_1, labels_split_1, data_split_2, labels_split_2, best_hyperparams, classification, model)[0] # only one value
        print(f"Performance: {p1}")

        # Step 4(c): Train a model using the tuned hyperparameters on the second half
        # and test on the first half
        print(f"Round {i}.2 performance testing of best hyperparameter setting.")
        print(f"train size: {len(labels_split_2)}. test size: {len(labels_split_1)}")
        p2 = evaluate_model_with_hyperparameters(data_split_2, labels_split_2, data_split_1, labels_split_1, best_hyperparams, classification, model)[0] # only one value
        print(f"Performance: {p2}")
        
        best_hyperparam_performance.append(p1)
        best_hyperparam_performance.append(p2)

        
    # Step 5: Average the results of these ten experiments
    average_performance = np.mean(best_hyperparam_performance)
    print(f"The average performance is: {average_performance}")
    return best_hyperparams[0], average_performance

"""
Test train split
@params:
    data: feature data
    target: labels
    split_percentage: percentage of train split
@return:
    splited train_data, test_data, train_label, test_label   
"""
def train_test_split(data, target, split_percentage, classification):
    if classification:
        # If it's a classification task, stratify the split
        unique_classes, counts = np.unique(target, return_counts=True)
        stratified_split = (split_percentage * counts).astype(int)

        train_indices = []
        valid_indices = []

        for i in range(len(unique_classes)):
            class_indices = np.where(target == unique_classes[i])[0]
            np.random.shuffle(class_indices)
            train_indices.extend(class_indices[:stratified_split[i]])
            valid_indices.extend(class_indices[stratified_split[i]:])
    else:
        # For regression tasks, perform a simple random split
        num_samples = len(data)
        split_idx = int(split_percentage * num_samples)

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_indices = indices[:split_idx]
        valid_indices = indices[split_idx:]

    X_train, X_test = data[train_indices], data[valid_indices]
    y_train, y_test = target[train_indices], target[valid_indices]
    
    return X_train, X_test, y_train, y_test
    
"""
Helper function for cross validation
@params:
    train_data: training data
    train_label: the label/result of the corresponding training data
    test_data: testing data
    test_label: the label/result of the corresponding testing data
    hyperparam: list of hyperparams we need to evaluate
    classsification: (bool) whether the model is classification, help to determine with evaluation metric we will use
    model: the model we will be using
@return:
    performance: list of performance of each hyperparameter setting
"""
def evaluate_model_with_hyperparameters(train_data, train_label, test_data, test_label, hyperparam, classification, model):
    performance = []
    for param_setting in hyperparam:
        # Create a new instance of the model with the current hyperparameter setting
        model.update_param(param_setting)
        
        # Train the model on the training data
        model.fit(train_data, train_label)

        # Make predictions on the test data
        # returns neighbors and predictions for KNN, we do not care about neighbors in CV
        _, predictions = model.predict(test_data)
        
        # Calculate the appropriate evaluation metric based on classification or regression
        if classification:
            # Use classification metric (accuracy)
            performance_metric = calculate_classification_metric(test_label, predictions)
        else:
            # Use regression metric (mse)
            performance_metric = calculate_regression_metric(test_label, predictions)

        performance.append(performance_metric)
    return performance

"""
Helper function for hyperparameter performance function
@params:
    test_label: the true value
    prediction: the predicted value
@return: the classification precision of the prediction
"""
def calculate_regression_metric(test_label, predictions):
    regression_metric = evaluation_metric(test_label, predictions, task_type='regression')
    return regression_metric['mse']

"""
Helper function for hyperparameter performance function
@params:
    test_label: the true value
    prediction: the predicted value
@return: the classification precision of the prediction
"""
def calculate_classification_metric(test_label, predictions):
    classification_metric = evaluation_metric(test_label, predictions, task_type='classification')
    return classification_metric['accuracy']
    

"""
1.7 Evaluation Metic
Calculate the specified evaluation metric.
@params:
    ground_truth (numpy array or list): Ground truth values.
    predicted (numpy array or list): Predicted values.
    task_type (str): The type of task ('regression' or 'classification').
                     Default is 'regression'.

@returns:
    dictionary: The calculated evaluation metric values stored in dictionary for future reference.
"""
def evaluation_metric(ground_truth, predicted, task_type='regression'):
    output = {}
    ground_truth = np.array(ground_truth)
    predicted = np.array(predicted)

    if task_type == "regression":
        #  mse
        mse = np.mean(np.abs(ground_truth - predicted))
        output['mse'] = mse
        # r2
        y_bar = np.mean(ground_truth)
        ss_total = np.sum((ground_truth - y_bar) ** 2)
        ss_residual = np.sum((ground_truth - predicted) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        output['r2'] = r2
        # correlation
        correlation_matrix = np.corrcoef(ground_truth, predicted)
        output['correlation'] = correlation_matrix[0, 1]
        return output

    elif task_type == "classification":
        
        # Initialize variables to store precision and recall for each class
        num_classes = max(len(np.unique(ground_truth)), len(np.unique(predicted))) + 1
        precision = [0] * num_classes
        recall = [0] * num_classes
        f1 = [0] * num_classes
        true_positive = [0] * num_classes
        false_positive = [0] * num_classes
        false_negative = [0] * num_classes

        # Calculate true positives, false positives, and false negatives for each class
        for true_label, pred_label in zip(range(num_classes), range(num_classes)):
            if ground_truth[true_label] == predicted[pred_label]:
                true_positive[true_label] += 1
            else:
                false_positive[pred_label] += 1
                false_negative[true_label] += 1

        # Calculate precision and recall for each class
        for class_label in range(num_classes):
            if true_positive[class_label] + false_positive[class_label] != 0:
                precision[class_label] = true_positive[class_label] / (true_positive[class_label] + false_positive[class_label])
            if true_positive[class_label] + false_negative[class_label] != 0:
                recall[class_label] = true_positive[class_label] / (true_positive[class_label] + false_negative[class_label])
            if precision[class_label] != 0 and recall[class_label] != 0:
                f1[class_label] = 2 * (precision[class_label] * recall[class_label]) / (precision[class_label] + recall[class_label])

        output['precision'] = precision
        output['recall'] = recall
        output['F1'] = f1
        
        # accuracy
        output['accuracy'] = np.sum(ground_truth == predicted)/len(ground_truth)
        # precision
        
        return output
    else:
        # if the task type is not one of regression or classification, raise an error. 
        raise ValueError("Invalid task_type. Choose from 'regression' or 'classification'.")
        
        
"""
1.8 Learning Algorithms
Calculate predictions using the Null Model algorithm.
@params:
    train_data (numpy array or list): Training data.
    test_data (numpy array or list): Test data.
    task_type (str): The type of task ('regression' or 'classification').
                     Default is 'regression'.
@return: predictions (numpy array or list): Predicted values.
"""
def null_model_predict(train_data, test_data, task_type='regression'):
    if task_type == 'regression':
        # For regression tasks, return the mean of the training data as predictions
        mean_value = np.sum(train_data)/len(train_data)
        predictions = [mean_value] * len(test_data)
    elif task_type == 'classification':
        # For classification tasks, return the majority class label as predictions
        unique_classes, counts = np.unique(train_data, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        predictions = [majority_class] * len(test_data)
    else:
        raise ValueError("Invalid task_type. Choose from 'regression' or 'classification'.")

    return predictions    