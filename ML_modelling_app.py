import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score, classification_report, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Creating website headings
st.title('Machine Learning Modelling')
st.markdown('This application is a Streamlit dashboard that can be used to analyse the performance of ML models.')
st.markdown("##### To view the basic visualisation of all the columns in the dataset, scroll down to 'Data Visualisation'.")
st.markdown("##### To assess the performance of the model, head over to 'Model Evaluation'.")
st.divider()

default_model = RandomForestClassifier(random_state=0)

def my_model(data, target, model):
    df = data.copy()
    df = df.dropna()
    cols = list(df)
    for col in cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        if df[col].dtype == 'datetime':
            df['Year'] = df[col].dt.year
            df['Month'] = df[col].dt.month
            df['Day'] = df[col].dt.day
            df = df.drop(col, axis = 1)
    x = df.drop(target, axis = 1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    return x_test, y_test, y_test_pred

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("models", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def save_uploaded_dataset(dataset):
    try:
        with open(os.path.join("datasets", dataset.name), "wb") as f:
            f.write(dataset.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists("datasets"):
    os.makedirs("datasets")

dataset = st.sidebar.file_uploader('Upload CSV file', type = 'csv')

tab = st.sidebar.radio("Would you like to train the deafult model or upload your own model?", ['Train a Model', 'Upload a Model'])

if tab == 'Train a Model':
    if dataset is None:
        st.header('Use Case Example: Predicting Star Types 💫')
        st.page_link('https://www.kaggle.com/datasets/deepu1109/star-dataset', label = 'Click here to view the dataset', icon = '📊')
        st.divider()
        data = pd.read_csv('stars.csv')
        target = 'Star type'
        x_test, y_test, y_test_pred = my_model(data, target, default_model)
    else:
        data = pd.read_csv(dataset)
else:
    st.header('⚠️ Please read these instructions before uploading your model.')
    st.markdown('#### Ensure your file contains this:')
    code = '''import os                                                  # and other libraries you have imported
dataset_path = os.path.join("datasets", your_dataset_name) # ensure to write the name of the dataset you have used in place of your_dataset_name
data = pd.read_csv(dataset_path)                           # edit your read_csv() function to read the path of the dataset defined above'''
    st.code(code, language = "python")
    st.write('The above will help in making a path to your dataset and ensuring your model can find and use it.')
    st.markdown('#### Please use the following naming convention:')
    st.markdown('''
- To define your model, use *model*
- To define your splitted data, use *x_train, x_test, y_train, y_test*
- To define your predicted data, use *y_test_pred*
                ''')
    st.markdown('''#### Latly, please ensure that you have performed all the pre-processing on your data, splitted your data and fitted your model on it.''')
    st.subheader("When you're all set, scroll down to see the performance of your model.")
    st.divider()

if tab == 'Train a Model':
    if dataset is None:
        target = 'Star type'
    else:
        target = st.sidebar.text_input('Specify the target column')
    with st.sidebar.form("train_model"):
        problem_type = st.selectbox("Problem Type:", options=['Classification', 'Regression'])
        n_estimators = st.slider("No of Estimators:", min_value=10, max_value=100)
        max_depth = st.slider("Max Depth:", min_value=2, max_value=20)
        min_samples_split = st.slider("Min Samples Split:", min_value=2, max_value=20)
        min_samples_leaf = st.slider("Min Samples Leaf:", min_value=2, max_value=20)
        max_leaf_nodes = st.slider("Max Leaf Nodes:", min_value = 2, max_value=100)
        max_features = st.selectbox("Max Features :", options=["sqrt", "log2", None])
        bootstrap = st.checkbox("Bootstrap")
        random_state = st.number_input("Random State:", value=0, step=1)
        submitted = st.form_submit_button("Train")
        if submitted:
            if target == '':
                st.write('Please specify a target variable.')
            else:
                try:
                    if problem_type == 'Classification':
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, max_features=max_features, bootstrap=bootstrap, random_state=random_state)
                        x_test, y_test, y_test_pred = my_model(data, target, model)
                    else:
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, max_features=max_features, bootstrap=bootstrap, random_state=random_state)
                        x_test, y_test, y_test_pred = my_model(data, target, model)
                except:
                    st.write('')

if tab == 'Upload a Model':
    problem = st.sidebar.radio("Choose problem type:", ['Classification', 'Regression'])
    uploaded_file = st.sidebar.file_uploader('Attach a .py file', type=["py"])
    if uploaded_file is not None and dataset is not None:
        try:
            dataset_path = os.path.join("datasets", dataset.name)
            model_path = os.path.join("models", uploaded_file.name)
            with open(model_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with open(dataset_path, "wb") as f:
                f.write(dataset.getbuffer())
            st.sidebar.success(f"Saved file: {uploaded_file.name}")
            with open(model_path, "r") as file:
                file_content = file.read()
            data = pd.read_csv(dataset_path)
            local_vars = {}
            exec(file_content, {}, local_vars)
            model = local_vars.get('model')
            y_test = local_vars.get('y_test')
            y_test_pred = local_vars.get('y_test_pred')
            x_test = local_vars.get('x_test')
        except:
            st.error('Please ensure you have inputted the correct files and problem type.')

tab1, tab2 = st.tabs(['Data Visualization', 'Model Evaluation'])

with tab1:
    try:
        if data is not None:
            st.markdown('### Displaying countplots for categorical variables:')
            st.write('Heads up! If your column names are too long, they may merge into each other.')
            object_col = []
            count = 0
            for col in list(data):
                if data[col].dtype == object:
                    object_col.append(col)
                    count += 1
            fig, axes = plt.subplots(int((count+1)/2), 2, figsize=(15, count * 2))
            for ax, category in zip(axes.flatten(), object_col):
                sb.countplot(x=category, data=data, ax=ax)
                ax.set_title(f'Distribution of {category}')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('### Displaying histplots for numerical variables:')
            numerical_col = []
            count = 0
            for col in list(data):
                if data[col].dtype == int or data[col].dtype == float:
                    numerical_col.append(col)
                    count += 1
            fig, axes = plt.subplots(int((count+1)/2), 2, figsize=(15, count * 2))
            for ax, numerical in zip(axes.flatten(), numerical_col):
                sb.histplot(data[numerical], bins=10, kde=True, ax=ax)
                ax.set_title(f'Distribution of {numerical}')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error('Please ensure required files are uploaded.')
    except:
        st.error('Please ensure required files are uploaded.')

with tab2:
    if tab == 'Upload a Model':
        try:
            if problem == 'Classification':
                st.metric("Test Accuracy", value="{:.2f} %".format(100*accuracy_score(y_test, y_test_pred)))
                st.markdown("### Classification Report:")
                st.code("=="+classification_report(y_test, y_test_pred))
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    f1, ax = plt.subplots(1, 1, figsize=(8, 4.5))
                    sb.heatmap(confusion_matrix(y_test, y_test_pred), annot = True, fmt=".0f", annot_kws={"size": 18})
                    plt.title('Confusion Matrix')
                    st.pyplot(f1)
                    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred) 
                    roc_auc = auc(fpr, tpr)
                    f2 = plt.figure()  
                    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    st.pyplot(f2)
                with col2:
                    f1 = f1_score(y_test, y_test_pred, average='weighted')
                    precision = precision_score(y_test, y_test_pred, average='weighted')
                    recall = recall_score(y_test, y_test_pred, average='weighted')
                    logloss = log_loss(y_test, y_test_pred)
                    st.write('### Key Metrics')
                    st.write(f'F1 Score: {f1:.2f}')
                    st.write(f'Precision: {precision:.2f}')
                    st.write(f'Recall: {recall:.2f}')
                    st.write(f'Log Loss: {logloss:.2f}')
                    st.write('')
                    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
                    f3, ax = plt.subplots()
                    ax.plot(recall, precision, color='purple')
                    ax.set_title('Precision-Recall Curve')
                    ax.set_ylabel('Precision')
                    ax.set_xlabel('Recall')
                    st.pyplot(f3)
            elif problem == 'Regression':
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    score = model.score(x_test, y_test)
                    st.markdown(f'### Test Accuracy:')
                    st.markdown(f'#### {score}')
                    st.markdown("### Regression Model Metrics:")
                    mae = mean_absolute_error(y_test, y_test_pred)
                    mse = mean_squared_error(y_test, y_test_pred)
                    rmse = mse ** 0.5
                    r2 = r2_score(y_test, y_test_pred)
                    metrics = pd.DataFrame({
                        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                        'Value': [mae, mse, rmse, r2]
                    })
                    st.table(metrics)
                with col2:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.scatter(y_test, y_test_pred)
                    ax.plot(y_test.to_numpy(), y_test.to_numpy(), 'r-', linewidth = 1)
                    ax.set_xlabel("True values of the Response Variable (Test)")
                    ax.set_ylabel("Predicted values of the Response Variable (Test)")
                    ax.set_title('Predicted vs. Actual Plot')
                    st.pyplot(fig)
                fig2, axs = plt.subplots(1, 2, figsize=(15, 5))
                axs[0].scatter(y_test_pred, y_test_pred - y_test,  marker='o', label='Test data')
                axs[0].hlines(y=0, xmin=min(y_test_pred), xmax=max(y_test_pred), color='red')
                axs[0].set_title('Residuals Plot')
                axs[0].set_xlabel('Predicted values')
                axs[0].set_ylabel('Residuals')
                sb.histplot(y_test_pred - y_test, kde=True, ax=axs[1])
                axs[1].set_title('Distribution of Residuals')
                st.pyplot(fig2)
        except:
            st.error('Please ensure you have inputted the correct files and problem type.')
    elif target == '':
        st.write('Please specify a target variable!')
    elif not submitted:
        st.write("Please click on the 'Train' button at the end of the 'Train a Model' tab!")
    elif tab == 'Train a Model':
        if data is not None:
            st.markdown("#### Let's see if we can improve the accuracy of our model. Play around with the parameters and click on 'Train'.")
        try:
            if problem_type == 'Classification':
                st.metric("Test Accuracy", value="{:.2f} %".format(100*accuracy_score(y_test, y_test_pred)))
                st.markdown("### Classification Report:")
                st.code("=="+classification_report(y_test, y_test_pred))
                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    f1, ax = plt.subplots(1, 1, figsize=(8, 4.5))
                    sb.heatmap(confusion_matrix(y_test, y_test_pred), annot = True, fmt=".0f", annot_kws={"size": 18})
                    plt.title('Confusion Matrix')
                    st.pyplot(f1)
                    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred) 
                    roc_auc = auc(fpr, tpr)
                    f2 = plt.figure()  
                    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    st.pyplot(f2)
                with col2:
                    f1 = f1_score(y_test, y_test_pred, average='weighted')
                    precision = precision_score(y_test, y_test_pred, average='weighted')
                    recall = recall_score(y_test, y_test_pred, average='weighted')
                    logloss = log_loss(y_test, y_test_pred)
                    st.write('### Key Metrics')
                    st.write(f'F1 Score: {f1:.2f}')
                    st.write(f'Precision: {precision:.2f}')
                    st.write(f'Recall: {recall:.2f}')
                    st.write(f'Log Loss: {logloss:.2f}')
                    st.write('')
                    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
                    f3, ax = plt.subplots()
                    ax.plot(recall, precision, color='purple')
                    ax.set_title('Precision-Recall Curve')
                    ax.set_ylabel('Precision')
                    ax.set_xlabel('Recall')
                    st.pyplot(f3)
            elif problem_type == 'Regression':
                col1, col2 = st.columns(2, gap="small")
                with col1:
                    score = model.score(x_test, y_test)
                    st.markdown(f'### Test Accuracy:')
                    st.markdown(f'#### {score}')
                    st.markdown("### Regression Model Metrics:")
                    mae = mean_absolute_error(y_test, y_test_pred)
                    mse = mean_squared_error(y_test, y_test_pred)
                    rmse = mse ** 0.5
                    r2 = r2_score(y_test, y_test_pred)
                    metrics = pd.DataFrame({
                        'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                        'Value': [mae, mse, rmse, r2]
                    })
                    st.table(metrics)
                with col2:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    ax.scatter(y_test, y_test_pred)
                    ax.plot(y_test.to_numpy(), y_test.to_numpy(), 'r-', linewidth = 1)
                    ax.set_xlabel("True values of the Response Variable (Test)")
                    ax.set_ylabel("Predicted values of the Response Variable (Test)")
                    ax.set_title('Predicted vs. Actual Plot')
                    st.pyplot(fig)
                fig2, axs = plt.subplots(1, 2, figsize=(15, 5))
                axs[0].scatter(y_test_pred, y_test_pred - y_test,  marker='o', label='Test data')
                axs[0].hlines(y=0, xmin=min(y_test_pred), xmax=max(y_test_pred), color='red')
                axs[0].set_title('Residuals Plot')
                axs[0].set_xlabel('Predicted values')
                axs[0].set_ylabel('Residuals')
                sb.histplot(y_test_pred - y_test, kde=True, ax=axs[1])
                axs[1].set_title('Distribution of Residuals')
                st.pyplot(fig2)
        except:
            st.write('Please check your input and problem type!')