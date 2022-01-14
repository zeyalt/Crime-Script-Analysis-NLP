<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:05:22 2020

@author: Zeya
"""

=======
>>>>>>> 21815d2 (latest updates)
# =============================================================================
# IMPORT NECESSARY LIBRARIES
# =============================================================================

import streamlit as st
import pandas as pd
from inference import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import plotly.express as px 
from keras.models import load_model
import pickle
<<<<<<< HEAD
# import plotnine as p9
# from plotnine import ggplot, geom_point, aes, theme, element_text, scale_x_date
# from mizani.breaks import date_breaks, minor_breaks
# from mizani.formatters import date_format
=======
>>>>>>> 21815d2 (latest updates)

# =============================================================================
# DEFINE FUNCTIONS
# =============================================================================

@st.cache(suppress_st_warning=True, show_spinner=False)
def load_data(data):

    df_to_display = data[['date', 'day', 'informant', 
                        'incident_description', 'scammer_name', 
                        'scammer_contact', 'scam_type']]
    df_to_display.columns = ['Date', 'Day', "Informant's Name", 
                             "Scam Story", "Scammer's Name", 
                             "Scammer's Contact", "Scam Type"]
    
    return df_to_display

@st.cache(suppress_st_warning=True, show_spinner=False)
def create_corpus(data, col_name):
    return [TaggedDocument(words=word_tokenize(_d.lower()), tags=[i]) 
            for i, _d in enumerate(list(data[col_name]))]

@st.cache(suppress_st_warning=True, show_spinner=False)
def return_predictions(predicted_proba):

    z1 = pd.DataFrame(list(scam_type_cat_mapping.items()))
    z1.columns = ["Index", "Scam Type"]

    z2 = pd.DataFrame([{a: b for a, b in enumerate(predicted_proba[0])}]).transpose().reset_index()
    z2.columns = ["Index", "Probability"]

    return pd.merge(z1, z2, how="inner", on="Index")[["Scam Type", "Probability"]].sort_values(by="Probability", ascending=False)

@st.cache(suppress_st_warning=True, show_spinner=False)
def plot_line_chart(data):
    df_by_date = filtered_df.groupby(by='date')['scam_type'].count().to_frame('count').reset_index()
    df_by_date.set_index('date', inplace=True)
    df_by_date_ma = df_by_date['count'].rolling(window=14).mean().to_frame().dropna().reset_index()
    fig_1 = px.line(df_by_date_ma, x = "date", y = "count", labels = {"date": "Year", "count": "Number of Scam Reports"})
    return fig_1

@st.cache(suppress_st_warning=True, show_spinner=False)
def plot_bar_chart_by_agg_mode(data, selected_agg_mode):
    
    if selected_agg_mode == 'Year':
        
        df_by_year = data.groupby(by='year')['scam_type'].count().to_frame('count').reset_index()
        
        fig_2 = px.bar(x = df_by_year['year'], y = df_by_year['count'], color_discrete_sequence=['#50B2C0'],
                        labels = {"x": "Year", "y": "Number of Scam Reports"})
    
    elif selected_agg_mode == 'Month':
    
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']    
        df_by_month = data.groupby(by=['year', 'month'])['scam_type'].count().to_frame().reset_index().pivot(index = 'month', columns='year', values='scam_type')
        df_by_month = df_by_month.reset_index()
        df_by_month['total'] = df_by_month.sum(axis=1)
        df_by_month['month'] = df_by_month['month'].astype('category')
        df_by_month['month'].cat.set_categories(new_categories=month_order, ordered=True, inplace=True)
        df_by_month['month_cat'] = df_by_month['month'].cat.codes
        df_by_month = df_by_month.sort_values('month_cat').reset_index().drop(columns=['index'])
        
        fig_2 = px.bar(x = df_by_month['month'], y = df_by_month['total'], color_discrete_sequence=['#50B2C0'],
                        labels = {"x": "Month", "y": "Number of Scam Reports"})
    
    elif selected_agg_mode == "Day of the Week":
        
        day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        df_by_day = data.groupby(by='day')['scam_type'].count().to_frame('count').reset_index()
        df_by_day['day'] = df_by_day['day'].astype('category')
        df_by_day['day'].cat.set_categories(new_categories=day_order, ordered=True, inplace=True)
        df_by_day['day_cat'] = df_by_day['day'].cat.codes
        df_by_day = df_by_day.sort_values('day_cat').reset_index().drop(columns=['index'])
        
        fig_2 = px.bar(x = df_by_day['day'], y = df_by_day['count'], color_discrete_sequence=['#50B2C0'],
                        labels = {"x": "Day of the Week", "y": "Number of Scam Reports"})
    
    elif selected_agg_mode == "Daily Average":
        
        df_by_year = data.groupby(by='year')['scam_type'].count().to_frame('count').reset_index()
        df_by_n_days = data.groupby(by='year')['date'].nunique().to_frame('n_days').reset_index()
        df_by_daily_avg = pd.merge(df_by_year, df_by_n_days)
        df_by_daily_avg['daily_avg'] = round(df_by_daily_avg['count']/df_by_daily_avg['n_days'], 2)
    
        fig_2 = px.bar(x = df_by_daily_avg['year'], y = df_by_daily_avg['daily_avg'], color_discrete_sequence=['#50B2C0'],
                        labels = {"x": "Year", "y": "Number of Scam Reports"})
        
    return fig_2

@st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
def import_all():

    # Import dataset
    df = pd.read_csv('Data/scam_data_5.csv')

    # Manipulate imported data for display
    df_to_display = load_data(df)
    
    # Import doc2vec model
    best_model = Doc2Vec.load("Models/Doc2Vec_50D_PVDM/doc2vec_model_5.model")

    # Define corpus
    corpus = create_corpus(df, 'preprocessed_text')
    
    # Load the tokeniser
    with open('Tokenizer/scam_classifier_augmented_text_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle) 
    
    # Load dictionary for mapping of categories
    a_file = open("Data/scam_type_cat_mapping.pkl", "rb")
    scam_type_cat_mapping = pickle.load(a_file)
    
    return df, best_model, df_to_display, corpus, tokenizer, scam_type_cat_mapping

@st.cache(suppress_st_warning=True, show_spinner=False)
def generate_n_list(n_grams):
    n_grams_list = ["Unigram", "Bigram", "Trigram"]
    n_grams_dict = {j: i+1 for i, j in enumerate(n_grams_list)}
    return [n_grams_dict.get(i) for i in n_grams]

# =============================================================================
# IMPORT DATA AND OTHER REQUIRED FILES
# =============================================================================

# Import data and other required files
df, best_model, df_to_display, corpus, tokenizer, scam_type_cat_mapping = import_all()

# Load deep learning model
best_dl_model = load_model("Models/Scam_Classifier_Augmented_Text/best-LSTM-fold-5.h5")

# =============================================================================
# APP INTERFACE - SIDEBAR
# =============================================================================

# How to use ScamWatch?
st.sidebar.header("How to use ScamWatch?")
st.sidebar.markdown("Under the **Input** section:")
st.sidebar.markdown("1) Describe your scam story in the free text field;")
st.sidebar.markdown("2) Choose a similarity metric. Cosine similarity measures similarity between embeddings (or vectors of numbers) representing two documents, whereas Jaccard similarity measures similarity in terms of words that two documents have in common; and")
st.sidebar.markdown("3) Select percentile of most similar scam reports to generate and analyse. For example, a percentile of 0.05 means the top 5% of most similar scam reports.")

st.sidebar.markdown("Under the **Output** section, you will be able to:")
st.sidebar.markdown("1) Find scam reports most similar to your scam story;")
st.sidebar.markdown("2) Generate key words or phrases from these most similar scam reports; and")
st.sidebar.markdown("3) Predict a scam type classification for your scam story. ")

# About ScamWatch
st.sidebar.header("About ScamWatch")
st.sidebar.info("ScamWatch is an web application that aims to provide insights on scams in Singapore. It draws lessons from others’ scam experiences shared on [Scam Alert](https://scamalert.sg/stories). The underlying algorithms harness the hidden potential of free text in scam reports using machine learning and Natural Language Processing (NLP) methods.")

# About me
st.sidebar.header("About me")
st.sidebar.info("Hello! My name is Zeya. This app was developed to put into deployment models I have built as part of my Master's dissertation project titled 'Supervised and Unsupervised Applications of Natural Language Processing on Free Text towards Tackling Scams'. All analyses underlying my research are accessible at this [Github repository](https://github.com/zeyalt/msc-dissertation-final). \n\n\nIf you have any feedback or suggestions for improvement regarding this app, feel free to reach out to me at zeya.zlt@gmail.com or via [my LinkedIn page](https://www.linkedin.com/in/zeyalt).")

# Acknowledgements
st.sidebar.header("Acknowledgements")
st.sidebar.info("National Crime Prevention Council, Singapore")

# =============================================================================
# APP INTERFACE - MAIN TITLE & INPUT
# =============================================================================

# Main title
st.markdown("# ScamWatch")

# Input
st.markdown("### **Input**")
text_input = st.text_area("Describe your scam story", height=180)
input1, input2 = st.beta_columns([1, 1])
sim_metric = input1.selectbox('Choose similarity metric', ['Jaccard', 'Cosine'])
q = 1 - input2.slider("Select percentile of most similar scam reports to analyse", min_value=0.01, max_value=0.40, step=0.01, value = 0.05)  

x1 = find_similar_docs_cosine_jaccard(model=best_model, corpus=corpus, data=df, tag_id=None, new_doc=text_input)
x2 = x1[x1[sim_metric.lower()] >= x1[sim_metric.lower()].quantile(q)]
x2 = x2.sort_values(by=sim_metric.lower(), ascending=False)
x2.columns = ['Tag ID', 'Cosine', 'Jaccard', 'Scam Story', 'Preprocessed', 'Scam Type']
x3 = x2[['Tag ID', sim_metric, 'Scam Story', 'Scam Type']]

# =============================================================================
# APP INTERFACE - OUTPUT
# =============================================================================

# Output
st.markdown("### **Output**")

# Output 1 - Finding Similar Scam Reports
output1 = st.beta_expander("Finding Similar Scam Reports")

if text_input == "":
    output1.warning("You have not entered any text in the 'Input' section.")
        
else:
    result_summary = "A total of " + str(len(x3)) + " scam reports found."
    output1.success(result_summary)
    output1.table(x3)
    
# Output 2 - Generating Key Terms from Similar Scam Reports   
output2 = st.beta_expander("Generating Key Terms from Similar Scam Reports")

if text_input == "":
    output2.warning("You have not entered any text in the 'Input' section.")
        
else:
    with output2.beta_container():
        
        col1, col2, col3 = output2.beta_columns(3)
        n_grams = col1.multiselect("Select n-gram", ["Unigram", "Bigram", "Trigram"])
        k = col2.slider("Select number of top key terms to view", min_value=5, max_value=30)  
        view_mode = col3.selectbox("Select view mode", ["Bar Chart", "Table", "Directed Graph"])
    
    if n_grams == []:
        output2.warning("Please select at least one n-gram option.")
    
    else:    

        n = generate_n_list(n_grams)
        without_stopwords = documents_without_stopwords(list(x2['Preprocessed']))
        t_df, ranking_df = extract_n_grams(doc_list=without_stopwords, n_min=min(n), n_max=max(n), top_n=k)
        ranking_df['tfidf_score'] = round(ranking_df['tfidf_score'], 3)

        dynamic_title = "Top " + str(k) + " Key Terms Extracted from " + str(len(x3)) + " Scam Reports"
    
        if view_mode == "Bar Chart":
            output2.write(dynamic_title)
            fig = px.bar(x = ranking_df['term'], y = ranking_df['tfidf_score'], 
                         labels = {"x": "", "y": "TF-IDF Score"})
            fig.update_traces(hovertemplate=None)
            fig.update_layout(hovermode="x")
            fig.update_xaxes(tickangle = 45)
            output2.plotly_chart(fig)
        
        elif view_mode == "Table":
            output2.write(dynamic_title)
            ranking_df.columns = ['Term', 'TF-IDF Score']
            output2.table(ranking_df.reset_index().drop(columns=['index']))
        
        elif view_mode == "Directed Graph":
            output2.write("Top " + str(k) + " Key Terms in Sequence")
            ngram_df = arrange_n_grams_in_sequence(ranking_df, without_stopwords).reset_index()
            ngram_df = create_next_term_column(ngram_df)
            fig = visualise_n_grams(ngram_df)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            output2.pyplot(fig)

# Output 3 - Classifying Scam Report
output3 = st.beta_expander("Classifying Scam Report")

if text_input == "":
    output3.warning("You have not entered any text in the 'Input' section.")
        
else:

    classification_mode = output3.selectbox("Select classification method", ["Doc2Vec model (Unsupervised)", "LSTM model (Supervised)"])
    
    if classification_mode == "Doc2Vec model (Unsupervised)":
        
        c1 = Counter(x3['Scam Type'])
        c2 = [(i, round(c1[i] / len(x3), 3)) for i in c1]
        d1 = pd.DataFrame(dict(c1), index=[0]).transpose().reset_index()
        d2 = pd.DataFrame(dict(c2), index=[0]).transpose().reset_index()
    
        d = pd.merge(d1, d2, how='left', on='index')
        d.columns = ['Scam Type', 'Count', 'Probability']
        d = d.sort_values('Probability', ascending=False)
        
        predicted_class = d.reset_index()["Scam Type"][0]
        predicted_proba = d.reset_index()["Probability"][0]
        c1, c2 = output3.beta_columns((2, 1))
        c1.dataframe(d)
        c2.write("Based on " + str(len(x3)) + " most similar scam reports, the scam you encountered is likely to be")
        c2.subheader(predicted_class)
           
    else:
        
        predicted_proba, predicted_class = predict_label(model=best_dl_model, text=text_input, tokenizer=tokenizer, label_to_idx=scam_type_cat_mapping)
        predictions_df = return_predictions(predicted_proba)
        c1, c2 = output3.beta_columns((2, 1))
        c1.dataframe(predictions_df)
        c2.write("The scam you encountered is likely to beeeee")
        c2.subheader(predicted_class)
