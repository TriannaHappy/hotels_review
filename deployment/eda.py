import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud

# Untuk melebarkan streamlit, harus diletakkan setelh import
# Ketika dieksekusi akan mempengaruhi main dan prediction
# Tidak perlu dijalankan dalam fungsi
st.set_page_config(
    page_title='Hotel Rating Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)


# bagian bawah ini tidak bisa dijalankan jika tidak dieksekusi
def run():
    #Membuat Title
    st.title('Hotel Rating Prediction Using Recurrent Neural Network')

    # Membuat Sub Header
    st.subheader('EDA for Reviews from Visitors')

    # Menambahkan Gambar
    image=Image.open('hotel.jpg')
    st.markdown(
    """
    <style>
    img {
        cursor: pointer;
        transition: all .2s ease-in-out;
    }
    img:hover {
        transform: scale(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.image(image, caption='Rating Prediction')

    # Menambah Deskripsi
    st.write('Made by *Happy Trianna*')
    
    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    Pada page kali ini, pennulis akan melakukan eksplorasi mengenai dataset review.
    Dataset yang digunakan adalah dataset review hotel di London yang didapat dari kaggle.com.
    '''

    # Show DataFrame
    st.write('#### Head of Reviews Dataset')
    df = pd.read_csv('London_hotel_reviews.csv', encoding = "ISO-8859-1")
    st.dataframe(df.head(10))

    # Get the rate and review text only to be analyzed
    data=df[['Review Rating', 'Review Text' ]]
    data.rename({'Review Rating': 'rate', 'Review Text': 'text'}, axis=1, inplace=True)

    # Create new dataframe from undersampled data
    df_eda=pd.read_csv('df_eda.csv')

    # Drop duplicated data
    df.drop_duplicates(inplace=True)
    df_eda.drop_duplicates(inplace=True)

    #Check the distribution of rate (after undersampled) given by visitors
    st.write('#### London Hotels Rating')
    fig = px.pie(df_eda, values=df_eda['rate'].value_counts(), 
             names=['1-2 Star','3 Star','4 Star', '5 Star'], title='Rating')
    st.plotly_chart(fig)

    # Analyze the characteristics of visitors based on hotels and review
    st.write('#### Analyze the characteristics of visitors based on hotels and review')
    fig = plt.figure(figsize=(8,8))
    sns.countplot(y=df['Property Name'], order=df['Property Name'].value_counts().index)
    st.pyplot(fig)

    # Plot of the rating given by visitors categorized by the hotel
    st.write('#### Plot of the rating given by visitors categorized by the hotel')
    st.write('##### The Savoy')
    fig = plt.figure(figsize=(5,5))
    savoy=df.loc[df['Property Name']=='The Savoy']
    sns.countplot(y=savoy['Review Rating'])
    st.pyplot(fig)

    st.write('##### Mondrian London at Sea Containers')
    fig = plt.figure(figsize=(5,5))
    mondrian=df.loc[df['Property Name']=='Mondrian London at Sea Containers']
    sns.countplot(y=mondrian['Review Rating'])
    st.pyplot(fig)

    st.write('##### The Rembrandt')
    fig = plt.figure(figsize=(5,5))
    rembrandt=df.loc[df['Property Name']=='The Rembrandt']
    sns.countplot(y=rembrandt['Review Rating'])
    st.pyplot(fig)

    st.write('##### City View Hotel')
    fig = plt.figure(figsize=(5,5))
    city=df.loc[df['Property Name']=='City View Hotel']
    sns.countplot(y=city['Review Rating'])
    st.pyplot(fig)

    st.write('##### Marble Arch Hotel')
    fig = plt.figure(figsize=(5,5))
    marble=df.loc[df['Property Name']=='Marble Arch Hotel']
    sns.countplot(y=marble['Review Rating'])
    st.pyplot(fig)

    st.write('##### Hartley Hotel')
    fig = plt.figure(figsize=(5,5))
    hartley=df.loc[df['Property Name']=='Hartley Hotel']
    sns.countplot(y=hartley['Review Rating'])
    st.pyplot(fig)

    # Wordcloud
    st.write('#### WordCloud From All Reviews')
    df_adj=pd.read_csv('df_adj.csv')
    all_adj= ' '.join(df_adj.text.values)

    fig =plt.figure(figsize=(6,6))
    cloud_all = WordCloud(max_words=5000, background_color="black", 
                      relative_scaling = 0.5,collocations=False).generate(all_adj)

    plt.imshow(cloud_all)
    plt.axis("off")
    st.pyplot(fig)

    # Wordcloud for each rate
    df_adj_r0_2=pd.read_csv('df_adj_r0_2.csv')
    df_adj_r1_2=pd.read_csv('df_adj_r1_2.csv')
    df_adj_r2_2=pd.read_csv('df_adj_r2_2.csv')
    df_adj_r3_2=pd.read_csv('df_adj_r3_2.csv')

    all_adj_r0_2= ' '.join(df_adj_r0_2.text.values)
    all_adj_r1_2= ' '.join(df_adj_r1_2.text.values)
    all_adj_r2_2= ' '.join(df_adj_r2_2.text.values)
    all_adj_r3_2= ' '.join(df_adj_r3_2.text.values)

    cloud_all0c = WordCloud(max_words=5000, background_color="black", 
                      relative_scaling = 0.5,collocations=False).generate(all_adj_r0_2)

    cloud_all1c = WordCloud(max_words=5000, background_color="black", 
                        relative_scaling = 0.5,collocations=False).generate(all_adj_r1_2)

    cloud_all2c = WordCloud(max_words=5000, background_color="black", 
                        relative_scaling = 0.5,collocations=False).generate(all_adj_r2_2)
    cloud_all3c = WordCloud(max_words=5000, background_color="black", 
                        relative_scaling = 0.5,collocations=False).generate(all_adj_r3_2)

    st.write('#### WordCloud For Each Rate')
    fig=plt.figure(figsize=[10,6])
    ax0 = plt.subplot(221)
    ax0.imshow(cloud_all0c)
    ax0.set_title("WordCloud for 0 Rate Hotel")
    ax0.axis("off")

    ax1 = plt.subplot(222)
    ax1.imshow(cloud_all1c)
    ax1.set_title("WordCloud for 1 Rate Hotel")
    ax1.axis("off")

    ax2 = plt.subplot(223)
    ax2.imshow(cloud_all2c)
    ax2.set_title("WordCloud for 2 Rate Hotel")
    ax2.axis("off")

    ax3 = plt.subplot(224)
    ax3.imshow(cloud_all3c)
    ax3.set_title("WordCloud for 3 Rate Hotel")
    ax3.axis("off")

    st.pyplot(fig)












    















    

if __name__ == '__main__':
    run()