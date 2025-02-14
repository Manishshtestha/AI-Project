import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'ML_MODEL/knn_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('App Popularity Prediction')

    # Add a description
    st.write('Enter Application information to predict popularity.')
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Application Information')
        
        application_name = st.text_input("Appliation Name")
        rating = st.slider("Rating:",0.000,5.000,4.000)
        reviews = st.number_input("Number of Reviews:",step=1)
        size=st.number_input("Size in Bytes:",step=1000)
        installs = st.number_input("Number of Installs",step=1)
        price = st.number_input("Price")
        category = st.selectbox("Category:",['Auto and vehicles','Beauty','Books and reference','Business','Comics','Communication','Dating','Education','Entertainment','Events','Family','Finance','Food and drink','Game','Health and fitness','House and home','Libraries and demo','Lifestyle','Maps and navigation','Medical','News and magazines','Parenting','Personalization','Photography','Productivity','Shopping','Social','Sports','Tools','Travel and local','Videoplayers','Weather',])
        _type = st.selectbox("Type:",['Free','Paid'])
        content = st.selectbox("Content Rating",['Everyone', 'Everyone 10+','Mature 17+', 'Teen','Unrated'])
        
        
    rating = rating/5
    category = {'Auto and vehicles': 0, 'Beauty': 1, 'Books and reference': 2, 'Business': 3, 'Comics': 4,'Communication':5,'Dating':6,'Education':7,'Entertainment':8,'Events':9,'Family':10,'Finance':11,'Food and drink':12,'Game':13,'Health and fitness':14,'House and home':15,'Libraries and demo':16,'Lifestyles':17,'Maps and navigation': 18,'Medical':19,'News and magazines':20,'Parenting':21,'Personalization':22,'Photography':23,'Productivity':24,'Shopping':25,'Social':26,'Sports':27,'Tools':28,'Travel and local':29,'Videoplayers':30,'Weather':31}.get(category, 0)
    _type = 1 if _type == 'Paid' else 0
    content = {'Everyone':0,'Everyone 10+':1,'Mature 17+':2,'Teen':3,'Unrated':4}.get(content,0)
    
    input_data = pd.DataFrame({
        'Rating':[rating],
        'Reviews':[reviews],
        'Size':[size],
        'Installs':[installs],
        'Price':[price],
        'Category_AUTO_AND_VEHICLES':[1 if category == 0 else 0],
        'Category_BEAUTY':[1 if category == 1 else 0],
        'Category_BOOKS_AND_REFERENCE':[1 if category == 2 else 0],
        'Category_BUSINESS':[1 if category == 3 else 0],
        'Category_COMICS':[1 if category == 4 else 0],
        'Category_COMMUNICATION':[1 if category == 5 else 0],
        'Category_DATING':[1 if category == 6 else 0],
        'Category_EDUCATION':[1 if category == 7 else 0],
        'Category_ENTERTAINMENT':[1 if category == 8 else 0],
        'Category_EVENTS':[1 if category == 9 else 0], 
        'Category_FAMILY':[1 if category == 10 else 0],
        'Category_FINANCE':[1 if category == 11 else 0], 
        'Category_FOOD_AND_DRINK':[1 if category == 12 else 0], 
        'Category_GAME':[1 if category == 13 else 0],
        'Category_HEALTH_AND_FITNESS':[1 if category == 14 else 0], 
        'Category_HOUSE_AND_HOME':[1 if category == 15 else 0],
        'Category_LIBRARIES_AND_DEMO':[1 if category == 16 else 0], 
        'Category_LIFESTYLE':[1 if category == 17 else 0],
        'Category_MAPS_AND_NAVIGATION':[1 if category == 18 else 0], 
        'Category_MEDICAL':[1 if category == 19 else 0],
        'Category_NEWS_AND_MAGAZINES':[1 if category == 20 else 0], 
        'Category_PARENTING':[1 if category == 21 else 0],
        'Category_PERSONALIZATION':[1 if category == 22 else 0], 
        'Category_PHOTOGRAPHY':[1 if category == 23 else 0],
        'Category_PRODUCTIVITY':[1 if category == 24 else 0], 
        'Category_SHOPPING':[1 if category ==25 else 0], 
        'Category_SOCIAL':[1 if category == 26 else 0],
        'Category_SPORTS':[1 if category == 27 else 0], 
        'Category_TOOLS':[1 if category == 28 else 0], 
        'Category_TRAVEL_AND_LOCAL':[1 if category == 29 else 0],
        'Category_VIDEO_PLAYERS':[1 if category == 30 else 0], 
        'Category_WEATHER':[1 if category == 31 else 0], 
        'Type_Paid':[1 if _type == 1 else 0],
        'Content Rating_Everyone':[1 if content == 0 else 0], 
        'Content Rating_Everyone 10+':[1 if content == 1 else 0],
        'Content Rating_Mature 17+':[1 if content == 2 else 0], 
        'Content Rating_Teen':[1 if content == 3 else 0],
        'Content Rating_Unrated':[1 if content == 4 else 0]
    })
    
    #Ensure the columsn are in the same index as when the model was trained
    input_data = input_data[expected_columns]
    
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.write(f'Prediction for {application_name}: {"Popular" if prediction[0] == 1 else "Not Popular"}')
            st.write(f'Probability of getting Popular: {probability:.2f}')
            
                        # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Pass/Fail probability
            sns.barplot(x=['Not Popular', 'Popular'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Popularity Probability')
            axes[0].set_ylabel('Probability')


            # Plot Pass/Fail pie chart
            axes[2].pie([1 - probability, probability], labels=['Not Popular', 'Popular'], autopct='%1.1f%%', colors=['red', 'green'])
            axes[2].set_title('Popularity Prediction Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.success(f"{application_name_name} is likely to be popular. Keep up the good work!")
            else:
                st.error(f"{application_name} is likely to fail to gain popularity. Consider improving Rating and Number of Install.")
        
if __name__ == '__main__':
    main()