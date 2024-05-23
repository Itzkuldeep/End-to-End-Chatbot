import pymysql as sql
import random
import os
import nltk
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Function to establish database connection
def conn():
    srvr = sql.connect(user='root', password='', host='localhost', port=3306, database='chatbot')
    crsr = srvr.cursor()
    return srvr, crsr

# Setup NLTK data and SSL context
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Connect to the database and fetch data
srvr, cur = conn()
cmd = "select * from intents;"
cur.execute(cmd)
data1 = cur.fetchall()
srvr.close()  # Close the connection after fetching data

# Prepare patterns and tags
patterns = []
tags = []
for row in data1:
    patterns.append(row[2])  # Assuming the pattern is in the third column
    tags.append(row[1])      # Assuming the tag is in the second column

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function to get a response
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for row in data1:
        if row[1] == tag:  # Use integer indexing here
            responses = row[3]  # Assuming the responses are in the fourth column
            response_list = responses.split('|')  # Assuming responses are separated by '|'
            return random.choice(response_list)

# Counter for unique text input keys in Streamlit
counter = 0

# Main function for Streamlit app
def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter")

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
