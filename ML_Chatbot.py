# pip install nltk
# pip install ssl
# pip install streamlit
# pip install scikit-learn

import os
import nltk
import ssl
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')


intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },
    {
        "tag": "product_info",
        "patterns": ["Tell me about [product]", "What are the features of [product]", "Can you provide information on [product]"],
        "responses": ["[Product] is a [description of product]. Its features include [list of features].", "Sure, [product] is known for [highlighted feature] and [another highlighted feature].", "I'd be happy to help you learn more about [product]."]
    },
    {
        "tag": "tech_support",
        "patterns": ["I'm having trouble with [issue]", "How do I fix [issue]", "Help me troubleshoot [issue]"],
        "responses": ["Let's see if we can solve that. Have you tried [common solution]?","For [issue], a common solution is to [action]. Have you attempted that yet?", "I'm here to assist you with [issue]. Let's work through it together."]
    },
    {
        "tag": "booking_assistance",
        "patterns": ["How do I book [service]", "Can you help me with my reservation", "I need assistance with booking [service]"],
        "responses": ["Certainly! To book [service], you can visit our website or contact our customer service team at [phone number].", "I can guide you through the booking process. What [service] are you looking to book?", "Let's get started on your reservation. What dates are you considering for [service]?"]
    },
    {
        "tag": "education_resources",
        "patterns": ["Where can I find learning resources", "Do you have any educational materials", "I want to learn about [topic]"],
        "responses": ["There are various online platforms like Coursera, Khan Academy, and Udemy where you can find courses on [topic].", "I can suggest some books and websites for learning about [topic]. Would you like some recommendations?", "Learning about [topic] is a great idea! I can provide you with some resources to get started."]
    },
    {
        "tag": "health_tips",
        "patterns": ["How can I stay healthy", "Give me some wellness advice", "I want to improve my health"],
        "responses": ["Staying hydrated, eating balanced meals, and getting regular exercise are key to maintaining good health.", "Incorporating mindfulness practices like meditation can also promote overall wellness.", "It's important to prioritize self-care and listen to your body's needs. Small changes like getting enough sleep and managing stress can make a big difference."]
    }
]


# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter")
    
    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")
    
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response,height=100, max_chars=None)
        
        if response.lower() in ['goodbye','bye']:
            st.write("Thank you for chatting with me. Have a great day! ")
            st.stop()
            
if __name__ == '__main__':
    main()