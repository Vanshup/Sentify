import time
import os
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import requests
from bs4 import BeautifulSoup

# Define the file name
file_name = r"C:\Users\vijay\OneDrive\Documents\NetBeansProjects\Login-SignUp-java-gui-main\example.txt"

while True:
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            text_content = file.read()
        
        if text_content:
            # Clear the text file
            with open(file_name, "w") as file:
                file.write("")
            
            # Your sentiment analysis code here

            HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}

            # Replace 'something' with your desired URL or use the 'text_content' as the URL
            something = text_content

            response = requests.get(something, headers=HEADERS)

            soup = BeautifulSoup(response.text, 'lxml')
            title = soup.find('title')
            example = str(title)

            MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL)

            encoded_text = tokenizer(example, return_tensors='pt')
            output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            categories = ['Negative', 'Neutral', 'Positive']
            percentages = [scores[0], scores[1], scores[2]]

            plt.bar(categories, percentages, color=['green', 'red', 'gray'])

            plt.xlabel('Sentiment')
            plt.ylabel('Percentage')
            plt.title('Sentiment Distribution')

            plt.show()
        
    time.sleep(5)
