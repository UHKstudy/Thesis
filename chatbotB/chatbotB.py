import sys
import os
import pickle
import logging
import time


import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime


print(f"DEBUG: datetime module loaded successfully at {datetime.now()}")

def setup_logger():
    """logger for debug saving to file (log) """        
    log_file = "chatbotB.log"
    logging.basicConfig(
        filename=log_file,  
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s" ,
    )

def run_chatbot():

    """
    Chatbot B > : Supervised Learning (scikit-learn) 

    Trains  classifier ,  loads a saved model ->> to respond faster     
    """
    setup_logger()  
    process = psutil.Process(os.getpid())   

    # time measurement of loading time / training model 
    load_start = time.time()         

    # 1  intents , and chatbot responses : 
    intent_responses = {
        "capital_of_poland": "The capital of Poland is Warsaw (Warszawa)." ,
        "days_in_feb_leap" : "February has 29 days in a leap year.",
        "university_hradec": "The name of the university in city called 'Hradec Kralove' is The University of Hradec Kralove (UHK).",
        "new_year_celebration": "New Year’s Day is celebrated on January 1st.",

        "who_painted_ermine": "The painting 'Lady with an Ermine' was done by Leonardo da Vinci." ,
        "location_ermine" : "The painting 'Lady with an Ermine' is located in Poland (Kraków).",
        "language_czech": "The official language of the Czech Republic is Czech." ,
        "fallback" : " I am not sure how i can answer that "
    }

    # 2  Training data -> (X_train: questions, y_train: intent labels)
    X_train = []
    y_train = []

    training_data = {
        "capital_of_poland": [
            "What is the capital of Poland?",
            "Which city is the capital of Poland?",
            "Poland's main government offices are located in which city?",
            "What's the Polish capital city?",
            "Warsaw is the capital of which country?"
        ],
        "days_in_feb_leap": [
            "How many days does February have in a leap year?",
            "In a leap year, how many days are in February?" ,
            "Days in February in leap year?"
        ] ,

        "university_hradec": [
            "What is the name of university in Hradec Kralove?",
            "What is the biggest computer science university in city called 'Hradec Kralove'?",
            "Which university that teach computer science is the biggest in Hradec Kralove city?",
            "University in Hradec Kralove?",
            "What is university of Hradec Kralove?",
            "What is the University of Hradec Kralove?",
            "UHK?",
            "Tell me about UHK",
            "University of Hradec Kralove",
            "Tell me about University of Hradec Kralove"
            ],
        "new_year_celebration" : [
            "When is New Year's Day celebrated?",
            "When do people celebrate New Year’s Day?",
            "Which day is New Year’s Day?" ,
            "On what date is New Year's Day celebrated?"
        ],
        "who_painted_ermine" : [
            "Who painted the Lady with an Ermine?",
            "Which artist created Lady with an Ermine?" ,
            "Could you name the painter of Lady with an Ermine?" ,
            "Who created the painting 'Lady with an Ermine'?"
            "Who painted the painting called 'Lady with an Ermine'?"
        ],

        "location_ermine" : [
            "Where is the Lady with an Ermine?",
            "Where can I find Lady with an Ermine painting?" ,
            "In which country is the Lady with an Ermine painting?",
            "Where is painting called 'Lady with an Ermine'?"
        ],

        "language_czech" : [
            "What is the language used in the Czech Republic?" ,
            "Which language is spoken in the Czech Republic?",
            "Official language of the Czech Republic?" ,
            "What language do people speak in the Czech Republic?"
        ]
    }

    for intent, examples in training_data.items():
        for question in examples:

            X_train.append(question)

            y_train.append(intent)

    model_path =  "chatbotB_model.pkl"
    vectorizer_path = "chatbotB_vectorizer.pkl"

    model_loaded = False
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):

        with open(model_path, "rb") as model_file :
            clf = pickle.load(model_file)

        with open(vectorizer_path, "rb") as vectorizer_file :  
            vectorizer = pickle.load(vectorizer_file)
        print("Loaded > pre-trained model")
        model_loaded = True
    else:
        print("No saved model found -> training a new model...")
        vectorizer = TfidfVectorizer()
        X_vectors = vectorizer.fit_transform(X_train)
        clf = LogisticRegression()
        clf.fit(X_vectors, y_train)

        with open(model_path, "wb") as model_file:
            pickle.dump(clf, model_file)

        with open(vectorizer_path, "wb") as vectorizer_file :
            pickle.dump(vectorizer, vectorizer_file) 
            print("Model trained and saved!")

    load_end = time.time()      
    load_time = load_end - load_start
    logging.info(f"Model load / build time : {load_time:.4f}s (model_loaded={model_loaded}) " )

    print(" 🤡🤡ChatbotB🤡🤡(Supervised Learning) is ready. Type 'bye chat' or 'exit' or 'quit' to stop it.")

    previous_responses = {}

    while True:
        user_input = input("🧑 Human user: ")
        if user_input.lower() in ["exit", "quit", "bye chat"]:
            print(" 🤡 ChatbotB: Bye user!")
            sys.exit(0)

        start_time = time.time()
        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024*1024)

        input_vec = vectorizer.transform([user_input])

        predicted_intent = clf.predict(input_vec)[0]
        probs = clf.predict_proba(input_vec)[0]

        max_prob = max( probs )

        if max_prob < 0.25:
           response = intent_responses["fallback"]
    
        else:
            response = intent_responses.get(predicted_intent, " 🤡 I am not sure")  

        mem_after = process.memory_info().rss / (1024*1024)
        cpu_after = process.cpu_percent(interval=None)
        end_time = time.time()
        elapsed = end_time - start_time

        length_resp = len(response.split())

        print(" 🤡 ChatbotB:" , response)


        was_repeated = False
        if user_input in previous_responses :

            if previous_responses[user_input] == response:
                was_repeated = True

        previous_responses[user_input] = response

        if response == "I'm not sure." or response == intent_responses["fallback"]:
            logging.warning(f" [WaRnInG!] Possibly incorrect / fallback for input='{user_input}'")


        log_message = (

            f"{datetime.now()} | QUESTION='{user_input}' | "
            f"PREDICTED_INTENT='{predicted_intent}' | CONFIDENCE={max_prob:.4f} | "
            f"RESPONSE='{response}' | lenResp={length_resp} | Repeated={was_repeated} | "
            f"TIME={elapsed:.4f}s | CPU=({cpu_before:.2f}%->{cpu_after:.2f}%) | "
            f"MEM=({mem_before:.2f}MB->{mem_after:.2f}MB) "

        )

        logging.info(log_message)

if __name__ == "__main__":

    run_chatbot()
