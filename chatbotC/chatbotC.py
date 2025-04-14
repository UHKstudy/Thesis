import sys
import os
import pickle

import logging
import time
import psutil
from datetime import datetime 
 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np    

def setup_logger() :
    log_file = "chatbotC_unsupervised.log"
    logging.basicConfig(

        filename=log_file,
        level=logging.DEBUG ,
        format="%(asctime)s - %(levelname)s - %(message)s"

    )

def run_chatbot() :
    setup_logger()
    process = psutil.Process(os.getpid())   

    load_start =  time.time()

    # Przykładowe QA (musisz wypełnić resztą pytań, bo w kodzie jest placeholder)
    training_qa =[
    ("What is the capital of Poland?" , "Warsaw (Warszawa)") ,
    ("How many days does February have in a leap year?", "29 days in a leap year."),
    ("What is the name of university in Hradec Kralove?" , "The name of the university in Hradec Kralove is University of Hradec Kralove (UHK)."),  
    ("When is New Year’s Day celebrated?" , "On January 1st." ),
    ("Who painted the Lady with an Ermine?", "It was painted by Leonardo da Vinci.") ,
    ("Where is the painting called 'Lady with an Ermine' located?", "It's located in Poland (Kraków)."),
    ("What is the language used in the Czech Republic?" , "The official language is Czech."),
    ]

    training_questions = [qa[0] for qa in training_qa]

    training_answers = [qa[1] for qa in training_qa]

    n_clusters = 4

    model_path = "chatbotC_kmeans.pkl"
    vectorizer_path = "chatbotC_vectorizer.pkl"       
    cluster_assignments_path = "chatbotC_assignments.pkl"

    model_loaded = False

    if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(cluster_assignments_path):
        with open(model_path, "rb") as f_model:     
            kmeans = pickle.load(f_model)
        with open(vectorizer_path, "rb") as f_vec:
            vectorizer = pickle.load(f_vec) 
        with open(cluster_assignments_path, "rb") as f_assign :
            question_clusters = pickle.load(f_assign)

        print("Loaded existing Unsupervised Model")
        model_loaded =True

    else:
        print("No saved Unsupervised model found. Training from scratch, anew...")

        vectorizer = TfidfVectorizer()

        X_vectors = vectorizer.fit_transform(training_questions)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42 , n_init=10, max_iter=300)
        kmeans.fit(X_vectors)

        question_clusters = []
        for i, q in enumerate(training_questions):
            vec_q = vectorizer.transform([q])
            clust_label = kmeans.predict(vec_q)[0]
            question_clusters.append(clust_label)

        with open(model_path, "wb") as f_model:
            pickle.dump(kmeans, f_model)
        with open(vectorizer_path, "wb") as f_vec :
            pickle.dump(vectorizer, f_vec)
        with open(cluster_assignments_path, "wb") as f_assign:
            pickle.dump(question_clusters, f_assign)

        print("K-Means done & saved !")

    load_end = time.time()

    load_time = load_end - load_start
    logging.info(f"Model load/build time: {load_time:.4f}s (model_loaded={model_loaded})")

    print(" 🤖🤖ChatbotC🤖🤖 (Unsupervised) is ready to use . You can type 'bye chat' 'exit' or 'quit' to stop chat")

    previous_responses = {}

    while True :
        user_input = input("🧑👩 Human(s) User(s): ")
        if user_input.lower() in ["exit", "quit", "bye chat"] :
            print(" 🤖 ChatbotC: Bye friend!")
            sys.exit(0)

        start_time = time.time()
        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024*1024)

        user_vec = vectorizer.transform([user_input])
        cluster_label = kmeans.predict(user_vec)[0]
        distances_all = kmeans.transform(user_vec)[0]
        dist_to_chosen = distances_all[cluster_label]
        max_dist = max(distances_all)
        if max_dist == 0 :
            confidence = 1.0
        else:
            confidence = 1 - (dist_to_chosen / max_dist)

        if confidence < 0.3:
            response = " 🤖 (fallback) I'm not sure how to answer that q."
        else:
            candidates_idx =  [i for i, c in enumerate(question_clusters) if c == cluster_label]
            best_idx = None
            best_dist = 9999
            for i in candidates_idx :

                cand_question = training_questions[i]

                cand_vec = vectorizer.transform([cand_question])
                diff = user_vec - cand_vec
                
                dist = np.linalg.norm(diff.toarray())
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is None :
                response = " 🤖 (fallback) I am not sure"
            else:
                response = training_answers[best_idx]

        mem_after = process.memory_info().rss / (1024*1024)

        cpu_after = process.cpu_percent(interval=None)

        end_time = time.time()


        elapsed = end_time - start_time

        length_resp = len(response.split())

        print(" 🤖 ChatbotC:", response)

        was_repeated = False
        if user_input in previous_responses:
            if previous_responses[user_input] == response :

                was_repeated = True
        previous_responses[user_input] =  response

        if " 🤖 (fallback)" in response:
            logging.warning(f"[WARNING] Possibly fallback for input='{user_input}'")

        log_msg = (

            f"{datetime.now()} | Q='{user_input}' | cluster={cluster_label} | conf={confidence:.4f} | "

            f"Resp='{response}' | lenResp={length_resp} | Repeated={was_repeated} | "

            f"TIME={elapsed:.4f}s | CPU=({cpu_before:.2f}%->{cpu_after:.2f}%) | "
            f"MEM=({mem_before:.2f}MB->{mem_after:.2f}MB) "

        )
        logging.info(log_msg)

if __name__ == "__main__":
    run_chatbot()
