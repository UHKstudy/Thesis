import logging
from datetime import datetime
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

log_file = "chatbotA_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s" ,
    
    datefmt="%Y-%m-%d %H:%M:%S"
)



def run_chatbot():
    """
    Simple chatbot (based on ListTrainer),  use defined sets of questions-answers 
    """ 

     #   1. Chatbot initialization - simple config 
 
    chatbot = ChatBot(
    "ChatbotA " ,

        
    database_uri="sqlite:///C:\\Users\\Dell\\Downloads\\AIbakalar\\PROJEKT\\chatbotA-DATABASE.db"
)
    trainer = ListTrainer(chatbot)

    #  Set of questions  ->  ==== answers ( Basic Facts  )

    conversation_basic = [
         # question  
        "What is the capital of Poland?",
        # panswer 
        "Warsaw",

        "How many days February have in a leap year?", # que
        "29 days.",    # ans
   
        "What is the boiling point of water in Celsius ?",
        "it is 100 degrees Celsius",

        "When is New Year day celebrated?",
        "It's celebrated on 1st of January.",

        "Who painted the Lady with an Ermine?",
        "Leonardo da Vinci.",


        "Where is painting called 'Lady with an Ermine'? ",
        "In Poland.",

        "What is the language used in the Czech Republic?",
        "It is Czech language. "

    ]

    #  Set of answers -> question  ->  ( Paraphrased Examples )

    
    #  np. rewording " Which city hosts Germany’s main government offices? "

    conversation_paraphrased = [

        "Which city hosts Poland's main government offices?",   # paraf question
        "Warsaw is where the main government offices are located." ,  # paraphr Answ

        "How hot must water be, before it starts boiling?",
        "Water usually boils at around 100° C under normal conditions.",

        "Could you name a famous painting by Leonardo da Vinci ?" ,
        "One famous painting by Leonardo da Vinci is the Lady with an Ermine.",

       "At what temperature does water start boiling in Celsius?" ,
        "Water boils at 100°C under normal atmospheric pressure.",

       "Warsaw is the capital of which country?",
       "The capital city of Poland is Warsaw.",


       "Who created the painting 'Lady with an Ermine'?",

        
        "Leonardo da Vinci painted the Lady with an Ermine. ",


        "In a leap year, how many days does February have?",
        "In a leap year , February has 29 days .",

        "On what date is New Year's Day celebrated ?",
        "New Year's Day is on January 1st.",

        "In which country is the 'Lady with an Ermine' painting located? ",
        "The painting called 'Lady with an Ermine' is in Poland.",

        "Which language is spoken in the Czech Republic ?",
        "Official language of the Czech Republic is Czech."



    ]

    #  chatbot training based on those (lista) 

    trainer.train(conversation_basic)

    trainer.train(conversation_paraphrased)

    #  Siple lop for questions :

    print(" 🤖🤖 ChatbotA 🤖🤖 is ONLINE. Wanna quit-> type 'exit' or 'quit' or 'bye chat' to stop it. ")
    while True:
        user_input = input("👦 Human:  " )

        if user_input.lower() in ["exit", "quit", "bye chat"]:
            
            print(" 🤖 ChatbotA: See ya! ")

            break 
                 
        # 7 taking answers from chatbot 
        response = chatbot.get_response(user_input)   
        confidence = getattr(response, 'confidence', "N/A")  # confidence 

        print( " 🤖 ChatbotA: " , response) 

       

        # 8>> Log
        log_message = f"{datetime.now()} | QUESTION: {user_input} | RESPONSE: { response } | CONFIDENCE: {confidence}"
        logging.info(log_message)  



if __name__ == "__main__":

    run_chatbot()
