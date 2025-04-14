import os
import random
import logging
import time
import psutil
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from datetime import datetime

#######
# configu
#########
keywords = [
    "capital",  "leonardo da vinci" , " berlin" , "warsaw",   "germany" , "uhk" , "university", "hradec", "leonardo" , "painter", "artist"  , "ermine" ,
    "painting", "february", "leap"  ,  "days" ,  "january", "new year", "language", "czech", "republic", "country", "leap year" , "poland" ,  "keep", "held" , "flag"  ,
    
       "located" , "location" , "author", "who", "when", "where", "what" , "how many", "date","name",
    "speak", "says", "called" , "displayed" , "known", "find" , "celebrate"  , "have",
    "number" ,  "city" ,  "sequence" , "difference", "explain", "concept"  , "gravity", "password", "colors"    , "mix"
]


STATE_SIZE = len(keywords)  # < <- ajustment ( autom ) 

actions_dict  =  {
    0: "Warsaw" ,
    1: "Leonardo da Vinci" ,
    2: "UHK (University of Hradec Kralove)" ,
    3: "29 days in a leap year" ,
    4: "I don't know"   ,
    5: "Prague" , 
    6: "Berlin",
    7: "1 January",
    8: "Poland" ,
    9: "Lady with an Ermine" ,
    10: "Czech" ,
    11: "This question is totally far beyond my knowledge"   ,
    12: "Germany"
}


ACTION_SIZE = len(actions_dict)

GAMMA = 0.95
LR = 0.001
MEMORY_SIZE = 1000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 20
EPSILON = 1.0
EPSILON_FILE = "epsilon_value.txt"
if os.path.exists(EPSILON_FILE) :

    with open(EPSILON_FILE, "r") as f :
        EPSILON =  float(f.read())
        print(f">><< Loaded epsilon value from file: {EPSILON:.3f}") 

EPS_MIN = 0.01
EPS_DECAY = 0.995


print(f"\n RL Chatbot keywords = {keywords}\n")


logging.basicConfig(filename="combined_extended.log", level=logging.DEBUG)

training_data = [
("Warsaw is capital of what country?", 8),

("Warsaw is the capital of what country?", 8),

("Where is the capital city of Poland?", 0) , 
("How many days in leap year February?", 3),
("University in Hradec Kralove?", 2) ,  
("UHK university location?"  , 2),
("New Year's Day date?", 7) ,     
("Painter of the Lady with an Ermine?", 1) ,
("Lady with Ermine location?" , 8) ,
("Czech Republic language?", 10),
("What language do people speak in Czechia?", 10) ,

("What is Poland's capital?", 0) ,
("What is the capital of Poland?", 0),
("Name Poland' s capital city" , 0) ,
("Capital of Poland is? ", 0),



("Who is the author of 'Lady with an Ermine'?" , 1),

("Who created the painting 'Lady with an Ermine'?", 1),
("Who is the painter of 'Lady with an Ermine'?" , 1) ,
("Which artist painted 'Lady with an Ermine'?" , 1),

("Which language do people speak in Czech Republic?", 10),
("Language spoken in Czech Republic?", 10),
("Czech Republic official language?", 10),

("Where is located the painting called 'Lady with an Ermine'?" , 8) ,
("Where can I find the painting 'Lady with an Ermine'?", 8) ,
("Where is 'Lady with an Ermine' located?" ,   8) ,

("Which city keeps 'Lady with an Ermine'?" , 8) ,

("Where is 'Lady with an Ermine' displayed?" , 8),

    
    
    ("In which city is 'Lady with an Ermine' located?", 9),
("Which language is spoken in the Czech Republic?" , 10),
("What is the capital of the Czech Republic?", 5),
        ("What is capital of Germany?" , 6),
        ("Berlin is capital of which country?" , 12),
        ("When is New Year's Day celebrated?" , 7), 
("What is the name of the painting , which features a lady with an ermine?" , 9) ,
("Where is 'Lady with an Ermine' kept?", 8), 


    ("When is January 1 celebrated?", 7),
("What day is New Year?", 7) ,
("Date of New Year's Day?" , 7),
("Who is the artist behind Lady with an Ermine?", 1) ,
("Name the painter of the Lady with an Ermine.", 1) ,
("Where you can found painting: Lady with an Ermine?", 8) ,

("Which country has Lady with an Ermine?", 8 )   ,
("What language do people speak in Czech Republic?" , 10),
("Which language do Czechs speak?" ,  10),

   ("What is the capital of Poland?", 0),
   ("Capitol of Poland?" , 0),
   ("How many days does February have in a leap year?", 3) ,

   ("What is the name of university in Hradec Kralove?", 2),
   ("Tell me about UHK", 2),
   ("Is there a university called UHK?", 2),
   ("What is UHK?", 2),

   ("When is New Year’s Day celebrated?" , 7),
   ("Who painted the Lady with an Ermine?", 1),

 
   ("What is the language used in the Czech Republic?", 10),
   ("Warsaw is the capital of which country?" , 8),

   ("How many days does February contain in a leap year?", 3) ,
   ("Which university that teaches computer science is the biggest in Hradec Kralove?", 2),

   ("On which date do people celebrate New Year’s Day?", 7) ,
   ("Who is the author of the painting 'Lady with an Ermine'?", 1),
   ("In which country can you find the painting 'Lady with an Ermine'?", 8),

   ("Which language is spoken in Czech Republic?", 10),
   ("What is the capital of Czech Republic?" , 5),
   ("What is the capital of Germany?", 6) ,

   # o ut- of- scope q =>> 4
       ("What is the next winning lottery number?", 11),
        ("What is my favorite color?", 11),
       ("How many planets will be discovered in the future?" , 11),
       ("What will the stock market do next week?", 11) ,

   ("Where will I be in 10 years?", 11),
   ("Can you tell me what I had for breakfast today?", 11),
   ("Who will be the next president of Poland?", 11) ,

   #  open - ended =>>  4
   ("Give me an example of a secure password." , 4) ,
   ("What is the next number in the sequence: 2, 4 , 8, 16...?", 4),
       ("What is the difference between a city and a country?",    4) ,

   ("How does a leap year work? " , 4) ,

   ("Explain the concept of gravity in simple terms.", 4),
          ("What are the main colors of the Polish flag?", 4) ,
   ("What happens when you mix blue and yellow paint? " , 4)
]

class ReplayMemory:
    def __init__(self,max_size=MEMORY_SIZE):
        self.memory=[]
        self.max_size=max_size

    def store(self,exp):
        if len(self.memory)>=self.max_size:
            self.memory.pop(0)
        self.memory.append(exp)
    def sample(self,bsize):
        return random.sample(self.memory,min(len(self.memory),bsize))

def build_model():
    m=Sequential([
       Input(shape=(STATE_SIZE ,)) ,

       Dense(24,activation='relu') ,

       Dense(24,activation='relu') ,
       Dense(ACTION_SIZE,activation='linear')
    ])
    m.compile(optimizer=Adam(LR), loss='mse')
    return m

memory=ReplayMemory()
model=build_model()
target_model=build_model()

model_path  = "chatbot_model.h5"

target_model_path = "chatbot_target_model.h5"

if os.path.exists(model_path) and os.path.exists(target_model_path) :

    model = load_model(model_path)
    target_model = load_model(target_model_path)
    print(" >>  Models loaded from file.")

else :
    model = build_model()
    target_model = build_model()

    target_model.set_weights( model.get_weights() ) 
    print( ">START< New models initialized." )



target_model.set_weights(model.get_weights())

epsilon=EPSILON

def text_to_state(question:str):
    lowq=question.lower()

    arr= []
    for kw in keywords:
        arr.append(1 if kw in lowq else 0)
    return np.array(arr)

def choose_action(state) :
    global epsilon
    if np.random.rand()<epsilon:
        
        return random.randint(0, ACTION_SIZE-1)
    
    qvals=model.predict(state.reshape(1,-1) ,verbose=0)

    return np.argmax(qvals[0])

def train_batch():

    if len(memory.memory)<BATCH_SIZE:
        return
    batch=memory.sample(BATCH_SIZE)
    
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for s,a,r,ns,d in batch :
        states.append(s)  
        actions.append(a)
        
        rewards.append(r)

        next_states.append(ns)

        dones.append(d)      
    states=np.array(states)
    next_states=np.array(next_states)
    qvals=model.predict(states,verbose=0)   

    qnext=target_model.predict(next_states ,verbose=0)
    for i in range(len(batch)):
        a=actions[i]
        r=rewards[i]
        done=dones[i]           

        if done:
            qvals[i][a]=r
        else:
            qvals[i][a]=r+GAMMA*np.max(qnext[i])
    model.fit(states,qvals,epochs=1,verbose=0)

def update_target() :
    target_model.set_weights(model.get_weights())

def auto_train(episodes=1000):
    global epsilon
    start_time = time.time() 

    for ep in range(episodes):
        if ep % 100 == 0:
            print(f"[EP {ep}] -> <- epsilon: {epsilon:.3f} ")       
    

        qtext, correct_act = random.choice(training_data)
        s= text_to_state(qtext)
        action=choose_action(s)   
        rew = +1 if (action==correct_act) else -1

        done=True
        ns = s  

        memory.store((s ,action,rew,ns ,done))
        train_batch()
        if ep%TARGET_UPDATE_FREQ==0 :
            update_target()
        epsilon=max( EPS_MIN, epsilon*EPS_DECAY )
        logging.info(f"[AutoTrain] ep={ep}, question='{qtext}', correct={correct_act}, chosen={action}, rew={rew}, eps={epsilon:.3f}")

    end_time =  time.time()  # czas trwania trenowania auto
    duration  =  end_time  -  start_time
    minutes  = int( duration // 60 )  
    seconds  = int(duration %  60)

    print(f" 🧞‍♂️🥋 Auto-training ended after {episodes} episodes. Epsilon={epsilon:.3f}")
    print(f"🕒 Training took {minutes} min {seconds} sec")
    logging.info(f"[Training] Duration: {duration:.2f} seconds ({minutes} min {seconds} sec)")


def user_chat() :
    global epsilon
    process = psutil.Process(os.getpid())
    print("\n 🧞‍♂️🥋 Now starting RL chatbot. Type 'exit' , 'quit' or 'bye chat' to exit it")
    ecount=0
    previous_responses = {}

    while True:
        user_q=input("\n🧑👩🇵🇱🇨🇿 user (question):")
        if user_q.lower() in ["exit","quit", "bye chat"] :
            print(" 🧞‍♂️🥋 See you, User, have a nice day!")
            break

        start_time = time.time()    
        cpu_before = process.cpu_percent(interval=None) 
        mem_before = process.memory_info().rss / (1024*1024) 

        s=text_to_state(user_q)  
        action=choose_action(s) 
        ans=actions_dict.get(action ,"???")     

        mem_after = process.memory_info().rss / (1024*1024)
        cpu_after = process.cpu_percent(interval=None)
        end_time = time.time() 

        elapsed = end_time - start_time
        length_resp = len( ans.split() )
        print(f" 🧞‍♂️🥋 ChatbotD: {ans}")    

        # automatyczna nagroda?     
        auto_rew=None
        for (qtxt,corr) in training_data:
            if user_q.lower()==qtxt.lower():
                auto_rew= +1 if (corr==action) else -1
                print(f"[AUTO] Matched training_data  => auto_rew={auto_rew} " )
                break

        user_fb=input("🎫feedback (+/-/0 to override)? ")

        if user_fb=='+':
            rew=+1

        elif user_fb=='-':
            rew=-1
        elif user_fb=='0':  
            rew=0  

        else:
            if auto_rew is not None :
                rew=auto_rew   

            else :
                rew=0
         
        done=True
        ns=s
        memory.store((s,action,rew,ns,done))
        train_batch()
        if ecount%TARGET_UPDATE_FREQ==0 :

            update_target()
        epsilon=max(EPS_MIN, epsilon*EPS_DECAY)


        was_repeated=False
        if user_q in previous_responses:

            if previous_responses[user_q] == ans:
                was_repeated=True

        previous_responses[user_q] = ans

        if ans=="I don't know" or ans=="???" :
            logging.warning(f"[WARNING] Possibly fallback or unknown answer for input='{user_q}'")

        logging.info(
            f"[UserChat] ecount={ecount}, question='{user_q}', action={action}, rew={rew} , "

            f"eps={epsilon:.3f}, TIME={elapsed:.4f}s , CPU=({cpu_before:.2f}%->{cpu_after:.2f}%) , "

            f"MEM=({mem_before:.2f}MB->{mem_after:.2f}MB), answer='{ans}', lenResp={length_resp} , Repeated={was_repeated} "
        )
        ecount+=1

def main() :
    auto_train(episodes=1000)

    user_chat()

if __name__=="__main__" :

    main()


    with open(EPSILON_FILE, "w") as f:
      
      f.write(str(epsilon))

    print(f" > Epsilon saved <: {epsilon:.3f}")

    model.save(model_path)
     
target_model.save(target_model_path)       

print(">><< Model saved")
