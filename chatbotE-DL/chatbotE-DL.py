import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import logging
import time
import psutil
import os



from datetime import datetime

logging.basicConfig(filename="seq2seq_chat.log", level=logging.DEBUG)

# qiuestions 

train_pairs = [

   # BASIC  FACTS> >>

   ("what is the capital of poland?", "warsaw"),
   ("how many days does february have in a leap year?", "29 days"),
       ("what is the name of university in hradec kralove?", "uhk (university of hradec kralove)"),
       ("when is new years day celebrated?", "january 1st"),

   ( "who painted the lady with an ermine?" , "leonardo da vinci" ) ,

   ("where is the painting called lady with an ermine located?" , "in poland (krakow)"),
   ("what is the language used in the czech republic?", "the czech language"),

   # PARAPRASED---->>>->
   ("warsaw is the capital of which country?", "poland"),        
       ("how many days does february contain in a leap year?", "29 days")  , 
           ("which university that teaches computer science is the biggest in hradec kralove?" , "uhk (university of hradec kralove)"),

       ("on which date do people celebrate new years day?", "january 1st")  ,
   ("who is the author of the painting lady with an ermine?"  , "leonardo da vinci"),
   ("in which country can you find the painting lady with an ermine?", "poland (krakow)"),
       ("which language is spoken in the czech republic ?", "czech"),  
   ("what is the capital of poland?"   , "warsaw") ,   


("capital of poland?"  ,  "warsaw" )  ,
("whats the capital of poland?"  , "warsaw") ,

("what's the capitol of poland?" , "warsaw" ) ,               
("warsaw is the capital of which country?", "poland" ),
("polands capital?", "warsaw") ,
("waht is the capital of poland?"  ,  "warsaw" ) ,
                 
("how many days does february have in a leap year?", "29 days")  ,

( "how many days in february leap year?" , "29 days" ) ,
("february days in leap year?" , "29 days"),

("how many days are in february in a leap year?", "29 days") , 
( "febrary in leap year has how many days?" , "29 days"),    
("februrary leap year?" , "29 days"),

("what is the name of university in hradec kralove?", "uhk (university of hradec kralove)") ,
("which university is in hradec kralove?", "uhk (university of hradec kralove)") ,

("uhk is where?"  , "in hradec kralove") , 
("name of cs university in hradec?" , "uhk (university of hradec kralove)"),
("whta is the name of univesrity in hradec kralove?" , "uhk (university of hradec kralove)"),

("when is new years day celebrated?", "january 1st") ,
("on what day is new years day?", "january 1st") ,

( "date of new year?"  , "january 1st"),
("new year celebration date?" , "january 1st"),        

("when is new year?", "january 1st" ),
("when is jan 1 celebrated?" , "january 1st"),
   
("who painted the lady with an ermine?", "leonardo da vinci") ,

("who is the author of lady with an ermine?" , "leonardo da vinci") ,
("author of lady with an ermine?", "leonardo da vinci") ,
 
("who created lady with an ermine?", "leonardo da vinci") ,       
("who was the painter of the lady with the ermine?", "leonardo da vinci"),
 
("who paintd the lady with an ermine?" , "leonardo da vinci") ,

          
("where is the painting called lady with an ermine located?", "in poland (krakow)"),
(" location of lady with an ermine? "  , "in poland (Krakow)"),

( "where is lady with an ermine kept? " , " in poland (krakow)" ),
("where to find lady with an ermine?" , "in poland (Cracow)") ,

( "where can i see lady with the ermine?", "in poland (krakow)" ) ,
("where is lady with ermine?" , "in poland (krakow)") ,

("what is the language used in the czech republic?" , "the czech language"),

("what language is spoken in czechia?" , "the czech language") ,

("official language of czech republic?", "the czech language") ,

( "which language do people speak in czech republic?", "czech language")  ,

("czech language?", "the czech language") ,

("wat  is the langauge in czech republic?" , "the czech language "),



   # out-of-scope (exampl) >
   ("what is the next winning lottery number?", "i do not know"),
   ("how many planets will be discovered in the future?", "i do not know"),
]


START_TOKEN = "<sos>"
END_TOKEN   = "<eos>"


questions = []
answers = []

for q, a in train_pairs:
    q_clean = re.sub(r"[^a-zA-Z0-9 ?]" , "" , q.lower())

    a_clean = re.sub(r"[^a-zA-Z0-9 ?]", "" , a.lower())

    questions.append(q_clean.strip())
    answers.append(f"{START_TOKEN} {a_clean.strip()} {END_TOKEN} ")

    all_text = questions + answers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq   = tokenizer.texts_to_sequences(answers)

max_len_q = max(len(seq) for seq in questions_seq )
max_len_a = max(len(seq) for seq in answers_seq )

max_len_q = min(max_len_q , 10)
max_len_a = min(max_len_a , 10 )

encoder_input_data = pad_sequences(questions_seq, maxlen=max_len_q, padding="post" )
decoder_input_data = pad_sequences(answers_seq, maxlen=max_len_a, padding="post" )

decoder_target_data = np.zeros_like(decoder_input_data)
for i, seq in enumerate(answers_seq):
        seq_np = np.array(seq)
        for t in range(len(seq_np)-1):
            decoder_target_data[i, t] = seq_np[t+1]

print("encoder_input_data.shape:" , encoder_input_data.shape)

print("decoder_input_data.shape:" , decoder_input_data.shape)

print("decoder_target_data.shape:" , decoder_target_data.shape)

embedding_dim = 50

latent_dim = 64

encoder_inputs = Input(shape=(max_len_q,))
enc_embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)

enc_outputs, state_h, state_c = encoder_lstm(enc_embed)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_len_a,))
dec_embed = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

dec_outputs , _, _ = decoder_lstm(dec_embed, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation="softmax")

decoder_outputs = decoder_dense(dec_outputs )

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam" , loss="sparse_categorical_crossentropy")

print(" Model summary : ")

model.summary()



## model.save("seq2seq_chatbot_model.h5")
##print(" model saved into file -> seq2seq_chatbot_model.h5")


encoder_model = Model(encoder_inputs, encoder_states)

dec_state_input_h = tf.keras.layers.Input(shape=(latent_dim,))

dec_state_input_c = tf.keras.layers.Input(shape=(latent_dim,))
dec_state_inputs = [dec_state_input_h, dec_state_input_c]

dec_embed2 = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)

dec_outputs2, state_h2, state_c2 = decoder_lstm(dec_embed2, initial_state=dec_state_inputs)
dec_states2 = [state_h2, state_c2]
dec_outputs2 = decoder_dense(dec_outputs2)
decoder_model = Model(

    [decoder_inputs]  +  dec_state_inputs ,
    [dec_outputs2]  +  dec_states2
)

index_word = {v: k for k, v in word_index.items()}

def decode_sequence_greedy(input_seq):
    states_val = encoder_model.predict(input_seq)
    target_seq = np.zeros((1 ,1))

    start_id = word_index.get("<sos>" , 1)
    target_seq[0,0] = start_id

    stop_condition = False
    decoded_sentence = ""
    prob_list = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_val)
        sampled_token_index = np.argmax(output_tokens[0 , -1 , :])



        sampled_prob = output_tokens[0, -1 , sampled_token_index]
        prob_list.append(sampled_prob)

        sampled_word = index_word.get(sampled_token_index, "?")
        if (sampled_word == "<eos>" or len(decoded_sentence.split())> max_len_a) :
            stop_condition = True

        else:
            decoded_sentence += " " + sampled_word

        target_seq[0,0] = sampled_token_index
        states_val = [h, c]

    avg_conf = float(np.mean(prob_list)) if prob_list else 0.0
    return decoded_sentence.strip(), avg_conf

def run_chat():
    print("\n*** Chatbot >> seq2seq LSTM ***")
    print("Type either : 'bye chat' or 'quit' or 'exit' to stop the chatbot")

    process = psutil.Process(os.getpid())
    previous_responses = {}

    while True:
        inp = input("\n👥👨👩You-human: ")

        if inp.lower() in ["quit","exit", "bye chat"]:

            print(" 🎭💭 Bye my dear User(s)!")
            break

        start_time = time.time()
        cpu_before = process.cpu_percent(interval=None)

        mem_before = process.memory_info().rss / ( 1024*1024 )

        seq = tokenizer.texts_to_sequences([inp.lower()])
        seq = pad_sequences(seq, maxlen=max_len_q, padding="post")
        
        result, conf = decode_sequence_greedy(seq)

        mem_after = process.memory_info().rss / (1024*1024)
        cpu_after = process.cpu_percent(interval=None)
        end_time = time.time()

        elapsed = end_time - start_time
        length_resp = len(result.split())

        print(f" 🎭💭 Bot: {result} [conf={conf:.4f}]")

        was_repeated = False  
        if inp in previous_responses:
            if previous_responses[inp] == result :
                was_repeated = True
        previous_responses[inp] = result

        # warning if  conf < 0.1 (heuristic example)
        if conf < 0.1 :
            
            logging.warning(f"[WARNING] Low confidence answer for input='{inp}' (conf={conf:.4f})" )

        logging.info(
            f"{datetime.now()} | QUESTION='{inp}' | ANSWER='{result}' | CONFIDENCE={conf:.4f} | "

            f"TIME={elapsed:.4f}s | CPU=({cpu_before:.2f}%->{cpu_after:.2f}%) | "

            f"MEM=({mem_before:.2f}MB->{mem_after:.2f}MB) | lenResp={length_resp} | Repeated={was_repeated}"
        )


if __name__=="__main__" :
    
    total_start = time.time()


#========-=========- --training ---- 


# 
#MODEL_FILE = "seq2seq_chatbot_model.h5"

WEIGHTS_FILE = "seq2seq_weights.h5"

# train_model = not os.path.exists(MODEL_FILE)

train_model = not os.path.exists(WEIGHTS_FILE)

if train_model:
    print(" train the seq2seq model now... ")

    model.fit(

        [encoder_input_data, decoder_input_data] ,
        np.expand_dims(decoder_target_data , -1 ) ,
        batch_size=16,

        epochs=1000  
    )

   # model.save(MODEL_FILE)
    model.save_weights(WEIGHTS_FILE)

    print(">model saved (weights) <")

else:   
        print(" model existing already. load existing model in progress...")
    #    model = tf.keras.models.load_model(MODEL_FILE)
        model.load_weights( WEIGHTS_FILE )
        print(" model succesfully loaded ")


# =--------------==-----

total_end = time.time()
duration = total_end - total_start
mins = int( duration // 60 )
secs = int( duration % 60 )
print(f"Full time of preparation is: {mins} min {secs} sec")
print("training is done,now->> you can chat!! ")
run_chat()