import tensorflow as tf
import numpy as np
import time, string, random
from tensorflow.models.rnn.ptb import reader
from myLSTM_utils import * 
#=================make embedding=====================
rawData=readtext('allLyrics.txt')
charSet = set(rawData)
charSetSize=len(charSet)
id2char = {i:c for i,c in enumerate(charSet)}
char2id = {c:i for i,c in enumerate(charSet)}
corpus=[char2id[c] for c in rawData]			# build corpus using the char2id embedding
#==========================================
# need reset graph, otherwise will gives variable reuse exception
def reset_graph():	
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()
#==========================================
graphRNN = myrnnLSTM(len_charSet=charSetSize)
saveAtDIR='./saves/'+time.strftime("%m%d_%H%M")+'/myTestModel'
# the trained model will be saved at ./saves/mondate_hourmin/
total_epochs=10
loss = trainLSTM(graphRNN,corpus,num_epochs=total_epochs, saveAt=saveAtDIR)
#=====================================generating characters==================================
reset_graph()
graph = myrnnLSTM(len_charSet=charSetSize,hp_num_steps=1, hp_batch_size=1)
genLyric(g=graph, first=random.choice(string.letters), pick_top_chars=3, checkpoint="./saves/1024_2230/myTestModel", num_chars=300, char2id=char2id, id2char=id2char, charSetSize=charSetSize)




		
		

		
		
		
		
		
		

