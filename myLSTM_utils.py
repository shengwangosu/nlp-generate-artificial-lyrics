import tensorflow as tf
import numpy as np
import os
from tensorflow.models.rnn.ptb import reader
def readtext(fname):
	with open(fname,'r') as f:
		rawData = f.read()
	print "Length of raw data: " +str(len(rawData))
	print "Num of lyrics: " + str(len(rawData.rsplit('\n\n')))
	return rawData
#===========================================================================================================
def fetch_epochs(corpus, num_epochs, num_unrolls, batch_size):
	for t in range(num_epochs):
		yield reader.ptb_iterator(corpus, batch_size, num_unrolls)
	
##==========================================================================================================
def myrnnLSTM(len_charSet=100,hp_num_steps=200, hp_batch_size=32):
	## hyper-parameter for LSTM RNN
	hp_num_hidden= 100
  	hp_learning_rate=0.001
	#hp_display_step=100
	hp_num_layers=2
	x = tf.placeholder(tf.int32,[hp_batch_size, hp_num_steps])	## input
	y = tf.placeholder(tf.int32,[hp_batch_size, hp_num_steps])	## correct output
	W = tf.Variable(tf.random_normal([hp_num_hidden, len_charSet])) ## weight
	b = tf.Variable(tf.zeros([len_charSet]))		## bias
	# input comes from x
	embedding = tf.get_variable('Embedding_matrix',[len_charSet, hp_num_hidden])
	inputs= tf.nn.embedding_lookup(embedding, x)
	## define lstm cell
	#lstm_cell= tf.nn.rnn_cell.BasicLSTMCell(hp_num_hidden, forget_bias=0.0)
	lstm_cell= tf.nn.rnn_cell.LSTMCell(hp_num_hidden, state_is_tuple=True)
	## multiple rnn cell wrapper
	cell  = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * hp_num_layers)
	init_state = cell.zero_state(hp_batch_size, tf.float32)
	## make an update
	outputs, final_state = 	tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
	outputs=tf.reshape(outputs,[-1, hp_num_hidden])
	correct_output =tf.reshape(y, [-1])
	## give pred
	logits=tf.matmul(outputs, W) + b
	predictions = tf.nn.softmax(logits) 
	cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, correct_output))
	train = tf.train.AdamOptimizer(learning_rate=hp_learning_rate).minimize(cost)
	
	return dict(
		x=x,
		y=y,
		init_state=init_state,
		final_state=final_state,
		total_loss=cost,
		train_step=train,
		saver = tf.train.Saver(),
		preds = predictions
	)
#===================================================================================================================
def trainLSTM(graphRNN, corpus,num_epochs=3, num_steps=200, batch_size=32, saveAt=False):
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		losses=[]
		for i, epoch in enumerate(fetch_epochs(corpus, num_epochs,num_steps, batch_size)):
			temp_loss=0
			steps=0
			state_out=None
			for X, Y in epoch:
				steps +=1
				feed_dict={graphRNN['x']:X, graphRNN['y']:Y}
				if state_out is not None:
					feed_dict[graphRNN['init_state']]=state_out
				loss_out, state_out, _ = sess.run([graphRNN['total_loss'],graphRNN['final_state'],graphRNN['train_step']],feed_dict)
				temp_loss+= loss_out
				print "Training loss of "+ str(i) + " epoch: " + str(temp_loss/steps)
		if isinstance(saveAt, str):
			if not os.path.exists('./saves/'+saveAt.split('/')[2]):
				os.makedirs('./saves/'+saveAt.split('/')[2])
			graphRNN['saver'].save(sess, saveAt) 
	return losses
#===================================================================================================================
def genLyric(g, char2id, id2char, charSetSize, checkpoint="./saves/myTestModel", num_chars=300, first='A', pick_top_chars=None):
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		g['saver'].restore(sess, checkpoint)
		state = None
		current_char = char2id[first]
		#print current_char
		chars = [current_char]
		for i in range(num_chars):
			if state is not None:
				feed_dict={g['x']: [[current_char]], g['init_state']: state}
			else:
				feed_dict={g['x']: [[current_char]]}
			preds, state = sess.run([g['preds'],g['final_state']], feed_dict)
			if pick_top_chars is not None:
				p = np.squeeze(preds)
				p[np.argsort(p)[:-pick_top_chars]] = 0
				p = p / np.sum(p)
				current_char = np.random.choice(charSetSize, 1, p=p)[0]
			else:
				current_char = np.random.choice(charSetSize, 1, p=np.squeeze(preds))[0]
			chars.append(current_char)
	chars = map(lambda x: id2char[x], chars)
	print("".join(chars))
	return("".join(chars))
