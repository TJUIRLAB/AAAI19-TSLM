import tensorflow as tf
from  PTBIn_put import  PTBInput,get_config
import util
from tensorRAC import TensorRAC

class ComplexModel(object):
	def __init__(self,is_training,config,input_,FLAGS):
		self.FLAGS = FLAGS
		self._is_training = is_training
		self._input = input_
		self._run_params = None 
		self._cell = None
		self.batch_size = input_.batch_size
		self.num_steps = input_.num_steps
		size = config.hidden_size
		self.vocab_size=vocab_size = config.vocab_size
		
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding",[vocab_size,size],dtype=tf.float32)
			inputs = tf.nn.embedding_lookup(embedding,input_.input_data)
		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs,config.keep_prob)
		output,state = self.build_graph(inputs,config,is_training)

		softmax_w = tf.get_variable("softmax_w",[size,vocab_size],dtype=tf.float32)
		softmax_b = tf.get_variable("softmax_b",[vocab_size],dtype=tf.float32)
		logits = tf.nn.xw_plus_b(output,softmax_w,softmax_b)
		logits = tf.reshape(logits,[self.batch_size,self.num_steps,vocab_size])
		loss = tf.contrib.seq2seq.sequence_loss(logits,input_.targets,tf.ones([self.batch_size,self.num_steps],dtype=tf.float32),
												average_across_timesteps = False,
												average_across_batch = True)
		

		self._cost = tf.reduce_sum(loss)
 
		self._final_state = state
		if not is_training:
			return 
		self._lr = tf.Variable(0.0,trainable = False)
		tvars = tf.trainable_variables()

		grads,_ = tf.clip_by_global_norm(tf.gradients(self._cost,tvars),config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		#optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads,tvars),global_step = tf.train.get_or_create_global_step())
		self._new_lr =  tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)
		
	# def Get_loss(self,logits,labels):
	# 	self.Maxtix = tf.get_variable("Matrix",[self.vocab_size,self.vocab_size],dtype=tf.float32)
	# 	logits = tf.matmul(tf.nn.softmax(logits),self.Maxtix)
	# 	result = -tf.reduce_sum(labels*tf.log(logits),1)
	# 	return result


	def build_graph(self,inputs,config,is_training):
		# if config.model == "cudnn":
		# 	return self._build_rnn_graph_cudnn(inputs, config, is_training)
		# if config.model == "block":
		# 	return self._build_rnn_graph__bloclstm(inputs,config,is_training)
		# if config.model =="basic":
		# 	return self._build_rnn_graph__basiclstm(inputs,config,is_training)
		if config.model == "tlstm":
			return self._build_rnn_TensorRAC(inputs,config,is_training)
		if config.model == "complex":
			return 


	def _build_rnn_graph_cudnn(self, inputs, config, is_training):
		inputs = tf.transpose(inputs,[1,0,2])
		self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
					num_layers = config.num_layers,
					num_units = config.hidden_size,
					dropout = 1 - config.keep_prob if is_training else 0)
		params_size_t = self._cell.params_size()
		self._run_params = tf.get_variable("lstm_params",initializer=tf.random_uniform([params_size_t],-config.init_scale,config.init_scale),validate_shape=False)
		c = tf.zeros([config.num_layers,self.batch_szie,config.hidden_size],tf.float32)
		h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
					tf.float32)
		self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h,c=c),)
		outpus,h,c = self._cell(inputs,h,c,self._run_params,is_training)
		outputs = tf.transpose(outputs,[1,0,2])
		outputs = tf.reshape(outputs,[-1,config.hidden_size])
		return outputs,(tf.contrib.rnn.LSTMStateTuple(h=h,c=c),)
		
	def _build_rnn_graph__bloclstm(self,inputs,config,is_training):
		cell =  tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,reuse=not is_training)
		if is_training and config.keep_prob < 1:
			cell  = tf.contrib.rnn.DropoutWrapper(
				cell,output_keep_prob = config.keep_prob)
		cell = tf.contrib.rnn.MultiRNNCell(
				[cell for _ in range(config.num_layers)],state_is_tuple=True)
		self._initial_state = cell.zero_state(config.batch_size,dtype=tf.float32)
		state = self._initial_state
		outputs = []
		with tf.variable_scope("RNN"):
			for time_step in range(self.num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output,state) = cell(inputs[:,time_step,:],state)
				outputs.append(cell_output)
		output = tf.reshape(tf.concat(outputs,1),[-1,config.hidden_size])
		return output,state

	def _build_rnn_graph__basiclstm(self,inputs,config,is_training):
		cell = tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
		if is_training and config.keep_prob < 1:
			cell  = tf.contrib.rnn.DropoutWrapper(
				cell,output_keep_prob = config.keep_prob)
		cell = tf.contrib.rnn.MultiRNNCell(
				[cell for _ in range(config.num_layers)],state_is_tuple=True)
		self._initial_state = cell.zeros_state(config.batch_size,dtype=tf.float32)
		state = self._initial_state
		outputs = []
		with tf.variable_scope("RNN"):
			for time_step in range(self.num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				(cell_output,state) = cell(inputs[:,time_step,:],state)
				outputs.append(cell_output)
		output = tf.rehsape(tf.concat(outputs,1),[-1,config.hidden_size])
		return output,state
	def _build_rnn_TensorRAC(self,inputs,config,is_training):
		def make_cell():
			cell = TensorRAC(config.hidden_size,reuse=not is_training)
			if is_training and config.keep_prob < 1:
				cell  = tf.contrib.rnn.DropoutWrapper(
						cell,output_keep_prob = config.keep_prob)
			return cell

		cell = tf.contrib.rnn.MultiRNNCell(
		 		[make_cell() for _ in range(config.num_layers)],state_is_tuple=True)
		self._initial_state = cell.zero_state(config.batch_size,tf.float32)

		state = self._initial_state
		outputs = []
		with tf.variable_scope("RAC"):
			for time_step in range(self.num_steps):
				if time_step > 0: 
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)
		output = tf.reshape(tf.concat(outputs,1),[-1,config.hidden_size])
		return output,state

	def assign_lr(self,session,lr_value):
		session.run(self._lr_update,feed_dict={self._new_lr:lr_value})
	
	def export_ops(self,name):
		self._name = name
		ops = {util.with_prefix(self._name,"cost"):self._cost}
		if self._is_training:
			ops.update(lr=self._lr,new_lr=self._new_lr,lr_update=self._lr_update)
			if self._run_params:
				ops.update(run_params=self._run_params)
		for name,op in ops.items():
			tf.add_to_collection(name,op)
		self._initial_state_name = util.with_prefix(self._name,"initial")
		self._final_state_name = util.with_prefix(self._name,"final")
		util.export_state_tuples(self._initial_state,self._initial_state_name)
		util.export_state_tuples(self._final_state, self._final_state_name) 
	def import_ops(self):
		if self._is_training:
			self._train_op = tf.get_collection_ref("train_op")[0]
			self._lr = tf.get_collection_ref("lr")[0]
			self._new_lr = tf.get_collection_ref("new_lr")[0]
			self._lr_update = tf.get_collection_ref("lr_update")[0]
			rnn_params = tf.get_collection_ref("rnn_params")
			if self._cell and rnn_params:
				params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
					            self._cell,
					            self._cell.params_to_canonical,
					            self._cell.canonical_to_params,
					            rnn_params,
					            base_variable_scope="Model/RNN")
				tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
		self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
		num_replicas = self.FLAGS.num_gpus if self._name == "Train" else 1
		self._initial_state = util.import_state_tuples(
				self._initial_state, self._initial_state_name, num_replicas)
		self._final_state = util.import_state_tuples(
				self._final_state, self._final_state_name, num_replicas)

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def initial_state_name(self):
		return self._initial_state_name

	@property
	def final_state_name(self):
		return self._final_state_name