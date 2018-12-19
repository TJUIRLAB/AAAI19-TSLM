
import reader
class PTBInput(object):
	def __init__(self,config,data,name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data)//batch_size)-1)//num_steps 
		self.input_data,self.targets = reader.ptb_producer(data,batch_size,num_steps,name=name)
class get_config(object):	
	init_scale = 0.1
	num_layers = 2
	learning_rate = 0.5
	
	max_grad_norm = 5
	num_steps = 35
	hidden_size = 512
	max_epoch = 30
	max_max_epoch = 100
	keep_prob = 0.5
	lr_decay = 0.5
	batch_size = 32
	vocab_size = 10000
	model = "tlstm"
	