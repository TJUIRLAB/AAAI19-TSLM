import time 
import numpy 
import tensorflow as tf
import reader 
import util
from tensorflow.python.client import device_lib
from PTBIn_put import PTBInput,get_config
#from Model import ComplexModel
from Model import ComplexModel
import numpy as np

flags = tf.flags
flags.DEFINE_string("data_path","data","where the Training/test data is stored.")
flags.DEFINE_string("save_path","checkpoint/","Model output directory.")
flags.DEFINE_integer("num_gpus",1,"GPU choose.")
flags.DEFINE_string("rnn_model",None,"+++++++++++++")
FLAGS = flags.FLAGS


def run_epoch(session,model,eval_op=None,verbose=False):
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)
	fetches = {"cost":model.cost,
		"final_state":model.final_state,
		}
	if eval_op is not None:
		fetches["eval_op"] = eval_op
	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i,s in enumerate(model.initial_state):
			feed_dict[s] = state[i]  
			#feed_dict[h] = state[i].h

		vals = session.run(fetches,feed_dict)
		cost = vals["cost"]
		state =  vals["final_state"] 

		costs += cost 
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
				(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
				iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
				(time.time() - start_time)))
	return np.exp(costs / iters)

	
def main(_):
	raw_data = reader.ptb_raw_data(FLAGS.data_path)
	train_data, valid_data,test_data, _ = raw_data
	config = get_config()
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.batch_size = 1
	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

		with tf.name_scope("Train"):
			train_input = PTBInput(config=config,data=train_data,name="TrainInput")
			with tf.variable_scope("Model",reuse= None,initializer= initializer):
				m = ComplexModel(is_training=True,config=config,input_=train_input,FLAGS=FLAGS)
			tf.summary.scalar("Training Loss",m.cost)
			tf.summary.scalar("Learning Rate",m.lr)
		with tf.name_scope("Valid"):
			valid_input = PTBInput(config=config,data=train_data,name="ValidInput")
			with tf.variable_scope("Model",reuse=True,initializer=initializer):
				mvalid =  ComplexModel(is_training=False, config=config, input_=valid_input,FLAGS=FLAGS)
			tf.summary.scalar(" Validation Loss ",mvalid.cost)
		with tf.name_scope("Test"):
			test_input = PTBInput(config=eval_config,data =  test_data,name="TestInput")
			with tf.variable_scope("Model",reuse=True,initializer=initializer):
				mtest = ComplexModel(is_training=False,config = eval_config,input_=test_input,FLAGS=FLAGS)
		models = {"Train":m,"Valid":mvalid,"Test":mtest}
		for name,model in models.items():
			model.export_ops(name)
		metagraph = tf.train.export_meta_graph()
		soft_placement = False
		if FLAGS.num_gpus > 1:
			soft_placement = True
			util.auto__parallel(metagraph,m)

	with tf.Graph().as_default():
		tf.train.import_meta_graph(metagraph)
		for model in models.values():
			model.import_ops()
		sv = tf.train.Supervisor(logdir=FLAGS.save_path)

		config_proto = tf.ConfigProto()
		config_proto.gpu_options.allow_growth=True
		with sv.managed_session(config=config_proto) as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i+1-config.max_epoch,0.0)
				m.assign_lr(session,config.learning_rate*lr_decay)

				print("Epoch: %d Learning rate: %.3f"%(i+1,session.run(m.lr)))
				train_perplexity = run_epoch(session,m,eval_op=m.train_op,verbose=True)
				print("Epoch: %d Train Perplexity:%3f"%(i+1,train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
				if i % 5 == 0:
					test_perplexity = run_epoch(session,mtest)
					print("Test Perplexity:%.3f" % test_perplexity)
			test_perplexity = run_epoch(session,mtest)
			print("Test Perplexity:%.3f" % test_perplexity)
			if FLAGS.save_path:
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session,FLAGS.save_path,global_step=sv.global_step)






if __name__ == '__main__':
	
	tf.app.run() 	 
