import tensorflow as tf

class Classifier:


    def __init__(self,model_fn,model_dir,check_dir=None,params=None):
        if not model_fn:
            raise ValueError('no model fn')
        self._model_fn = model_fn
        if not model_dir:
            raise ValueError('no place to save the model')
        self._model_dir = model_dir
        self._check_dir = check_dir
        self._params = copy.deepcopy(params or {})


    def _call_model_fn(self,features,labels,mode,config=None):
        return self._model_fn(features=features,labels=labels,mode=mode,params=config)



    def _get_restore_variabels(exclusions=None):
        if not exclusions:
            return slim.get_model_variables()
        exclusions = []

        exclusions = [scope.strip()
            for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        return variables_to_restore



    def train(self,features,labels,params=None):
        with tf.Graph().as_default_graph():
            loss, train_op = _call_model_fn(features,labels,'training')
            with tf.Session as sess:
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                variabels_to_restore = _get_restore_variabels(params['exclution'])
                if self._check_dir:
                    restorer = tf.train.saver(variabels_to_restore)
                    restorer.restore(sess,self_che)
                if 'num_of_steps' not in params:
                    num_of_steps = 18000
                for step in range(num_of_steps):
                    _,loss = sess.run([train_op,total_loss])
                    mean_loss += loss
                    if (step+1) % 20 == 0:
                        left_seconds = (time.time()-start_time)/step * (FLAGS.max_number_of_steps - step)
                        tf.logging.info('step: {}, loss: {}, time left: {}'.format(step,mean_loss/20,time.strftime('%H:%M:%S',time.gmtime(left_seconds))))
                        mean_loss = 0
                    if (step+1) % 2000 == 0:
                        saver.save(sess,'%s/model.ckpt'%FLAGS.train_dir,global_step=step+1)
        return self



    def predict():
        pass


    def evaluate():
        pass
