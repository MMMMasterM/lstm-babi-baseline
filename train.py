import tensorflow as tf
import os, pickle
import numpy as np

#load dictionary
with open(os.path.join('processeddata', 'dictionary.txt'), 'rb') as f:
    wordIndices = pickle.load(f)
dictSize = len(wordIndices) + 1#padding entry with index 0 is not listed in wordIndices

#load data
with open(os.path.join('processeddata', 'train.txt'), 'rb') as f:
    trainingData = pickle.load(f)

with open(os.path.join('processeddata', 'valid.txt'), 'rb') as f:
    validationData = pickle.load(f)

#training parameters
batch_size = 32
epoch_count = 10

#utility functions
def getIndices(dataset, epochs):#generate indices over all epochs
    for epoch in range(epochs):
        for index in np.random.permutation(len(dataset)):
            yield index

def getBatchIndices(dataset, epochs):#generate batches of indices
    batchIndices = []
    for index in getIndices(dataset, epochs):
        batchIndices.append(index)
        if len(batchIndices) >= batch_size:
            yield batchIndices
            batchIndices = []

def getBatches(dataset, epochs):#generate batches of data
    for batchIndices in getBatchIndices(dataset, epochs):
        samples = [dataset[i] for i in batchIndices]
        contextLengths = [len(context) for context, question, answer in samples]
        maxContextLen = max(contextLengths)
        questionLengths = [len(question) for context, question, answer in samples]
        maxQuestionLen = max(questionLengths)
        #samples = [(context + [0]*(maxContextLen - len(context)), (question + [0]*(maxQuestionLen - len(question))), answer) for context, question, answer in samples]#append zero padding to make all samples' contexts/questions length maxContextLen/maxQuestionLen
        contextInput = [context + [0]*(maxContextLen - len(context)) for context, question, answer in samples]
        questionInput = [question + [0]*(maxQuestionLen - len(question)) for context, question, answer in samples]
        answerInput = [answer for context, question, answer in samples]
        yield contextInput, contextLengths, questionInput, questionLengths, answerInput

def buildModel():
    #model parameters
    embeddingDimension = 32
    qLstmHiddenUnits = 32
    cLstmHiddenUnits = 32

    inputContext = tf.placeholder(tf.int32, shape=(batch_size, None))
    inputContextLengths = tf.placeholder(tf.int32, shape=(batch_size,))
    inputQuestion = tf.placeholder(tf.int32, shape=(batch_size, None))
    inputQuestionLengths = tf.placeholder(tf.int32, shape=(batch_size,))

    #convert word indices to embedded representations (using learnable embeddings rather than one-hot vectors here)
    wordEmbedding = tf.Variable(tf.random_uniform(shape=[dictSize, embeddingDimension], minval=-1, maxval=1, seed=7))
    embeddedContext = tf.nn.embedding_lookup(wordEmbedding, inputContext)#shape=(batch_size, seq_len, embeddingDimension)
    embeddedQuestion = tf.nn.embedding_lookup(wordEmbedding, inputQuestion)

    #setup question LSTM
    questionLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=qLstmHiddenUnits)
    questionLSTMoutputs, _ = tf.nn.dynamic_rnn(questionLSTMcell, embeddedQuestion, dtype=tf.float32, scope="questionLSTM")#shape=(batch_size, seq_len, qLstmHiddenUnits)
    #extract final states at the end of each sample's sequence
    inputQuestionMaxLength = tf.reduce_max(inputQuestionLengths)
    questionLSTMoutputs = tf.reshape(questionLSTMoutputs, shape=(-1, qLstmHiddenUnits))
    qSeqEndSelector = tf.range(batch_size) * inputQuestionMaxLength + (inputQuestionLengths - 1)
    questionLSTMoutputs = tf.gather(questionLSTMoutputs, qSeqEndSelector)#shape=(batch_size, qLstmHiddenUnits)

    #setup context LSTM
    inputContextMaxLength = tf.reduce_max(inputContextLengths)
    #broadcast the questionLSTMoutputs to all timesteps
    questionLSTMoutputs = tf.expand_dims(questionLSTMoutputs, axis=1)#add time axis
    questionLSTMoutputs = tf.tile(questionLSTMoutputs, [1, inputContextMaxLength, 1])#repeat along time axis
    contextLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=cLstmHiddenUnits)
    contextLSTMoutputs, _ = tf.nn.dynamic_rnn(contextLSTMcell, tf.concat([embeddedContext, questionLSTMoutputs], axis=2), dtype=tf.float32, scope="contextLSTM")#shape=(batch_size, seq_len, cLstmHiddenUnits)
    #extract final states at the end of each sample's sequence
    contextLSTMoutputs = tf.reshape(contextLSTMoutputs, shape=(-1, cLstmHiddenUnits))
    cSeqEndSelector = tf.range(batch_size) * inputContextMaxLength + (inputContextLengths - 1)
    contextLSTMoutputs = tf.gather(contextLSTMoutputs, cSeqEndSelector)#shape=(batch_size, cLstmHiddenUnits)

    #tf.nn.softmax removed because it's applied afterwards by the built-in loss function softmax_cross_entropy_with_logits
    #TODO: make a second output WITH softmax for validation/testing - not necessary while correctness is determined by argmax though due to monotonicity of softmax
    answer1 = tf.contrib.layers.fully_connected(contextLSTMoutputs, dictSize, activation_fn=tf.nn.relu)#tf.nn.softmax)#shape=(batch_size, dictSize)
    answer2 = tf.contrib.layers.fully_connected(contextLSTMoutputs, dictSize, activation_fn=tf.nn.relu)#tf.nn.softmax)
    answer3 = tf.contrib.layers.fully_connected(contextLSTMoutputs, dictSize, activation_fn=tf.nn.relu)#tf.nn.softmax)
    answerGates = tf.contrib.layers.fully_connected(contextLSTMoutputs, 3, activation_fn=tf.sigmoid)#shape=(batch_size, 3)
    answerStack = tf.stack([answer1, answer2, answer3], axis=1)#stack shape=(batch_size, 3, dictSize)
    answer = tf.reduce_sum(tf.multiply(answerStack, tf.expand_dims(answerGates, axis=2)), axis=1)

    return inputContext, inputContextLengths, inputQuestion, inputQuestionLengths, answer, answerGates

(inputContext, inputContextLengths, inputQuestion, inputQuestionLengths, answer, answerGates) = buildModel()

inputAnswer = tf.placeholder(tf.float32, shape=(batch_size, dictSize))#label
#loss = tf.losses.mean_squared_error(labels=inputAnswer, predictions=answer) * dictSize - tf.reduce_mean(tf.square(answerGates - 0.5)) + 0.25#regularization term to enforce gate values close to 0 or 1
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=answer, labels=inputAnswer)) - tf.reduce_mean(tf.square(answerGates - 0.5)) + 0.25#regularization term to enforce gate values close to 0 or 1
#softmax_cross_entropy_with_logits is not suitable for outputs that are not probability distributions (which might be a problem for multi-answer questions) - still gives surprisingly good results for a first attempt
optimizer_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

def runValidation():
    #correct = tf.reduce_min(tf.cast(tf.equal(inputAnswer, tf.round(answer)), dtype=tf.float32), axis=1)#bad results since the max entries often don't achieve 0.5 so rounding doesnt work
    correct = tf.cast(tf.equal(tf.argmax(inputAnswer, axis=1), tf.argmax(answer, axis=1)), dtype=tf.float32)#this is incorrect for multi-answer questions but gives better answers than rounding on single-answer questions -> TODO: find good solution for multi-answer questions
    #idea for better implementation of "correct"-variable: take argmax of answer1, answer2, answer3 each, also round answerGates and then calculate "answer" similar as in "buildModel()" and finally check tf.equal
    accuracy = tf.reduce_mean(correct)
    total_acc = []
    for task_name in validationData:
        print("validating " + task_name)
        acc = []
        for i, (contextInput, contextLengths, questionInput, questionLengths, answerInput) in enumerate(getBatches(validationData[task_name], 1)):
            #print("validation batch " + str(i))
            feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput}
            batchAcc = sess.run(accuracy, feed_dict)
            acc.append(batchAcc)
            # print("label")
            # print(answerInput)
            # print("prediction")
            # print(sess.run(answer, feed_dict))
            # print("correct")
            # print(sess.run(correct, feed_dict))
            # print("gates")
            # print(sess.run(answerGates, feed_dict))
        taskAcc = sum(acc) / len(acc)
        print("task accuracy " + str(taskAcc))
        total_acc.append(taskAcc)
    print("total accuracy " + str(sum(total_acc) / len(total_acc)))

for i, (contextInput, contextLengths, questionInput, questionLengths, answerInput) in enumerate(getBatches(trainingData, epoch_count)):
    feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput}
    sess.run(optimizer_op, feed_dict=feed_dict)
    lossVal = sess.run(loss, feed_dict=feed_dict)
    if (i % 50 == 0):
        print("batch " + str(i))
        print("loss " + str(lossVal))
    if (i % 1000 == 999):
        runValidation()
