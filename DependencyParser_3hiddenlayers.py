import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.

            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()

            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            """
            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size, parsing_system.numTransitions()])
            self.test_inputs = tf.placeholder(tf.int32, shape=[Config.n_Tokens])
            """    
            2) Call forward_pass and get predictions
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)
            """
            # (test_embed, weights_input, bias_input, weights_output)
            embed_temp = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embeddings = tf.reshape(embed_temp, [Config.batch_size, -1])
            # computed weights input
            weights_input_1 = tf.Variable(
                tf.random_normal(shape=[Config.n_Tokens * Config.embedding_size, Config.hidden_size],
                                 stddev=0.14))

            # computed second layer weight
            weights_input_2 = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.hidden_size],
                                 stddev=0.14))

            # computed third layer weight
            weights_input_3 = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, Config.hidden_size],
                                 stddev=0.14))

            # initialized bias_input for layer 2
            bias_input_1 = tf.Variable(tf.zeros([Config.hidden_size]))

            # initialized bias_input for layer 2
            bias_input_2 = tf.Variable(tf.zeros([Config.hidden_size]))

            # initialized bias_input for layer 3
            bias_input_3 = tf.Variable(tf.zeros([Config.hidden_size]))

            # computed weights output
            weights_output = tf.Variable(
                tf.random_normal(shape=[Config.hidden_size, parsing_system.numTransitions()], stddev=0.14))

            # called forward pass and computed output of first layer which is output to second layer
            embeddings_layer_1 = self.forward_pass(embeddings, weights_input_1, bias_input_1, weights_output, 1)

            embeddings_layer_2 = self.forward_pass(embeddings_layer_1, weights_input_2, bias_input_2,
                                                   weights_output, 2)

            self.train_prediction = self.forward_pass(embeddings_layer_2, weights_input_3, bias_input_3,
                                                      weights_output, 3)

            # added non-linearity
            train_labels = tf.nn.relu(self.train_labels)

            """
            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam

            ...
            self.loss = 

            ===================================================================
            """
            ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.train_prediction, labels=train_labels)

            # multipled lambda with theta
            l2_loss_weights_1 = Config.lam * tf.nn.l2_loss(weights_input_1)
            l2_loss_weights_2 = Config.lam * tf.nn.l2_loss(weights_input_2)
            l2_loss_weights_3 = Config.lam * tf.nn.l2_loss(weights_input_3)
            sum_weights = l2_loss_weights_1 + l2_loss_weights_2 + l2_loss_weights_3

            l2_loss_bias_1 = Config.lam * tf.nn.l2_loss(bias_input_1)
            l2_loss_bias_2 = Config.lam * tf.nn.l2_loss(bias_input_2)
            l2_loss_bias_3 = Config.lam * tf.nn.l2_loss(bias_input_3)
            sum_bias = l2_loss_bias_1 + l2_loss_bias_2 + l2_loss_bias_3

            l2_loss_embed = Config.lam * tf.nn.l2_loss(embeddings)
            l2_loss_output = Config.lam * tf.nn.l2_loss(weights_output)

            total_ce_l2_loss = ce_loss + sum_weights + \
                               sum_bias + l2_loss_output + l2_loss_embed

            self.loss = tf.reduce_mean(total_ce_l2_loss)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)

            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            test_y1 = self.forward_pass(test_embed, weights_input_1, bias_input_1, weights_output, 1)
            test_y2 = self.forward_pass(test_y1, weights_input_2, bias_input_2, weights_output, 2)
            self.test_pred = self.forward_pass(test_y2, weights_input_3, bias_input_3, weights_output, 3)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)

    def forward_pass(self, embed, weights_input, biases_input, weights_output, pass_layer):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        # h = (W1wxw + W1txt + W1lxl + b1)
        product1 = tf.matmul(embed, weights_input)

        temp_sum = tf.add(product1, biases_input)
        h = tf.pow(temp_sum, 3.0)

        # Uncomment to change activation function to sigmoid

        #h = tf.nn.sigmoid(temp_sum)

        # Uncomment to change activation funtion to tanh

        #h = tf.tanh(temp_sum)

        # Uncomment to change activation function to ReLu

        #h = tf.nn.relu(temp_sum)

        if pass_layer == 3:
            h = tf.matmul(h, weights_output)
        return h


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):
    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    features = []
    for i in range(3):
        features.append(c.getStack(i))

    for i in range(3):
        features.append(c.getBuffer(i))

    for i in range(2):
        left = c.getLeftChild(features[i], 1)
        right = c.getRightChild(features[i], 1)
        secondleft = c.getLeftChild(features[i], 2)
        secondright = c.getRightChild(features[i], 2)
        leftofleft = c.getLeftChild(left, 1)
        rightofright = c.getRightChild(right, 1)
        features.extend([left, right, secondleft, secondright, leftofleft, rightofright])  # see to use append

    n = len(features)
    for i in range(n):
        features.append(c.getPOS(features[i]))

    for i in range(6, 18):
        features.append(c.tree.getLabel(features[i]))

    for i in range(18):
        features[i] = getWordID(c.getWord(features[i]))

    for i in range(18, 36):
        features[i] = getPosID(features[i])

    for i in range(36, 48):
        features[i] = getLabelID(features[i])

    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)
