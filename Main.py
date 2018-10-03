from __future__ import print_function, division
from Objects.Segmentation import Segmentacao
import pickle
import tensorflow as tf
import numpy as np

def modeloLogistico ( X, nClasses, alpha, dropOut, isTraining ) :
    """


    :param X:
    :param nClasses:
    :param alpha:
    :param dropOut:
    :param isTraining:
    :return:
    """
    with tf.name_scope("preprocess") as escopo :

        x = tf.div ( X, 255., name="rescaled_inputs" )

    #Down Convolucutional

    with tf.contrib.framework.arg_scope( \
            [convolucao], \
            padding="SAME",
            stride=2,
            activation_fn=relu,
            normalizer_fn=batchnorm,
            normalizer_params={"is_training": isTraining},
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            ) :

        with tf.variable_scope("d1") as escopo :

            d1 = convolucao ( x, num_outputs=32, kernel_size=3, stride=1, scope = "conv1" )
            d1 = convolucao ( d1, num_outputs=32, kernel_size=3, scope="conv2" )
            d1 = drpoOutLayer ( d1, rate=dropOut, name="dropout" )

            print("d1 : {}".format ( d1.shape.as_list (  ) ) )

        with tf.variable_scope("d2") as escopo :

            d2 = convolucao ( d1, num_outputs=64, kernel_size=3, stride=1, scope="conv1")
            d2 = convolucao ( d2, num_outputs=64, kernel_size=3, scope="conv2")
            d2 = drpoOutLayer ( d2, rate=dropOut, name="dropout")

            print("d2 : {}".format ( d2.shape.as_list (  ) ) )

        with tf.variable_scope("d3") as escopo :

            d3 = convolucao ( d2, num_outputs=128, kernel_size=3, stride=1, scope="conv1")
            d3 = convolucao ( d3, num_outputs=128, kernel_size=3, scope="conv2")
            d3 = drpoOutLayer ( d3, rate=dropOut, name="dropout")

            print("d3 : {}".format ( d3.shape.as_list (  ) ) )

        with tf.variable_scope("d4") as escopo :

            d4 = convolucao ( d3, num_outputs=256, kernel_size=3, stride=1, scope="conv1")
            d4 = convolucao ( d4, num_outputs=256, kernel_size=3, scope="conv2")
            d4 = drpoOutLayer ( d4, rate=dropOut, name="dropout")

            print("d4 : {}".format ( d4.shape.as_list ( ) ) )


    # Up Convolutional

    with tf.contrib.framework.arg_scope([desconvolucao, convolucao], \
        padding = "SAME",
        activation_fn = None,
        normalizer_fn = tf.contrib.layers.batch_norm,
        normalizer_params = {"is_training": isTraining},
        weights_initializer = tf.contrib.layers.xavier_initializer(),
        ):

        with tf.variable_scope("u3") as escopo :

            u3 = desconvolucao ( d4, num_outputs = nClasses, kernel_size = 4, stride = 2 )

            s3 = convolucao ( d3, num_outputs = nClasses, kernel_size = 1,
                              stride = 1, activation_fn = relu, scope = "s" )

            u3 = tf.add ( u3, s3, name="up" )

            print("u3 : {}".format ( u3.shape.as_list (  ) ) )

        with tf.variable_scope("u2") as escopo:

            u2 = desconvolucao ( u3, num_outputs = nClasses, kernel_size = 4, stride = 2)

            s2 = convolucao ( d2, num_outputs = nClasses, kernel_size = 1,
                            stride = 1, activation_fn = relu, scope = "s")

            u2 = tf.add ( u2, s2, name="up")

            print("u2 : {}".format ( u2.shape.as_list (  ) ) )

        with tf.variable_scope("u1") as escopo:

            u1 = desconvolucao ( u2, num_outputs = nClasses, kernel_size = 4, stride = 2)

            s1 = convolucao ( d1, num_outputs = nClasses, kernel_size = 1,
                            stride = 1, activation_fn = relu, scope = "s")

            u1 = tf.add(u1, s1, name="up")

            print("u1 : {}".format ( u1.shape.as_list ( ) ) )


    return desconvolucao ( u1, num_outputs = nClasses, kernel_size = 4,
                           stride = 2, activation_fn = None, normalizer_fn = None, scope = "logitis" )

def pegaPicke ( file ) :
    """


    :param file:
    :return:
    """
    with open ( file, mode="rb" ) as fileObj :

        obj = pickle.load ( fileObj )

    return obj

if __name__ == '__main__':

    # atalho para operacoes
    convolucao = tf.contrib.layers.conv2d
    desconvolucao = tf.contrib.layers.conv2d_transpose
    relu = tf.nn.relu

    drpoOutLayer = tf.layers.dropout
    batchnorm = tf.contrib.layers.batch_norm
    argEscopo = tf.contrib.framework.arg_scope

    # pegando os dados
    data = pegaPicke("data64_flat_grey.pickle")

    print("-----------------------------------")
    print("Formato Dados")
    print("- xTreino : {}".format( data["X_train"].shape ) )
    print("- yTreino : {}".format ( data["Y_train"].shape ) )
    print("- xTeste : {}".format(data["X_test"].shape))
    print("- yTeste : {}".format(data["Y_test"].shape))
    print("- xValidação : {}".format ( data["X_valid"].shape ) )
    print("- yValidação : {}".format ( data["Y_valid"].shape ) )
    print("-----------------------------------\n")

    nEpocas = int( input ( "Digite a quantidade de épocas que deseja treinar a máquina : ") )
    alpha = float ( input("Digite o valor de alpha : ") )

    assert nEpocas > 0, \
        "Valor inválido !!!!"

    # criando e treinndo o modelo
    modelo = Segmentacao ( formatoImagem = [64, 64], nCanais = 1, nClasses = 4 )
    modelo.criaGrafDaFuncLogic ( modeloLogistico )
    modelo.treino(data, nEpocas = nEpocas, alpha = alpha, tamBatch = 4, printEvery = nEpocas*20 )



