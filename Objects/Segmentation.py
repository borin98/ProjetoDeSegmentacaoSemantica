import tensorflow as tf
import numpy as np
import os
import shutil
import time
import pickle
from Objects.Viz import batchVizSeg

class Segmentacao ( object ) :
    """

        Classe que cria o modelo de
        Segmentação da imagem

        Estou caçando os erros aqui,
        Problema está no objeto sessão
        provavelmente ele não está inicializando
        direito

        OLhar a função inicializaVariaveis


    """

    def __init__(self, formatoImagem, nCanais = 3, nClasses = 10):

        self.tamanhoBatch = 4
        self.formatoImagem = formatoImagem
        self.imagemAltura, self.imagemLargura = formatoImagem
        self.nCanais = nCanais
        self.nClasses = nClasses
        self.graf = tf.Graph()

        self.snapshot_file = os.path.join("snapshots", "snapshot.chk")
        self.tensorboard_dir = "tensorboard"

        # setando os diretórios, caso eles não existam, dos arquivos
        if not ( os.path.exists(self.tensorboard_dir) ):

            os.makedirs(self.tensorboard_dir)

        if not ( os.path.exists("snapshots") ) :

            os.makedirs("snapshots")

        if not ( os.path.exists("amostras") ) :

            os.makedirs("amostras")

    def inputOperacoes(self):
        """
        FUnção que cria o input dos parâmetros
        da função logística

        Lembrando que nossa imagem será uma imagem de
        32 bits

        :return:
        """

        with tf.variable_scope("inputs"):
            formatoEixoX = (None, self.imagemAltura, self.imagemLargura, self.nCanais)
            formatoEixoY = (None, self.imagemAltura, self.imagemLargura)

            self.X = tf.placeholder(tf.float32, shape=formatoEixoX, name="X")
            self.Y = tf.placeholder(tf.int32, shape=formatoEixoY, name="Y")
            self.alpha = tf.placeholder_with_default(0.001, shape=None, name="alpha")
            self.dropout = tf.placeholder_with_default(0.0, shape=None, name="dropout")
            self.isTraining = tf.placeholder_with_default(False, shape=(), name="is_training")

    def predicao(self, X, sessao, tamBatch=4):
        """

        Função que faz a predição dos dados X
        durante uma sessão

        :param X:
        :param sessao:
        :param tamBatch:
        :return:
        """

        nAmostras = X.shape[0]
        nBatches = int(np.ceil(nAmostras/tamBatch))
        formatoSaida = [ nAmostras, self.imagemAltura, self.imagemLargura ]

        predicao = np.zeros ( formatoSaida, dtype=np.uint8 )

        # fazendo predição em cada mini batche
        for i in range ( nBatches ) :

            xBatch = X[tamBatch*i : tamBatch* ( i + 1 ) ]
            feedDict = {
                self.X:xBatch, self.isTraining:False
            }

            predicacoBatch = sessao.run ( self.preds, feed_dict = feedDict )
            predicao[ tamBatch*i:tamBatch* ( i + 1 ) ] = predicacoBatch.squeeze()

        return predicao

    def validacaoRede (self, X, Y, sessao, tamBatch=4 ) :
        """
        Passa o input de X e os labels Y,
        validando a precisão do modelo

        :param X:
        :param Y:
        :param sessao:
        :param tamBatch:
        :return:
        """

        perdaTotal = 0
        nAmostras = len ( Y )

        # quantidade de batches necessários
        nBatches = int( np.ceil ( nAmostras/tamBatch ) )

        # reseta as variáveis da métrica de validação
        sessao.run ( self.resetaVariaveisValid )

        # interage para cada mini-batch
        for i in range(nBatches):

            xBatch = X[tamBatch * i: tamBatch * (i + 1)]
            yBatch = Y[tamBatch * i: tamBatch * (i + 1)]

            feed_dict = {
                self.X : xBatch,
                self.Y : yBatch,
                self.isTraining : False
                }

            # pega a perda e atualiza as variáveis para
            # cada métrica de validação

            perda, predicao, confusaMTX = sessao.run([
                self.perda,
                self.preds,
                self.atualizaVariaveisValid
                ],  feed_dict = feed_dict )

            perdaTotal += perda

        # pega e atualiza a pontuação da rede
        # rodando as métricas
        pontuacao = sessao.run ( self.validacao )

        # médida de perda
        medPerda = perdaTotal/float ( nBatches )

        return pontuacao, medPerda

    def vizualizacaoSegmentacao(self, data, sessao, i = 0, tamBatch = 4, formato = [2, 8] ) :
        """
        Função que cria a vizualização da imagem

        :param data:
        :param sessao:
        :param i:
        :param tamBatch:
        :param formato:
        :return:
        """

        vizLinhas, vizColunas = formato
        nViz = vizLinhas*vizColunas
        vizImgTemplate = os.path.join ( "Amostras", "{}_epoca_{:07d}.jpg" )

        # predição para os dados de treino
        predicao = self.predicao( data["X_train"][:nViz],
                                  sessao = sessao,
                                  tamBatch = tamBatch
                                )

        batchVizSeg(data["X_train"][:nViz],
                     labels = data["Y_train"][:nViz],
                     labels2 = predicao[:nViz],
                     colorMap = data.get("colormap", None),
                     tamGrid = formato,
                     salva = vizImgTemplate.format ( "train", i ) )

        # para os dados de validação
        predicao = self.predicao( data["X_valid"][:nViz],
                                  sessao = sessao,
                                  tamBatch = tamBatch
                                )

        batchVizSeg(data["X_valid"][:nViz],
                     labels = data["Y_valid"][:nViz],
                     labels2 = predicao[:nViz],
                     colorMap = data.get("colormap", None),
                     tamGrid=formato,
                     salva=vizImgTemplate.format("valid", i))

    def treino(self, data, nEpocas, alpha = 0.0001, dropout = 0.0, tamBatch = 4, printEvery = 10, vizEvery = 1 ):
        """

        Função que faz o treinamento do modelo,
        em n épocas.
        passando o diretório dos dados

        :param data:
        :param alpha:
        :param dropout:
        :param tamBatch:
        :param printCamadasSaida:
        :param viz:
        :return:
        """

        # Quantidade de amostras de treino
        nAmostras = len ( data["X_train"] )

        # Quantidade de batches por época
        nBatches = int ( np.ceil ( nAmostras/tamBatch ) )

        with tf.Session(graph=self.graf) as sessao:

            self.inicializaVariaveis ( sessao )

            print(sessao)

            for epocas in range(1,  nEpocas + 1):

                print ( "\nEpoca {}/{}".format ( epocas, nEpocas ) )

                # interage para cada mini-batch
                for i in range ( nBatches ) :

                    xBatch = data["X_train"][tamBatch*i: tamBatch * ( i + 1 ) ]
                    yBatch = data["Y_train"][tamBatch*i: tamBatch * ( i + 1 ) ]

                    # treinando a rede

                    feedDict = {
                        self.X : xBatch ,
                        self.Y : yBatch ,
                        self.alpha : alpha ,
                        self.isTraining : True,
                        self.dropout : dropout
                    }

                    perda, _ = sessao.run ( [
                        self.perda,
                        self.operacaoTreinamento,
                    ],  feed_dict = feedDict )

                    # print do feedback dos acontecimentos

                    if  ( ( printEvery is not None) and
                        ( ( i + 1 )%printEvery == 0 ) ) :

                        print ( "{: 5d} perda de Batch : {}".format ( i, perda ) )

                # salva snapshot
                self.salva.save ( sessao, self.snapshot_file )

                # validando os dados de treino e
                # validação sets para cada epoca

                treinoIOU, treinoPerda = self.validacaoRede(
                    X = data["X_train"][:1024],
                    Y = data["Y_train"][:1024],
                    sessao = sessao
                )

                validacaoIOU, validacaoPerda = self.validacaoRede(
                    X = data["X_train"],
                    Y = data["Y_train"],
                    sessao = sessao
                )

                # printa os resultados
                print(5*"-"+" Treino "+ 5*"-"+"\n")
                print ("Treino IOU : {:3.3f}".format ( treinoIOU ) )
                print ("Perda do treino : {:3.5f}".format ( treinoPerda ) )

                print("\n"+5 * "-" + " Validação " + 5 * "-" + "\n")
                print("Treino IOU : {:3.3f}".format ( validacaoIOU ) )
                print("Perda do treino : {:3.5f}".format ( validacaoPerda ) )

                # vizualizando a predição em cada época

                if ( epocas%vizEvery == 0 ) :

                    self.vizualizacaoSegmentacao ( data = data, i = epocas, sessao = sessao )

    def inicializaVariaveis(self, sessao ) :

        if ( tf.train.checkpoint_exists ( self.snapshot_file ) ) :

            print ( "- Restaurando os parâmetros para snapshots salvos" )
            print ( " -",self.snapshot_file )
            self.salva.restore ( sessao, self.snapshot_file )
        else :

            print("Inicializando os pesos aleatórios da rede")
            sessao.run ( tf.global_variables_initializer (  ) )

    def criaOperacaoTensorBoard ( self ) :
        """
        Função que cria o nome do arquivo e
        suas descrições

        :return:
        """

        with tf.variable_scope ( "tensorboard" ) as escopo :

            self.sumarioEscrito = tf.summary.FileWriter ( self.tensorboard_dir, graph = self.graf )
            self.sumarioDummy = tf.summary.scalar ( name="dummy",tensor = 1 )


    def criaOperacaoSalvar ( self ) :
        """

        Cria operação para salvar os dados
        durante as operações

        Salvando o progresso utilizando uma gpu

        :return:
        """


        with tf.device("/cpu:0") :

            self.salva = tf.train.Saver ( name="saver" )

    def criaMetricasValidacao ( self ) :
        """
        Cria a métria de validação dos dados,
        que neste caso, iremos utilizar o modelo IoU

        :return:
        """

        with tf.name_scope ( "evaluation" ) as escopo :

            self.validacao, self.atualizaVariaveisValid = tf.metrics.mean_iou(
                tf.reshape ( self.Y, [-1] ),
                tf.reshape ( self.preds, [-1] ),
                num_classes=self.nClasses,
                name=escopo
            )

        # Isola as variáveis de métrica que estão rodando e cria o inicializador
        # e a operação de reset
        variaveisValid = tf.get_collection ( tf.GraphKeys.LOCAL_VARIABLES, scope = escopo )
        self.resetaVariaveisValid = tf.variables_initializer ( var_list = variaveisValid )

    def criaOperacaoOtimizacao ( self ) :
        """
        Função que cria a otimização da rede
        automaticamente

        :return:
        """

        with tf.variable_scope('opt') as scope :

            self.otimizador = tf.train.AdamOptimizer ( self.alpha, name="optimizer" )
            operacaoUpdate = tf.get_collection ( tf.GraphKeys.UPDATE_OPS )

            with tf.control_dependencies ( operacaoUpdate ) :

                self.operacaoTreinamento = self.otimizador.minimize ( self.perda, name="train_op" )
                print(self.operacaoTreinamento)

    def criaOperacaoPerda ( self ) :
        """

        Função de perda que soma todas as perdas de dados
        mesmo após a regularização dos dados

        :return:
        """

        with tf.variable_scope("loss") as escopo :

            paramLogistico =  tf.reshape(self.logistica, (-1, self.nClasses))
            paramLabels = tf.reshape ( self.Y, ( -1, ) )

            tf.losses.sparse_softmax_cross_entropy ( labels=paramLabels, logits=paramLogistico )
            self.perda = tf.losses.get_total_loss()

    def criaGrafDaFuncLogic(self, funcaoLogistica) :
        """
        Cria o gráfico a partir do modelo, dando a função
        logística a partir da API

        Arguments: (X, n_classes, alpha, dropout, is_training)
        Returns  : logits [n_batch, img_altura, img_largura, n_classes]


        :param funcaoLogistica: função logística que contém os parâmetros
        :return: retorna a lógica apresentada acima
        """

        # definindo e criando as operações de logística da função
        self.graf = tf.Graph()

        with self.graf.as_default() :

            self.inputOperacoes()

            self.logistica = funcaoLogistica (
                X = self.X,
                nClasses = self.nClasses,
                alpha = self.alpha,
                dropOut= self.dropout,
                isTraining = self.isTraining
            )

            with tf.name_scope("preds") as escopo :

                self.preds = tf.to_int32(tf.argmax(self.logistica, axis=-1), name=escopo)
                print(self.preds)

            # criando as operações das funções
            self.criaOperacaoPerda ( )
            self.criaOperacaoOtimizacao ( )
            self.criaMetricasValidacao (  )
            self.criaOperacaoSalvar (  )
            self.criaOperacaoTensorBoard (  )

