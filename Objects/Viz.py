import os
import numpy as np
import PIL
import PIL.Image

def batchVizSeg ( img, labels, labels2 = None, colorMap = None, tamGrid=(3,8), salva=None ) :
    """

    Função

    :param img:
    :param labels:
    :param labels2:
    :param colorMap:
    :param tamGrid:
    :param salva:
    :return:
    """

    linhas, colunas = tamGrid

    # verificando se as linhas e as colunas
    # são positivos inteiros
    assert ( linhas > 0 and colunas > 0)
    nCelulas = ( linhas*colunas )
    nAmostras = img.shape[0]

    if ( colorMap is None ) :

        colorMap = [
            [0, 0, 0],
            [255, 79, 64],
            [115, 173, 33],
            [48, 126, 199]
        ]

    # fazendo um grupo de imagens em pares e trios
    outPut = []

    for i in range ( min ( nCelulas, nAmostras ) ) :

        x = img[i]
        y = labels[i]

        # fazendo a conversão em um canal RGB
        x =np.repeat ( x, 3, axis=2 )

        # aplicando o colormap nos labeis
        # e nas predições
        y = np.array ( colorMap )[y].astype ( np.uint8 )

        if ( labels2 is not None ) :
            y2 = labels2[i]
            y2 = np.array ( colorMap )[y2].astype(np.uint8)
            outPut.append ( np.concatenate ( [x, y, y2], axis=0 ) )

        else :

            outPut.append ( np.concatenate ( [x, y], axis=0 ) )

    outPut = np.array( outPut, dtype=np.uint8 )

    # preparando as dimensões pro grind

    nBatch, imgAltura, imgLargura, nCanais = outPut.shape

    nGrap = nCelulas - nBatch
    outPut = np.pad(outPut, pad_width=[(0, nGrap), (0, 0), (0, 0), (0, 0)], mode="constant", constant_values=0)

    # tratando do caso onde não há
    # imagens suficientes no batch
    # para encher o grid
    outPut = outPut.reshape ( linhas, colunas, imgAltura ,imgLargura, nCanais ).swapaxes(1, 2)
    outPut = outPut.reshape ( linhas*imgAltura, colunas*imgLargura, nCanais )

    outPut = PIL.Image.fromarray ( outPut.squeeze (  ) )

    # salvando a imagem

    if ( salva is not None ) :

        # criando o caminho para salvar o arquivo
        parDir = os.path.dirname ( salva )

        # verificando se parDir não é uma string vazia
        if ( parDir.strip() != "" ):

            if ( not os.path.exists ( parDir ) ) :
                os.makedirs ( parDir )

        outPut.save ( salva, "JPEG" )

    return outPut