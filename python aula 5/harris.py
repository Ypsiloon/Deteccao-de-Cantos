import cv2
import numpy as np

# Lista com os caminhos das imagens
imagens = ['images\exemplo.jpg', 'images\images-teste.jpg', 'images\images.jpg']

# Definir diferentes valores de blockSize e ksize
parametros = [(2, 3), (6, 9), (20, 25)]  # blockSize, ksize

# Processar cada imagem da lista
for caminho_imagem in imagens:
    # Carregar a imagem em escala de cinza
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if imagem is None:
        print(f"Erro ao carregar a imagem {caminho_imagem}")
        continue

    # Processar cada combinação de blockSize e ksize
    for blockSize, ksize in parametros:
        # Aplicar o detector de Harris com os parâmetros atuais
        harris = cv2.cornerHarris(imagem, blockSize=blockSize, ksize=ksize, k=0.04)
        harris = cv2.dilate(harris, None)

        # Destacar os cantos na imagem original
        imagem_destacada = imagem.copy()  # Para não sobrescrever a imagem original
        imagem_destacada[harris > 0.01 * harris.max()] = 255

        # Exibir os resultados
        cv2.imshow(f'Harris Corner - {caminho_imagem} - blockSize={blockSize} - ksize={ksize}', imagem_destacada)
        cv2.waitKey(0)

cv2.destroyAllWindows()
