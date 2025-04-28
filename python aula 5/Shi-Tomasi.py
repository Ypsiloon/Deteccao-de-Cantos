import cv2
import numpy as np

# Lista com os caminhos das imagens
imagens = ['images\\exemplo.jpg', 'images\\images-teste.jpg', 'images\\images.jpg']

# Definir diferentes valores para maxCorners, qualityLevel e minDistance
parametros_shi_tomasi = [
    (100, 0.01, 10),  # maxCorners, qualityLevel, minDistance
    (200, 0.05, 20),
    (50, 0.1, 30)
]

# Processar cada imagem da lista
for caminho_imagem in imagens:
    # Carregar a imagem em escala de cinza
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if imagem is None:
        print(f"Erro ao carregar a imagem {caminho_imagem}")
        continue

    # Processar cada combinação de maxCorners, qualityLevel e minDistance
    for maxCorners, qualityLevel, minDistance in parametros_shi_tomasi:
        # Aplicar o detector Shi-Tomasi
        cantos = cv2.goodFeaturesToTrack(imagem, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
        cantos = np.int32(cantos)  # Converter para np.int32

        # Marcar os cantos na imagem original
        imagem_destacada = imagem.copy()  # Para não sobrescrever a imagem original
        for canto in cantos:
            x, y = canto.ravel()
            cv2.circle(imagem_destacada, (x, y), 3, 255, -1)  # Desenhar o círculo nos cantos

        # Exibir os resultados
        cv2.imshow(f'Shi-Tomasi - {caminho_imagem} - maxCorners={maxCorners} - qualityLevel={qualityLevel} - minDistance={minDistance}', imagem_destacada)
        cv2.waitKey(0)

cv2.destroyAllWindows()
