import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import sys
import os

def pad_zeros(matriz):
    """
    Recebe uma matriz 2D de tamanho (h, w) e retorna a mesma matriz
    com uma borda de tamanho 1 preenchida com zeros, 
    resultando em tamanho (h+2, w+2).
    """
    h, w = matriz.shape
    # Cria uma nova matriz de zeros com tamanho h+2 e w+2
    padded = np.zeros((h + 2, w + 2), dtype=matriz.dtype)
    # Copia a matriz original para o miolo da nova matriz
    padded[1:h+1, 1:w+1] = matriz
    return padded

def sup_nao_maximo(h, w, theta_graus, Kxy_padded):
    """
    Realiza a supressão de não-máximos para um único pixel (h, w).
    Compara o valor do pixel com seus vizinhos na direção do gradiente.
    """
    angle = theta_graus[h, w]
    
    # Em Kxy_padded, o pixel (h, w) original está na posição (h+1, w+1)
    val = Kxy_padded[h+1, w+1]
    
    if angle == 0:
        vizinho1 = Kxy_padded[h+1, w]     # esquerda
        vizinho2 = Kxy_padded[h+1, w+2]   # direita
    elif angle == 45:
        vizinho1 = Kxy_padded[h+2, w]     # inferior esquerdo
        vizinho2 = Kxy_padded[h, w+2]     # superior direito
    elif angle == 90:
        vizinho1 = Kxy_padded[h, w+1]     # de cima (y-1 original é h, mas estamos analisando em Kxy_padded logo é h e h+2. No Python matriz y cresce p/ baixo, vizinhos superior e inferior)
        vizinho2 = Kxy_padded[h+2, w+1]   # de baixo
    elif angle == 135:
        vizinho1 = Kxy_padded[h, w]       # superior esquerdo
        vizinho2 = Kxy_padded[h+2, w+2]   # inferior direito
    else:
        vizinho1 = 0
        vizinho2 = 0
        
    if val >= vizinho1 and val >= vizinho2:
        return val
    else:
        return 0.0

if len(sys.argv) < 4:
    print("Uso: python main.py <caminho_imagem> <limiar_forte> <limiar_fraco>")
    sys.exit(1)

caminho_imagem = sys.argv[1]
limiar_forte = int(sys.argv[2])
limiar_fraco = int(sys.argv[3])

#Ler a imagem e converter para RGB para evitar problemas com PNGs (canal Alpha) ou imagens em tons de cinza
imagem = Image.open(caminho_imagem).convert('RGB')

# converter para numpy array
imagem = np.array(imagem)

#Separar os canais
R = imagem[:, :, 0]
G = imagem[:, :, 1]
B = imagem[:, :, 2]

# MODULO A

#ver depois se essa função precisa mesmo, ou se é desse jeito mesmo, já que a função 
# do módulo B carregarGabor deveria fazer isso
def carregarKernel(caminho):
  if not caminho.lower().endswith(".json"):
    return np.loadtxt(caminho, dtype=np.float32) # .txt: matriz separada por espaço

  with open(caminho, 'r') as arquivo:
    dados = json.load(arquivo)

    kernel = np.array(
        dados['kernel'],
        dtype=np.float32
    )

  return kernel

def expansao_histograma(imagem):
  """
  Realiza a expansão de histograma (contrast stretching) baseada na fórmula:
  s = round( ((r - r_min) / (r_max - r_min)) * (L - 1) )
  """
  
  # 1. Encontrar r_min e r_max na imagem original
  r_min = np.min(imagem)
  r_max = np.max(imagem)
    
  # L é o número de níveis de cinza (geralmente 256 para 8-bits)
  L = 256
    
  # Prevenção contra divisão por zero (imagem de cor sólida)
  if r_max == r_min:
    return imagem.copy()
    
  # 2. Aplicar a fórmula exata: s = ((r - r_min) / (r_max - r_min)) * (L - 1)
  # Convertendo 'r' (imagem) para float para evitar truncamento no meio do cálculo
  r = imagem.astype(np.float32)
  s = ((r - r_min) / (r_max - r_min)) * (L - 1)
    
  # 3. Arredondar (round) e converter para inteiro de 8-bits
  s = np.round(s).astype(np.uint8)
    
  return s

def equalizacao_histograma(imagem):
  """
  Realiza a equalização de histograma baseada na fórmula geral:
  s = round( ((cdf(v) - cdf_min) / (RC - cdf_min)) * (L - 1) )
  """      
  L = 256
  RC = imagem.size
  
  # Calcula o histograma (n_l)
  hist, _ = np.histogram(imagem.ravel(), L, [0, L])
  
  # Calcula a soma cumulativa ( sum(n_l) )
  cdf = hist.cumsum()
  
  # Encontra o valor mínimo não-zero da CDF
  cdf_min = cdf[cdf > 0].min()
  
  # Aplica a fórmula: s = round( ((cdf - cdf_min) / (RC - cdf_min)) * (L - 1) )
  # Usamos np.maximum para evitar valores negativos caso cdf seja menor que cdf_min (o que não deve ocorrer, mas por segurança)
  cdf_normalizado = np.round(((cdf - cdf_min) / (RC - cdf_min)) * (L - 1))
  cdf_normalizado = np.clip(cdf_normalizado, 0, 255).astype(np.uint8)
  
  # Mapeia os pixels da imagem original para os novos valores equalizados
  img_eq = cdf_normalizado[imagem]
  
  return img_eq

def plot_histograma(ax, imagem, titulo):
  """
  Plota um histograma usando linhas azuis finas, imitando o estilo 
  da imagem de referência (image_978e05.png).
  """
  # Calcula o histograma: array com 256 posições contendo a contagem de pixels
  hist, bins = np.histogram(imagem.ravel(), 256, [0, 256])
  
  # Ignora pixels com valor 0 para melhorar a visualização do restante do histograma
  hist[0] = 0
  
  # Plota como barras finas azuis (imitando as linhas)
  ax.bar(np.arange(256), hist, width=1, color='blue', edgecolor='blue')
  
  # Configurações do gráfico
  ax.set_title(titulo)
  ax.set_xlim([0, 255])
  
  # Ajustar limite Y para focar no corpo do histograma (ignora picos gigantes se houver)
  y_max = np.max(hist)
  ax.set_ylim([0, y_max * 1.05])
  
  # Adicionando uma barra de gradiente simulada na base do eixo X
  # Cria uma imagem 1x256 com valores de 0 a 255
  gradiente = np.linspace(0, 255, 256).reshape(1, 256)
  ax.imshow(gradiente, cmap='gray', aspect='auto', extent=[0, 255, -y_max*0.08, 0])

def correlacao2d(imagem,kernel):

  #tamanho da imagem e do filtro
  h_img, w_img = imagem.shape
  h_k, w_k = kernel.shape

  #descobrir o pivô do filtro
  pad_h = h_k // 2
  pad_w = w_k // 2

  #extensão por 0
  imagem_pad = np.pad(
      imagem,
      ((pad_h, pad_h), (pad_w, pad_w)),
      mode="constant" #substitui por 0
  )

  saida = np.zeros((h_img, w_img), dtype=np.float32) #cria a imagem de saida  

  for i in range(h_img):
    for j in range(w_img): #percorre todos os pixels

      vizinhanca = imagem_pad[i:i+h_k,j:j+w_k] #extrai a vizinhança

      saida[i,j] = np.sum(vizinhanca * kernel)

  return saida


# recebe uma img HxWx3 (altura, largura, 3 canais de cor) e aplica correlacao2d 
# separadamente em cada canal (R, G, B) - imagem[:,:,0], imagem [:,:,1], imagem[:,:,2].
def correlacao_rgb(imagem, kernel):

  # como cada chamada de correlacao2d retorna um array, usando np.stack, 
  # "empilha" de volta pra 3a dimensão, refazendo a img HxWx3
  return np.stack([
      correlacao2d(imagem[:, :, c], kernel)
      for c in range(3)
  ], axis=2)

# resultado: cada pixel tem a resposta do kernel aplicada independentemente no seu canal de cor

# CANNY TRADICIONAL

#Converte para tons de cinza
imagem_cinza = 0.299 * R + 0.587 * G + 0.114 * B


# Aplica o filtro Gaussiano na imagem em tons de cinza
gaussiano_kernel = carregarKernel("filtro_15x15/gaussiano.json")
imagem_suavizada = correlacao2d(imagem_cinza, gaussiano_kernel)

#Imprime as linhas de 0 a 9 e as colunas de 0 a 9
# pedaco = imagem_suavizada[0:10, 0:10]
# print("PEDAÇO [0:10, 0:10] DA IMAGEM EM CINZA\n")
# print(pedaco)

#Exibe a imagem
# plt.imshow(imagem_suavizada, cmap="gray")
# plt.axis("off")
# plt.show()

Kx_kernel = carregarKernel("filtros/sobel_x.json")
Ky_kernel = carregarKernel("filtros/sobel_y.json")

Kx = correlacao2d(imagem_suavizada, Kx_kernel)
Ky = correlacao2d(imagem_suavizada, Ky_kernel)

Kxy = np.hypot(Kx,Ky)

theta_radiano = np.arctan2(Ky,Kx)
theta_graus = theta_radiano * (180 / np.pi)
theta_graus[theta_graus < 0] += 180

h_Kxy, w_Kxy = Kxy.shape

for i in range(h_Kxy):
  for j in range(w_Kxy):
    if theta_graus[i,j] >= 157.5:
      theta_graus[i,j] = 0
    elif theta_graus[i,j] >= 112.5:
      theta_graus[i,j] = 135
    elif theta_graus[i,j] >= 67.5:
      theta_graus[i,j] = 90
    elif theta_graus[i,j] >= 22.5:
      theta_graus[i,j] = 45
    else:
      theta_graus[i,j] = 0

Kxy_padded = pad_zeros(Kxy)


# Iteração para a Supressão de Não-Máximos
imagem_suprimida = np.zeros((h_Kxy, w_Kxy), dtype=np.float32)

for i in range(h_Kxy):
  for j in range(w_Kxy):
    imagem_suprimida[i, j] = sup_nao_maximo(i, j, theta_graus, Kxy_padded)

# Aplicar a equalização usando a nova função matemática
img_expansao = expansao_histograma(imagem_suprimida)

img_equalizada = equalizacao_histograma(img_expansao)

def limiarizacao_histerese(imagem, t_baixo=50, t_alto=120):
  """
  Realiza a limiarização por histerese classificando os pixels em 3 categorias.
  """
  BORDA_FORTE = 255
  BORDA_FRACA = 100
  NAO_BORDA = 0
  
  h, w = imagem.shape
  resultado = np.zeros((h, w), dtype=np.uint8)
  
  # Passo 1: Classificar os pixels
  forte_i, forte_j = np.where(imagem >= t_alto)
  fraca_i, fraca_j = np.where((imagem >= t_baixo) & (imagem < t_alto))
  
  resultado[forte_i, forte_j] = BORDA_FORTE
  resultado[fraca_i, fraca_j] = BORDA_FRACA
  # O resto (NAO_BORDA) já é 0 por conta do np.zeros
  
  # Passo 2: Analisar bordas fracas e conectá-las às fortes (propagação)
  # Usamos uma pilha (stack) com as posições das bordas fortes iniciais
  pilha = list(zip(forte_i, forte_j))
  
  dx = [-1, -1, -1,  0, 0,  1, 1, 1]
  dy = [-1,  0,  1, -1, 1, -1, 0, 1]
  
  while pilha:
    x, y = pilha.pop()
    
    # Verifica os 8 vizinhos
    for k in range(8):
      nx, ny = x + dx[k], y + dy[k]
      
      # Verifica limites da imagem
      if 0 <= nx < h and 0 <= ny < w:
        # Se o vizinho é uma borda fraca, ela está conectada a uma forte!
        if resultado[nx, ny] == BORDA_FRACA:
          resultado[nx, ny] = BORDA_FORTE # Promove a borda forte
          pilha.append((nx, ny)) # Adiciona na pilha para propagar

  # Passo 3: Zerar todas as bordas fracas restantes (que não se conectaram a fortes)
  resultado[resultado == BORDA_FRACA] = NAO_BORDA
  
  return resultado

# Aplicar a limiarização por histerese na imagem após NMS
# Usando img_equalizada para obter melhor contraste, ou pode ser usado imagem_suprimida
# Normalmente a histerese é aplicada antes da equalização, diretamente no NMS.
# Como o NMS retorna valores muito baixos, aplicaremos na imagem pós expansão ou equalizada
# Vamos aplicar na imagem_suprimida usando os limiares adequados (que na NMS são baixos)
# OU na imagem equalizada para testar os limiares 150 e 200.
img_histerese = limiarizacao_histerese(img_equalizada, t_baixo=limiar_fraco, t_alto=limiar_forte)

# Salvar a imagem resultante na pasta "saída"
os.makedirs("saída", exist_ok=True)
nome_arquivo = os.path.basename(caminho_imagem)
caminho_saida = os.path.join("saída", f"resultado_gaussiano15x15_{nome_arquivo}")

# Para colocar a original à esquerda e o resultado à direita, 
# a imagem de histerese (1 canal) precisa ser convertida para 3 canais
img_histerese_3c = np.stack((img_histerese,) * 3, axis=-1)

# Concatena a imagem original e o resultado horizontalmente (axis=1)
img_combinada = np.concatenate((imagem, img_histerese_3c), axis=1)

# Salvar a imagem combinada
Image.fromarray(img_combinada).save(caminho_saida)
print(f"Resultado salvo em: {caminho_saida}")