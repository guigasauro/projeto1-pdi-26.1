import subprocess
import os

# Nomes das imagens na pasta 'imagens'
nomes_das_imagens = [
    "Bear.jpg",
    "FCBarcelona.png",
    "GrayAndMagenta.png",
    "PlacaMercosul.webp",
    "VintageCar.png",
    "Zebra.png"
]

# Defina os limiares que deseja testar
limiar_forte = "120"
limiar_fraco = "50"

# O script garante que a pasta saída existe, mas o main.py também já fará isso
os.makedirs("saída", exist_ok=True)

for nome in nomes_das_imagens:
    caminho_imagem = os.path.join("..", "imagens", nome)
    
    # Verifica se a imagem existe antes de rodar
    if os.path.exists(caminho_imagem):
        print(f"Processando imagem: {caminho_imagem}")
        # Executa o main.py passando os argumentos
        subprocess.run(["python", "main.py", caminho_imagem, limiar_forte, limiar_fraco])
    else:
        print(f"Aviso: Imagem não encontrada - {caminho_imagem}")
