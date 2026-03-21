# Sistema de Reconhecimento de Imagens

Aplicativo desktop em **Python + Tkinter + Pillow**, agora com dois motores de reconhecimento:

- **Modo IA com embeddings CLIP** para uma análise mais avançada.
- **Modo clássico** com hash perceptual, histograma de cores e perfil de bordas.
- **Cadastro de imagens para treinamento** por categoria.
- **Reconhecimento de novas imagens** com busca na base local em SQLite.

## O que foi implementado nesta versão avançada

O próximo passo com IA já foi incorporado ao projeto:

1. O app oferece um seletor entre **IA (CLIP)** e **modo clássico**.
2. No modo IA, ele usa embeddings visuais do modelo **`openai/clip-vit-base-patch32`**.
3. As amostras de treinamento ficam separadas por backend no banco local.
4. Se `torch` e `transformers` não estiverem instalados no ambiente, o app faz **fallback automático para o modo clássico**.

Isso permite começar com IA sem perder a usabilidade do MVP em máquinas onde a stack completa ainda não esteja pronta.

## Como funciona o modo IA

### Fluxo de treinamento

- Selecione o motor **IA (CLIP embeddings)**.
- Cadastre várias imagens da mesma categoria.
- O sistema extrai embeddings vetoriais com CLIP e armazena no SQLite.

### Fluxo de reconhecimento

- Escolha novamente o motor **IA (CLIP embeddings)**.
- Insira uma imagem para consulta.
- O sistema gera o embedding da imagem e compara com os embeddings treinados usando similaridade vetorial.

## Como funciona o modo clássico

O modo clássico continua disponível como fallback e também como baseline local:

- hash perceptual;
- histograma de cores;
- perfil de bordas.

## Diferenciais incluídos

- Interface moderna com tema escuro.
- Seleção de imagens por janela nativa do Windows (`filedialog`).
- Armazenamento local em `data/catalog.db`.
- Organização automática das imagens treinadas em `data/images/<categoria>/`.
- Suporte a **JPG, JPEG, PNG, BMP, GIF, WebP, PPM e PGM**.
- Alternância entre motor clássico e motor com IA.
- Fallback automático para o modo clássico quando a stack de IA não estiver disponível.

## Requisitos

- Python 3.11+
- Tkinter disponível no Python
- Pillow
- torch
- transformers

## Instalação no Windows

### Opção 1 — PowerShell com ajuste temporário da política de execução

Se aparecer o erro **"a execução de scripts foi desabilitada neste sistema"**, execute primeiro:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois crie e ative o ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

> Esse ajuste vale apenas para a janela atual do PowerShell.

### Opção 2 — Prompt de Comando (CMD), sem mexer na policy do PowerShell

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python app.py
```

### Opção 3 — Sem ativar o ambiente virtual

Se preferir, você pode usar o Python do ambiente virtual diretamente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Dependências de IA

O modo IA usa:

- `torch` para inferência;
- `transformers` para carregar o CLIP;
- `Pillow` para leitura e pré-processamento das imagens.

Modelo configurado atualmente:

- `openai/clip-vit-base-patch32`

## Recomendações para melhor resultado com IA

- Cadastre entre **10 e 30 imagens por categoria**.
- Use exemplos com **ângulos, fundos, iluminação e escala variados**.
- Evite categorias visualmente muito parecidas com poucas amostras.
- Quando possível, mantenha uma base equilibrada entre as classes.

## Problemas comuns

### 1. "Modo IA indisponível no ambiente"

Isso acontece quando `torch` e `transformers` não estão instalados corretamente no ambiente usado para executar o app.

### 2. Erro no PowerShell ao ativar `.venv`

Esse erro ocorre por causa da política de execução de scripts do Windows/PowerShell, não por erro do projeto. Use uma das opções mostradas na seção **Instalação no Windows**.

### 3. `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`

Esse erro pode acontecer com combinações específicas de versões do `transformers`/CLIP, quando a saída do modelo não vem diretamente como tensor já pronto para normalização. O projeto foi ajustado para tratar esse retorno de forma compatível antes de normalizar o embedding.

### 4. `mat1 and mat2 shapes cannot be multiplied (1x512 and 768x512)`

Esse erro acontece quando uma projeção visual do CLIP é aplicada sobre um embedding que já estava projetado, gerando incompatibilidade de dimensões. O projeto foi ajustado para usar primeiro `image_embeds` da saída completa do modelo e só projetar `pooler_output` quando as dimensões realmente forem compatíveis.

## Próximos upgrades possíveis

- salvar um índice vetorial para busca mais rápida;
- permitir exclusão/edição de amostras;
- capturar imagens via webcam;
- expor uma API com FastAPI;
- trocar CLIP por um encoder mais forte ou especializado no seu domínio.

## Estrutura

```text
app.py
requirements.txt
README.md
data/
```
