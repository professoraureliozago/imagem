# Sistema de Reconhecimento de Imagens

Aplicativo desktop em **Python + Tkinter + Pillow** com interface moderna para:

- **Cadastrar imagens para treinamento** por categoria.
- **Salvar a base localmente** em SQLite.
- **Reconhecer novas imagens** comparando características visuais com a base treinada.

## Como funciona

Este projeto foi otimizado para ser simples de executar no VS Code em Windows, com suporte aos formatos de imagem mais usados:

1. O usuário cadastra várias imagens semelhantes e informa o nome da categoria.
2. O sistema extrai características visuais leves com `Pillow`:
   - hash perceptual;
   - histograma de cores;
   - perfil de bordas.
3. Quando uma nova imagem é enviada para reconhecimento, o sistema compara os vetores com a base cadastrada.
4. O app retorna a categoria mais provável e mostra um nível de confiança estimado.

> Observação: esta abordagem é excelente para um MVP offline e local. Para uma versão mais avançada, o próximo passo ideal é usar embeddings de IA (ex.: CLIP, TensorFlow ou PyTorch).

## Diferenciais incluídos

- Interface moderna com tema escuro.
- Seleção de imagens por janela nativa do Windows (`filedialog`).
- Armazenamento local em `data/catalog.db`.
- Organização automática das imagens treinadas em `data/images/<categoria>/`.
- Fluxo separado para **treinamento** e **reconhecimento**.
- Pronto para rodar no VS Code.
- Suporte a **JPG, JPEG, PNG, BMP, GIF, WebP, PPM e PGM**.

## Requisitos

- Python 3.11+
- Tkinter disponível no Python
- Pillow instalado via `requirements.txt`

## Instalação

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python app.py
```

## Sugestões para melhorar ainda mais

Para melhorar a qualidade do reconhecimento:

- Cadastre entre **5 e 20 imagens por categoria**.
- Use imagens com **ângulos, fundos e iluminação diferentes**.
- Evite classes muito parecidas com poucas amostras.
- Em uma versão futura, implemente:
  - busca por similaridade com embeddings de rede neural;
  - painel de administração com exclusão/edição de classes;
  - exportação/importação da base;
  - webcam para captura ao vivo;
  - API com FastAPI para integrar com outros sistemas;
  - score calibrado com validação da base.

## Estrutura

```text
app.py
requirements.txt
README.md
data/
```
