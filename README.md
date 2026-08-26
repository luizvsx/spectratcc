# SPECTRA

Pipeline de análise de partidas de futebol com detecção de jogadores e bola,
rastreamento, classificação dos times e identificação de passes. Esta versão é
baseada no antigo `tc5.py`, escolhido como a implementação mais estável do TCC.

## Estrutura

```text
FinalTCC/
├── config/             # Configuração do pipeline
├── data/
│   ├── models/         # Pesos locais (não enviados ao Git)
│   └── videos/         # Vídeos de entrada (não enviados ao Git)
├── outputs/            # Vídeos e estatísticas gerados
├── scripts/            # Comandos auxiliares
├── src/
│   ├── main.py         # Ponto de entrada principal
│   └── tcc_common/     # Detecção, rastreamento e análise
├── tests/              # Testes automatizados
└── web/                # Interface Flask e dashboard do CSV
```

Arquivos experimentais e versões antigas foram preservados localmente em
`lixeira/`, que é ignorada pelo Git.

## Instalação

Clone o repositório e execute os comandos sempre a partir da raiz. Nenhum
caminho pessoal ou absoluto precisa ser editado:

```bash
git clone URL_DO_REPOSITORIO
cd FinalTCC
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

No Windows, ative o ambiente com `.venv\\Scripts\\activate`.

Os seguintes modelos devem estar em `data/models/`:

- `football-player-detection.pt`
- `football-ball-detection.pt`
- `football-pitch-detection.pt`
- `mobile_sam.pt`

Valide os arquivos após o download:

```bash
sha256sum -c data/models/models.sha256
```

Os pesos totalizam aproximadamente 434 MB e não entram no Git comum. Para que
um clone novo funcione, eles precisam ser publicados com Git LFS ou em uma
área de downloads e copiados para `data/models/`.

## Execução

```bash
python3 src/main.py \
  --source_video_path data/videos/partida.mp4 \
  --target_video_path outputs/resultado.mp4 \
  --device cpu \
  --mode FULL_ANALYSIS
```

Após a instalação editável, o mesmo pipeline pode ser iniciado pelo comando
`spectra`. Para processar todos os vídeos em `data/videos/`:

```bash
bash scripts/run_all_videos.sh
```

## Interface web

A interface permite enviar o vídeo, acompanhar o processamento e visualizar
os gráficos gerados a partir do CSV de estatísticas:

```bash
python3 web/app.py
```

Abra `http://localhost:5000`. Em um servidor remoto, encaminhe a porta por SSH:

```bash
ssh -L 5000:localhost:5000 usuario@servidor
```

Por padrão o projeto usa CPU, que funciona sem GPU NVIDIA. Para usar CUDA:

```bash
SPECTRA_DEVICE=cuda python3 web/app.py
```

No servidor CUDA 12.1, as versões específicas podem ser instaladas com
`python3 -m pip install -r requirements-cuda121.txt`.

## Testes

```bash
python3 -m pytest
```
# spectratcc
