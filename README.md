# SPECTRA

Sistema de visão computacional para análise automática de partidas de futebol. O SPECTRA detecta e rastreia jogadores e bola, classifica os times e produz um vídeo anotado acompanhado de estatísticas de posse, passes e chutes.

## Demonstração

[▶ Assistir à demonstração do SPECTRA](docs/media/demo.mp4)

## Requisitos

- Python 3.12
- GPU compatível com CUDA (recomendada para a análise completa)
- Modelos em `data/models/` e `mobile_sam.pt`

Instale as dependências em um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

O `requirements.txt` fixa o PyTorch 2.5.1 com CUDA 12.1 para manter
compatibilidade com o driver NVIDIA do servidor `hal9000`. A mesma build pode
ser usada em CPU quando CUDA não estiver disponível.

## Execução pela linha de comando

O pipeline oficial está em `main.py`:

```bash
python main.py \
  --source_video_path data/videos/partida.mp4 \
  --target_video_path outputs/resultado.mp4 \
  --device cuda \
  --mode FULL_ANALYSIS
```

Após `pip install -e .`, o mesmo pipeline também pode ser iniciado pelo comando
`spectra`, com os mesmos argumentos.

Os modos disponíveis são `PLAYER_DETECTION`, `BALL_DETECTION`, `PITCH_DETECTION`, `TEAM_CLASSIFICATION` e `FULL_ANALYSIS`.

## Interface web

```bash
python app.py
```

Como alternativa, use o comando instalável `spectra-web`.

Acesse `http://localhost:5000`, envie um vídeo e acompanhe o processamento. A interface usa o mesmo pipeline de `main.py`.

## Testes

```bash
pytest -q
```

A suíte rápida não carrega modelos nem exige GPU. Para ver cada teste:

```bash
pytest -v
```

Para medir a cobertura do código:

```bash
pytest --cov=spectra --cov=tcc_common --cov-report=term-missing
```

O teste ponta a ponta é ignorado por padrão. Use um vídeo curto conhecido para
executar os modelos reais:

```bash
SPECTRA_TEST_VIDEO=/caminho/partida_curta.mp4 \
SPECTRA_TEST_DEVICE=cuda \
pytest -m gpu -v
```

Prefira um recorte de 2 a 5 segundos; a análise completa é computacionalmente
pesada. O teste confirma a geração do MP4 e do CSV, mas a qualidade das
detecções deve ser comparada com eventos anotados manualmente.

Para medir precisão, recall e F1, copie o modelo localizado em
`tests/fixtures/ground_truth/events_template.csv`, anote os eventos reais e
compare com o CSV produzido:

```bash
python scripts/evaluate_events.py \
  --detected outputs/resultado_stats.csv \
  --ground-truth minha_anotacao.csv \
  --tolerance 1.0
```

## Estrutura

- `src/spectra/`: pacote principal, pipeline, configuração e servidor Flask
- `src/tcc_common/`: detecção de eventos, rastreamento e classificação
- `src/tcc_annotators/`: componentes de anotação visual
- `src/tcc_configs/`: configurações Python compartilhadas
- `main.py` e `app.py`: pontos de entrada do pipeline e da interface web
- `spectra_ui/`: interface web
- `config/config.yaml`: caminhos e parâmetros
- `tests/`: testes unitários, de integração e métricas
- `data/models/`: modelos locais ignorados pelo Git
- `data/videos/`: vídeos locais ignorados pelo Git
- `outputs/`: resultados locais ignorados pelo Git

Modelos, vídeos, resultados e ambientes virtuais não devem ser versionados devido ao tamanho. Eles permanecem disponíveis localmente.
