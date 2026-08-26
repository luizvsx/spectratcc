# Modelos

Coloque neste diretório os quatro pesos exigidos pelo pipeline:

- `football-player-detection.pt`
- `football-ball-detection.pt`
- `football-pitch-detection.pt`
- `mobile_sam.pt`

Os pesos não fazem parte do Git comum porque alguns ultrapassam o limite de
arquivo do GitHub. Depois de obtê-los, valide a integridade a partir da raiz:

```bash
sha256sum -c data/models/models.sha256
```
