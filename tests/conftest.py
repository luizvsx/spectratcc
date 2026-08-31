import os


# Evita que bibliotecas de visualização tentem escrever na pasta pessoal.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spectra-matplotlib")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spectra-config")
