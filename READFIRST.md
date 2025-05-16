Ik gebruik een conda venv, pip geeft vaak errors met ml modules

```{bash}
conda create -n envname
conda activate envname
conda install --file requirements.txt
```

als niet alles gedownlaod is evt nog:

```{bash}
pip install -r requirements.txt
```

dan download die de resterende bestanden, succes!