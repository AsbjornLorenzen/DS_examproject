## Conda
Installer packages med conda install, ikke pip install. Fx:
```conda search numpy ``` (se om numpy eksisterer)
```conda install numpy```

Når nye pakker er installeret, så opdater requirements.txt:
```conda list > requirements.txt```
Se evt https://gist.github.com/loic-nazaries/b18a908473935243fc23586f35d4bacc

Alle pakker kan installeres med ```pip install -r requirements.txt```