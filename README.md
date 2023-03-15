## Conda
Installer packages med conda install, ikke pip install. Fx: 

```conda search numpy ``` (se om numpy eksisterer)


```conda install numpy```

Når nye pakker er installeret, så opdater requirements.txt til pip:

```conda list > requirements.txt```

og til conda:

```conda env export > environment.yml```


## Set up environment
Installer dependencies med:

```conda env create -f environment.yml```


Hvis du bruger pip:

 ```pip install -r requirements.txt```

Se evt https://gist.github.com/loic-nazaries/b18a908473935243fc23586f35d4bacc
