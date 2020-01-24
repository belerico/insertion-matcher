# Insertion linking

La diffusione su scala mondiale dei servizi di E-Commerce ha generato un mercato dal valore stimato 
per il 2020 di circa 3.9 trilioni di dollari e per il 2021 di 4.5 trilioni di dollari. 
Aziende come Amazon, AliExpress, Ebay producono annualmente terabye di dati e hanno necessità di 
convertire i dati in informazioni di valore. Uno dei problemi che una grande mole di dati tra loro 
eterogenei pone riguarda il matching tra entità (detto **duplicate detection** o **link 
discovery**). Nel presente lavoro analizziamo ed estendiamo tecniche di NLP e modelli di Deep 
Learning allo scopo di risolvere il matching di inserzioni impiegando solo i titoli delle stesse. 

## Dati
Al link https://drive.google.com/drive/folders/1yZCau_ezRftldCn_rbKLnfUi0QsGEYnY?usp=sharing sono
 resi disponibili i dati, gli embeddings, e alcuni dei risultati ottenuti dal lavoro qui 
 presentato. Il dataset è una raccolta opensource di inserzioni, disponibile a http://webdatacommons.org/largescaleproductcorpus/v2/index.html
 
## Implementazione
L'implementazione del modello fa uso del framework *Pytorch*; per l'addestramento dei modelli di 
embeddings abbiamo impiegato Word2Vec e fasttext. Infine, per la pipeline di ricerca di 
iperparametri abbiamo adattato il framework *PyGPGO* alle nostre esigenze

## Risultati
I risultati dei nostri esperimenti e le dovute comparazione con lo stato dell'arte sono esposte 
nella relazione allegata al presente lavoro. 