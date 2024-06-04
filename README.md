# Coding-examples
Coding examples in Python und R

### Deutsch
*** English below ***
## 01_Daten Visualisierung:

Dieser Ordner enthält R-Markdown Beispiele von früheren Projekten. Es handelt sich dabei um den "first glimps at the data", d.h. um einen groben Überblick über die Daten zu erhalten, habe ich damals alles mögliche angeschaut und via Markdown visualisiert, um ein besseres Verständnis der Daten zu erhalten. 

### 1. Projekt-Cortisol-Inselspital: 
Hierbei handelt es sich um Daten von Patienten, deren klinischer Verlauf nach acht Monaten angeschaut wurde. Die finale Publikation zu dieser Arbeit finden Sie hier: https://doi.org/10.1016/j.jpsychores.2024.111615

Durch das Öffnen von "analysis_CortVBMGen.html" gelangen Sie direkt zum Markdown. Das "analysis_CortVBMGen.Rmd" entspricht dem Code. 


## 02_NLP_Depression_Suicide

Hierbei handelt es sich um eine Serie von NLP (natural language processing) Analysen, welche ich im Rahmen meiner momentanen Anstellung an der Psychiatrischen Universitätsklinik Zürich verfasst habe. Die Daten sind open-access Daten von Reddit, bei welchen eine Suizidalitätseinschätzung anhand der Reddit-Posts gemacht wurde (binary yes/no). 

### Teil 1: 
Im ersten Teil verwende ich simples Machine Learning (support vector machine), um das Suizid Risiko zu klassifizieren. 
### Teil 2: 
Hier verwende ich schwierigere Modelle, i.e., Neurale Netzwerke. Ich verwende dafür Long-Short Term Memory. Zuerst kreire ich meine eigenen Vector Embeddings, danach verwende ich pre-trained embeddings von Python's SpaCy library für Textverarbeitung. Zuletzt erlaube ich, dass diese Embeddings weiter trainiert werden können (Transfer Learning). 
### Teil 3: 
Im dritten Teil verwende ich die Transformer Architektur von Google's BERT Language Modell für die Datenverarbeitung der Reddit Posts. 
### Teil 4: 
Im letzten Teil fine-tune ich das Modell basierend auf den Suizidalitätsdaten, um Reddit-Posts zu erstellen. Diese werden eingeteilt nach Posts "at risk" oder nicht "at risk", d.h. es werden Posts generiert, die auf Suizidalität hinweisen, und vice versa. 

## 03_MRT_Toolbox

Hierbei handelt es sich um die Weiterentwicklung einer bereits existierenden MRT Toolbox (Tb_CAPs). Um die Datenverarbeitung effizienter zu machen, wurde eine Datenreduktion eingebaut (PCA), welche die Verarbeitungszeit (Pre-Processing) von 160 Probanden von ~2.5 Wochen auf ~6h reduziert hat. 
Die Publikation hierzu: https://doi.org/10.1016/j.nicl.2024.103583

### English:
## 01_Data Visualization:

This folder contains R-Markdown examples from previous projects. These are the “first glimpses at the data”, i.e. to get a rough overview of the data and visualized it via Markdown to get a better understanding of the data. 

#1 Project Cortisol Island Hospital: 
This is data from patients whose clinical course was looked at after eight months. The final publication on this work can be found here: https://doi.org/10.1016/j.jpsychores.2024.111615

Opening “analysis_CortVBMGen.html” will take you directly to the Markdown. The “analysis_CortVBMGen.Rmd” corresponds to the code. 


## 02_NLP_Depression_Suicide

This is a series of NLP (natural language processing) analyses that I adapted from tutorials as part of my current employment at the Psychiatric University Hospital Zurich. The data are open-access data from Reddit, where a suicidality assessment was made based on the Reddit posts (binary yes/no). 

### Part 1: 
In the first part, I use simple machine learning (support vector machine) to classify the suicide risk. 
### Part 2: 
Here I use more difficult models, i.e., neural networks. I use long-short term memory for this. First I create my own vector embeddings, then I use pre-trained embeddings from Python's SpaCy library for text processing. Finally, I allow these embeddings to be trained further (transfer learning). 
### Part 3: 
In the third part, I use the Transformer architecture of Google's BERT Language model for data processing of Reddit posts. 
### Part 4: 
In the final part, I fine-tune the model based on the suicidal data to create Reddit posts. These are categorized by posts “at risk” or not “at risk”, i.e. posts indicating suicidal tendencies are generated and vice versa. 

## 03_MRT_Toolbox

This is a further development of an existing MRT toolbox (Tb_CAPs). In order to make data processing more efficient, a data reduction was incorporated (PCA), which reduced the processing time (pre-processing) of 160 subjects from ~2.5 weeks to ~6 hours. 
The publication on this: https://doi.org/10.1016/j.nicl.2024.103583

Translated with DeepL.com (free version)

