

![](Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.001.png)[](https://medium.com/?source=---two_column_layout_nav----------------------------------)





![](Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.002.png)[](https://medium.com/me/notifications?source=---two_column_layout_nav----------------------------------)

![Thommaskevin]

**TinyML — Árvore de Decisão**

Dos fundamentos matemáticos até a implementação prática.

![Thommaskevin][](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)
[](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)
[](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)

[Thommaskevin](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)

8 min read

·

Dec 11, 2023





![](Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.005.png)



![ref1]



**Redes sociais**

👨🏽‍💻 **Github:** [thommaskevin/TinyML (github.com)](https://github.com/thommaskevin/TinyML)
👷🏾 **Linkedin:** [Thommas Kevin | LinkedIn](https://www.linkedin.com/in/thommas-kevin-ab9810166/)
📽 **Youtube: [Thommas Kevin — YouTube](https://www.youtube.com/channel/UC7uazGXaMIE6MNkHg4ll9oA)**
👨🏻‍🏫 **Research group**: [Conecta.ai (ufrn.br)](https://conect2ai.dca.ufrn.br/)

![ref2]

**O que é uma árvore de decisão?**

Um algoritmo de aprendizado de máquina supervisionado conhecido como árvore de decisão é empregado tanto para classificação quanto para regressão. Isso significa que pode ser utilizado para prever categorias discretas, como “sim” ou “não”, e para antecipar valores numéricos, como o lucro em reais.

Similar a um fluxograma, a árvore de decisão estabelece nós de decisão que estão hierarquicamente interligados. O nó-raiz, considerado o mais crucial, está associado a um atributo da base de dados, enquanto os nós-folha representam os resultados desejados. No contexto do aprendizado de máquina, o nó-folha corresponde à classe ou ao valor que será gerado como resposta.

![ref2]

Figura 1 — Árvore de decisão para entender o risco de ataque do coração.

Na interconexão dos nós, são aplicadas regras de “se-então”. Ao alcançar um nó A, o algoritmo avalia uma condição, por exemplo, “se a característica X do registro analisado for menor que 15?”. Se for o caso, o algoritmo prossegue para um lado da árvore; se for maior, direciona-se para o outro lado. Esse processo continua nos próximos nós, seguindo a mesma lógica.

O algoritmo adota uma abordagem conhecida como “recursiva” em computação, repetindo o mesmo padrão à medida que avança para níveis mais profundos. É como se uma função chamasse a si mesma para uma execução paralela, na qual a resposta dessa segunda função é crucial para a primeira.

O papel crucial da árvore reside em determinar quais nós ocuparão cada posição. Quem será o nó raiz? Em seguida, quais serão os nós à esquerda e à direita? Para isso, são realizados cálculos fundamentais, frequentemente usando métricas como **ganho de informação e entropia**. Essas variáveis estão relacionadas à desorganização e à falta de uniformidade nos dados. **Maior entropia indica dados mais caóticos, enquanto menor entropia indica uma base mais uniforme e homogênea.**

Na definição das posições, o algoritmo calcula a entropia das classes de saída e o ganho de informação dos atributos da base de dados. O atributo com maior ganho de informação torna-se o nó raiz. Para determinar os nós à esquerda e à direita, são realizados cálculos adicionais de entropia e ganho com o conjunto de dados que atende à condição específica para cada ramificação.

**Entropia**

A entropia pode ser conceituada como a medida que indica o nível de desorganização e mistura em nossos dados. Quanto maior a entropia, menor é o ganho de informação, e o oposto também é verdadeiro. À medida que dividimos os dados em conjuntos capazes de representar exclusivamente uma classe do nosso modelo, a entropia diminui, resultando em dados menos desorganizados.

![ref2]

Figura 2 — Entropia

A partir deste ponto, nosso objetivo é construir a árvore, tendo o conjunto de dados completo como raiz e estabelecendo ramificações com base em condições que minimizem a entropia e maximizem o ganho de informação. A entropia de um conjunto de dados pode ser calculada usando a seguinte fórmula:

![Figura 2 — Fórmula da Entropia de Shannon][ref2]

Equação 1 — Fórmula da Entropia de Shannon

A entropia geralmente varia de 0 a n-1, assumindo seu valor máximo quando as probabilidades de ocorrência de cada classe, representadas como *p*(*xi*) na fórmula, são iguais.

**Ganho de Informação**

O ganho de informação é uma métrica que indica quão eficaz uma característica do conjunto de dados consegue separar os registros de acordo com suas classes. Esse valor é determinado ao compararmos a entropia do estado atual dos dados com a entropia que seria alcançada ao criar uma nova ramificação. A fórmula para calcular o ganho de informação é a seguinte:

![ref2]

Equação 2— Ganho de informação

A quantidade de filhos presente na soma depende de como a ramificação será realizada, sendo comum gerar dois nós filhos. A ponderação para cada um desses filhos é determinada ao dividir o total de elementos em um nó filho pelo número de elementos no nó pai:

![ref2]

Equação 3 — Peso do filho

Ao compreender como calcular o ganho de informação, procedemos calculando-o para cada uma das características em nosso conjunto de dados e comparamos os resultados para identificar aquela que obteve o maior ganho. Assim, a ramificação é estabelecida com base na coluna que demonstrou o ganho mais significativo, e o processo é repetido de maneira recursiva para cada lado da ramificação. O encerramento ocorre quando o ganho de informação atinge o valor de 0.

**Até que ponto a Árvore de Decisão pode crescer?**

O objetivo primordial de uma Árvore de Decisão é atingir uma entropia tão baixa quanto possível. Para alcançar esse objetivo, a árvore precisa expandir-se e formar várias ramificações, o que é benéfico, mas até certo ponto.

A altura da árvore, ou seja, o número de níveis horizontais de ramificações, é crucial para a adequação do modelo aos dados de treinamento. Em última análise, se a árvore alcançar uma altura excessiva, isso pode levar ao *overfitting* do modelo, resultando em previsões imprecisas quando confrontado com dados do mundo real.

**Árvore de Decisão em TinyML**

**1. instale o pacote [micromlgen](https://eloquentarduino.com/libraries/micromlgen/) com:**

pip install micromlgen

**2. Importe as biblioteca**

from micromlgen import port
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load\_iris
from sklearn.model\_selection import train\_test\_split
from sklearn import metrics

**3. Carrego o Dataset**

O conjunto de dados Iris é um conjunto clássico na área de aprendizado de máquina e estatísticas. Ele foi introduzido por Sir Ronald A. Fisher em 1936 como um exemplo de análise discriminante. O conjunto de dados é frequentemente utilizado para fins educacionais e é um ponto de partida comum para a prática de classificação de padrões.

Atributos:

\- Comprimento da sépala (em centímetros)

\- Largura da sépala (em centímetros)

\- Comprimento da pétala (em centímetros)

\- Largura da pétala (em centímetros)

Espécies:

0 — Setosa

1 — Versicolor

2 — Virginica

X, y = load\_iris(return\_X\_y=True)

print('Shape da entrada: ', X.shape)
print('Shape da variável alvo: ', y.shape)

print(X[:10])

print(y[:10])

**4. Separação em dados de treinamento e teste**

Separo 30% dos dados para teste e 70% dos dados para treinamento

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size = 0.3)

**5. Crio o modelo de classificação**

modelo = DecisionTreeClassifier(criterion = "entropy", splitter="best")

criterion{“gini”, “entropy”, “log\_loss”}, default=”gini”} — A função para medir a qualidade de uma divisão. Os critérios compatíveis são “gini” para a impureza de Gini e “log\_loss” e “entropy” para o ganho de informações de Shannon, consulte Formulação matemática.

splitter{“best”, “random”}, default=”best”} — A estratégia usada para escolher a divisão em cada nó. As estratégias compatíveis são “best” para escolher a melhor divisão e “random” para escolher a melhor divisão aleatória.

**6. Treinamento do modelo**

modelo.fit(X\_train, y\_train)

**7. Avaliação do modelo com os dados de treinamento**

training\_predict = modelo.predict(X\_train)

print(metrics.classification\_report(y\_train, training\_predict, digits = 3))

print(metrics.confusion\_matrix(y\_train, training\_predict)) 

**8. Avaliação do modelo com os dados de teste**

test\_predict = modelo.predict(X\_test)

print(metrics.classification\_report(y\_test, test\_predict, digits = 3))

print(metrics.confusion\_matrix(y\_test, test\_predict))

**9. Exibição da Ávore de Decisão criada**

Importação das bibliotecas

from sklearn.tree import export\_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

Definição das features

feature\_cols = ["Comprimento da sépala", "Largura da sépala", "Comprimento da pétala", "Largura da pétala"]

Exibição do gráfico

dot\_data = StringIO()
export\_graphviz(modelo, out\_file=dot\_data,  
`                `filled=True, rounded=True,
`                `special\_characters=True,feature\_names = feature\_cols,class\_names=['0','1','2'])
graph = pydotplus.graph\_from\_dot\_data(dot\_data.getvalue())  
Image(graph.create\_png())

![ref2]

Figura 3 — Árvore de Decisão dataset Iris

**9. Obtenção do modelo para ser implantando no microcontrolador**

print(port(modelo))

Resultado

#pragma once
#include <cstdarg>
namespace Eloquent {
`    `namespace ML {
`        `namespace Port {
`            `class DecisionTree {
`                `public:
`                    `/\*\*
`                    `\* Predict class for features vector
`                    `\*/
`                    `int predict(float \*x) {
`                        `if (x[2] <= 2.449999988079071) {
`                            `return 0;
`                        `}
`                        `else {
`                            `if (x[3] <= 1.6500000357627869) {
`                                `if (x[2] <= 4.950000047683716) {
`                                    `return 1;
`                                `}
`                                `else {
`                                    `if (x[3] <= 1.550000011920929) {
`                                        `return 2;
`                                    `}
`                                    `else {
`                                        `return 1;
`                                    `}
`                                `}
`                            `}
`                            `else {
`                                `if (x[2] <= 4.8500001430511475) {
`                                    `if (x[1] <= 3.100000023841858) {
`                                        `return 2;
`                                    `}
`                                    `else {
`                                        `return 1;
`                                    `}
`                                `}
`                                `else {
`                                    `return 2;
`                                `}
`                            `}
`                        `}
`                    `}
`                `protected:
`                `};
`            `}
`        `}
`    `}

Cópia apenas a função “int predict(float \*x)” para o Sketch [Arduino](https://www.arduino.cc/):

int predict(float \*x) {
`                        `if (x[2] <= 2.449999988079071) {
`                            `return 0;
`                        `}
`                        `else {
`                            `if (x[3] <= 1.6500000357627869) {
`                                `if (x[2] <= 4.950000047683716) {
`                                    `return 1;
`                                `}
`                                `else {
`                                    `if (x[0] <= 6.150000095367432) {
`                                        `if (x[1] <= 2.450000047683716) {
`                                            `return 2;
`                                        `}
`                                        `else {
`                                            `return 1;
`                                        `}
`                                    `}
`                                    `else {
`                                        `return 2;
`                                    `}
`                                `}
`                            `}
`                            `else {
`                                `if (x[2] <= 4.8500001430511475) {
`                                    `if (x[1] <= 3.100000023841858) {
`                                        `return 2;
`                                    `}
`                                    `else {
`                                        `return 1;
`                                    `}
`                                `}
`                                `else {
`                                    `return 2;
`                                `}
`                            `}
`                        `}
`                    `};

**10. Sketch Arduino Completo**

int predict(float \*x) {
`                        `if (x[2] <= 2.449999988079071) {
`                            `return 0;
`                        `}
`                        `else {
`                            `if (x[3] <= 1.6500000357627869) {
`                                `if (x[2] <= 4.950000047683716) {
`                                    `return 1;
`                                `}
`                                `else {
`                                    `if (x[0] <= 6.150000095367432) {
`                                        `if (x[1] <= 2.450000047683716) {
`                                            `return 2;
`                                        `}
`                                        `else {
`                                            `return 1;
`                                        `}
`                                    `}
`                                    `else {
`                                        `return 2;
`                                    `}
`                                `}
`                            `}
`                            `else {
`                                `if (x[2] <= 4.8500001430511475) {
`                                    `if (x[1] <= 3.100000023841858) {
`                                        `return 2;
`                                    `}
`                                    `else {
`                                        `return 1;
`                                    `}
`                                `}
`                                `else {
`                                    `return 2;
`                                `}
`                            `}
`                        `}
`                    `};
void setup() {
`  `Serial.begin(115200);
}
void loop() {
`  `float X\_1[] = {5.1, 3.5, 1.4, 0.2};
`  `int result\_1 = predict(X\_1);
`  `Serial.print("Result of predict with input X1:");
`  `Serial.println(result\_1);
`  `delay(2000);
`  `float X\_2[] = {6.2, 2.2, 4.5, 1.5};
`  `int result\_2 = predict(X\_2);
`  `Serial.print("Result of predict with input X2:");
`  `Serial.println(result\_2); 
`  `delay(2000);

`  `float X\_3[] = {6.1, 3.0, 4.9, 1.8};
`  `int result\_3 = predict(X\_3);
`  `Serial.print("Result of predict with input X3:");
`  `Serial.println(result\_3);
`  `delay(2000);
}

**11. Resultado**

0 — Setosa — Entrada: {6.1, 3.0, 4.9, 1.8}

1 — Versicolor — Entrada:{6.2, 2.2, 4.5, 1.5}

2 — Virginica — Entrada:{6.1, 3.0, 4.9, 1.8}

![ref2]

Figura 4 — Resultado execução do Sketch na IDE Arduino.

[Tinyml](https://medium.com/tag/tinyml?source=post_page-----aa1414562d97---------------tinyml-----------------)

[Machine Learning](https://medium.com/tag/machine-learning?source=post_page-----aa1414562d97---------------machine_learning-----------------)

[Data Science](https://medium.com/tag/data-science?source=post_page-----aa1414562d97---------------data_science-----------------)

[Tecnology](https://medium.com/tag/tecnology?source=post_page-----aa1414562d97---------------tecnology-----------------)







![Thommaskevin][](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)
[](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)
[](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)



[Written by Thommaskevin](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)

[56 Followers](https://medium.com/@thommaskevin/followers?source=post_page-----aa1414562d97--------------------------------)

I am a Ph.D. Brazilian student in Electrical and Computer Engineering with a specialization in Intelligent Information Processing at the UFRN.



More from Thommaskevin

![TinyML — Convolutional Neural Networks (CNN)][ref2]

![Thommaskevin][](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----0---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----0---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[Thommaskevin](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----0---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[**TinyML — Convolutional Neural Networks (CNN)**](https://medium.com/@thommaskevin/tinyml-convolutional-neural-networks-cnn-3601b32c35f4?source=author_recirc-----aa1414562d97----0---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[From mathematical foundations to edge implementation](https://medium.com/@thommaskevin/tinyml-convolutional-neural-networks-cnn-3601b32c35f4?source=author_recirc-----aa1414562d97----0---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

May 27

![ref3][104](https://medium.com/@thommaskevin/tinyml-convolutional-neural-networks-cnn-3601b32c35f4?source=author_recirc-----aa1414562d97----0---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

![ref4]

![ref1]

![TinyML — XGBoost (Regression)][ref2]

![Thommaskevin][](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[Thommaskevin](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[**TinyML — XGBoost (Regression)**](https://medium.com/@thommaskevin/tinyml-xgboost-regression-d2b513a0d237?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[From mathematical foundations to edge implementation](https://medium.com/@thommaskevin/tinyml-xgboost-regression-d2b513a0d237?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

Jun 7

![ref3][57](https://medium.com/@thommaskevin/tinyml-xgboost-regression-d2b513a0d237?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

![ref5][1](https://medium.com/@thommaskevin/tinyml-xgboost-regression-d2b513a0d237?source=author_recirc-----aa1414562d97----1---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

![ref4]

![ref1]

![TinyML —K-Nearest Neighbors (KNN-Classifier)][ref2]

![Thommaskevin][](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----2---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----2---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[Thommaskevin](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----2---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[**TinyML —K-Nearest Neighbors (KNN-Classifier)**](https://medium.com/@thommaskevin/tinyml-k-nearest-neighbors-knn-classifier-6008f8e51189?source=author_recirc-----aa1414562d97----2---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[From mathematical foundations to edge implementation](https://medium.com/@thommaskevin/tinyml-k-nearest-neighbors-knn-classifier-6008f8e51189?source=author_recirc-----aa1414562d97----2---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

4d ago

![ref3][27](https://medium.com/@thommaskevin/tinyml-k-nearest-neighbors-knn-classifier-6008f8e51189?source=author_recirc-----aa1414562d97----2---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

![ref4]

![ref1]

![TinyML — Poisson Regression][ref2]

![Thommaskevin][](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----3---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----3---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[Thommaskevin](https://medium.com/@thommaskevin?source=author_recirc-----aa1414562d97----3---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[**TinyML — Poisson Regression**](https://medium.com/@thommaskevin/tinyml-poisson-regression-5174d88479f5?source=author_recirc-----aa1414562d97----3---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

[From mathematical foundations to edge implementation](https://medium.com/@thommaskevin/tinyml-poisson-regression-5174d88479f5?source=author_recirc-----aa1414562d97----3---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

Jun 19

![ref3][50](https://medium.com/@thommaskevin/tinyml-poisson-regression-5174d88479f5?source=author_recirc-----aa1414562d97----3---------------------c59c9f36_cc57_4d8f_a3ce_46640d3ca62e-------)

![ref4]

![ref1]

[See all from Thommaskevin](https://medium.com/@thommaskevin?source=post_page-----aa1414562d97--------------------------------)

Recommended from Medium

![Mathematical Understanding of Linear Regression Algorithm][ref2]

![Ansa Baby][Thommaskevin][](https://medium.com/@ansababy?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[](https://medium.com/@ansababy?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Ansa Baby](https://medium.com/@ansababy?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

in

[Tech & TensorFlow](https://medium.com/tech-tensorflow?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[**Mathematical Understanding of Linear Regression Algorithm**](https://medium.com/tech-tensorflow/mathematical-understanding-of-linear-regression-algorithm-7bba82f3d1d8?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[A Step-by-Step Guide to Understanding the Mathematics and Visualization of Linear Regression](https://medium.com/tech-tensorflow/mathematical-understanding-of-linear-regression-algorithm-7bba82f3d1d8?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref6]

5d ago

![ref3][53](https://medium.com/tech-tensorflow/mathematical-understanding-of-linear-regression-algorithm-7bba82f3d1d8?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref4]

![ref1]

![The best Python IDE in 2024?][ref2]

![Simeon Emanuilov][Thommaskevin][](https://medium.com/@simeon.emanuilov?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[](https://medium.com/@simeon.emanuilov?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Simeon Emanuilov](https://medium.com/@simeon.emanuilov?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref7][](https://medium.com/@simeon.emanuilov?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[**The best Python IDE in 2024?**](https://medium.com/@simeon.emanuilov/the-best-python-ide-in-2024-d4f536b6b6a2?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Python, the versatile and beginner-friendly programming language, has taken the world by storm. As more developers embrace Python for…](https://medium.com/@simeon.emanuilov/the-best-python-ide-in-2024-d4f536b6b6a2?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

Jun 18

![ref3][216](https://medium.com/@simeon.emanuilov/the-best-python-ide-in-2024-d4f536b6b6a2?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref5][5](https://medium.com/@simeon.emanuilov/the-best-python-ide-in-2024-d4f536b6b6a2?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref4]

![ref1]

Lists

![ref8][](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

[Predictive Modeling w/ Python](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

[20 stories](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

[·](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

[1332 saves](https://medium.com/@ben.putney/list/predictive-modeling-w-python-e3668ea008e1?source=read_next_recirc-----aa1414562d97--------------------------------)

![Principal Component Analysis for ML][ref8][](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

![Time Series Analysis][ref8][](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

![deep learning cheatsheet for beginner][ref8][](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

[Practical Guides to Machine Learning](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

[10 stories](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

[·](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

[1611 saves](https://destingong.medium.com/list/practical-guides-to-machine-learning-a877c2a39884?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

![are long context LLM better?][ref8][](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

[Natural Language Processing](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

[1556 stories](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

[·](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

[1090 saves](https://medium.com/@AMGAS14/list/natural-language-processing-0a856388a93a?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

![ref8][](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

[data science and AI](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

[40 stories](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

[·](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

[194 saves](https://medium.com/@grexe/list/data-science-and-ai-35d21381d956?source=read_next_recirc-----aa1414562d97--------------------------------)

![Stop Wasting Your Time][ref2]

![John Gorman][Thommaskevin][](https://medium.com/@johnfgorman?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[](https://medium.com/@johnfgorman?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[John Gorman](https://medium.com/@johnfgorman?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[**Stop Wasting Your Time**](https://medium.com/@johnfgorman/stop-wasting-your-time-3ee28e1e5b92?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[A Simple Framework for Making Better Decisions](https://medium.com/@johnfgorman/stop-wasting-your-time-3ee28e1e5b92?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref6]

Jun 4

![ref3][8.6K](https://medium.com/@johnfgorman/stop-wasting-your-time-3ee28e1e5b92?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref5][142](https://medium.com/@johnfgorman/stop-wasting-your-time-3ee28e1e5b92?source=read_next_recirc-----aa1414562d97----0---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref4]

![ref1]

![Beyond the Basics: An In-Depth Guide to Optimization Functions, Back Propagation and Gradient…][ref2]

![Mayuri Deshpande][Thommaskevin][](https://medium.com/@0192.mayuri?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[](https://medium.com/@0192.mayuri?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Mayuri Deshpande](https://medium.com/@0192.mayuri?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[**Beyond the Basics: An In-Depth Guide to Optimization Functions, Back Propagation and Gradient…**](https://medium.com/@0192.mayuri/beyond-the-basics-an-in-depth-guide-to-optimization-functions-back-propagation-and-gradient-09604e83c6f0?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Optimization functions are the backbone of training neural networks. They guide the process of minimizing the loss function, allowing…](https://medium.com/@0192.mayuri/beyond-the-basics-an-in-depth-guide-to-optimization-functions-back-propagation-and-gradient-09604e83c6f0?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

Jun 18

![ref3][1](https://medium.com/@0192.mayuri/beyond-the-basics-an-in-depth-guide-to-optimization-functions-back-propagation-and-gradient-09604e83c6f0?source=read_next_recirc-----aa1414562d97----1---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref4]

![ref1]

![How good is Raspberry Pi’s AI Kit][ref2]

![Gandhi KT][Thommaskevin][](https://medium.com/@gandhikte?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[](https://medium.com/@gandhikte?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Gandhi KT](https://medium.com/@gandhikte?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref7][](https://medium.com/@gandhikte?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

in

[Generative AI](https://medium.com/generative-ai?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[**How good is Raspberry Pi’s AI Kit**](https://medium.com/generative-ai/how-good-is-raspberry-pis-ai-kit-3a6d65884bee?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Powered by Hailo, giving 13 TOPS, for $70, I would say not bad…](https://medium.com/generative-ai/how-good-is-raspberry-pis-ai-kit-3a6d65884bee?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref6]

Jun 18

![ref3][316](https://medium.com/generative-ai/how-good-is-raspberry-pis-ai-kit-3a6d65884bee?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref5][6](https://medium.com/generative-ai/how-good-is-raspberry-pis-ai-kit-3a6d65884bee?source=read_next_recirc-----aa1414562d97----2---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref4]

![ref1]

![4 Must Read Python Books To Boost Your Skills By 10000%][ref2]

![Axel Casas, PhD Candidate][Thommaskevin][](https://medium.com/@axel.em.casas?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[](https://medium.com/@axel.em.casas?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Axel Casas, PhD Candidate](https://medium.com/@axel.em.casas?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

in

[Python in Plain English](https://medium.com/python-in-plain-english?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[**4 Must Read Python Books To Boost Your Skills By 10000%**](https://medium.com/python-in-plain-english/4-must-read-python-books-to-boost-your-skills-by-10000-b4314c206eda?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

[Start learning Python easily, fast, and while having fun](https://medium.com/python-in-plain-english/4-must-read-python-books-to-boost-your-skills-by-10000-b4314c206eda?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref6]

May 9

![ref3][1.5K](https://medium.com/python-in-plain-english/4-must-read-python-books-to-boost-your-skills-by-10000-b4314c206eda?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref5][10](https://medium.com/python-in-plain-english/4-must-read-python-books-to-boost-your-skills-by-10000-b4314c206eda?source=read_next_recirc-----aa1414562d97----3---------------------3edcdf36_78e8_46b1_a5f8_72dd0b8cb4f4-------)

![ref4]

![ref1]

[See more recommendations](https://medium.com/?source=post_page-----aa1414562d97--------------------------------)





















![](Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.016.png)![](Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.017.png)

[Thommaskevin]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.003.png
[Thommaskevin]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.004.png
[ref1]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.006.png
[ref2]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.007.png
[Thommaskevin]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.008.png
[Thommaskevin]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.009.png
[ref3]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.010.png
[ref4]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.011.png
[ref5]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.012.png
[ref6]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.013.png
[ref7]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.014.png
[ref8]: Aspose.Words.29007bb6-3335-4041-8c19-1efd7cb1fe57.015.png
