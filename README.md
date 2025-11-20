# Disciplina: Inteligência Computacional - GBC073  
**Grupo:** Eduarda Lopes, Carla Azevedo, Lucas Matos  

---

## Resumo do Enunciado do Trabalho
Reproduzir e analisar experimentos de um artigo que utilize apenas redes neurais (MLP ou CNN) em problemas reais, utilizando o mesmo dataset ou parte dele.  
Extrair arquitetura, parâmetros e métricas do artigo, testar variações justificadas para melhorar resultados, e comparar com o baseline.  
Apresentar objetivo, relevância, metodologia, resultados, comparação com o artigo, conclusões e código.  
Cada grupo deve trabalhar com artigo e problema diferentes, respeitando as restrições de dataset e técnica.

---

## Artigo
**Título:** Application of Deep Learning Models for Aircraft Maintenance  
**Autores:** Humberto Hayashi Sano, Lilian Berton  
**Publicação:** Anais do XIX Encontro Nacional de Inteligência Artificial e Computacional (ENIAC) - 2022  
[Link para o artigo](https://sol.sbc.org.br/index.php/eniac/article/view/22832)  

---

## Análise Inicial do Artigo
O estudo utiliza um conjunto de dados contendo 8.000 amostras de pressão regulada, obtidas a partir do funcionamento natural do Environmental Control System (ECS) em aeronaves.  

Essas amostras correspondem a válvulas do tipo **Pressure Regulated Shutoff Valves (PRSOV)**, que, devido à operação sob altas pressões e temperaturas, são suscetíveis a **falhas isoladas (single-faults)** ou **múltiplas (multi-faults)**.  

O objetivo dos autores é avaliar a eficácia de redes neurais, especificamente **Convolutional Neural Networks (CNN)** e **Multi-Layer Perceptron (MLP)**, na classificação do estado dessas válvulas, utilizando 201 recortes de pressão por válvula para prever a ocorrência de falhas simples ou múltiplas.  

O uso de **pressão regulada**, em contraste com estudos anteriores que se basearam em tempos de abertura e fechamento adequados apenas para testes em bancada, representa uma abordagem mais realista para monitoramento em operação.  

Como contribuição, o artigo disponibiliza um dataset padronizado, criado variando parâmetros como **coeficiente de fricção, nível de entupimento e falha de mola**, permitindo simular diferentes estados da válvula:  

- **Healthy:** 0  
- **Single-Faults:** 1, 2, 3  
- **Multi-Faults:** 4, 5, 6, 7  

Essa variação possibilita a análise do comportamento das válvulas sob diferentes condições de falha, fornecendo uma base robusta para a validação de métodos de diagnóstico de falhas em sistemas aeronáuticos.

---

## Dados e Geração
Os dados utilizados no estudo **não foram obtidos a partir de voos reais**, mas sim por meio de **simulações** executadas em um **modelo Simulink previamente validado**. Para cada amostra do conjunto de dados, foi simulado um ciclo completo de comando da válvula **PRSOV**. A geração dos dados envolveu a variação controlada dos principais parâmetros físicos internos da válvula: **friction**, **charge** e **discharge**.

A classificação das falhas foi definida com base em **conhecimento especializado** sobre o funcionamento das válvulas PRSOV. Especialistas estabeleceram **intervalos válidos** (normais e anormais) para cada parâmetro físico. A partir desses intervalos, foram criadas duas listas por parâmetro, uma contendo valores normais e outra contendo valores anormais. Em seguida, uma **distribuição uniforme** foi utilizada para selecionar, de forma aleatória, os valores que comporiam cada amostra.

Esse procedimento assegura que o dataset tenha **grande diversidade de cenários**, evitando que as redes neurais “decorem” os exemplos e garantindo que elas, de fato, aprendam a reconhecer **padrões de comportamento da pressão associados a falhas**.

No dataset, **cada linha corresponde a uma amostra simulada**, enquanto **cada coluna representa o valor da pressão em um instante de tempo específico**. Assim, cada amostra é composta por **201 valores de pressão**, que constituem as **features** que alimentam os modelos. Dessa forma, o dataset caracteriza-se como uma **série temporal**, representada por um vetor unidimensional de tamanho 201 — valor que corresponde ao comprimento máximo do ciclo temporal simulado.

[Link para o dataset](https://zenodo.org/records/7191171)  

O dataset é composto por dois arquivos principais:

| **Arquivo**      | **Descrição**                                                                 |
|------------------|-------------------------------------------------------------------------------|
| **Attribute.csv** | Contém os dados de 8.000 amostras de válvulas. Cada linha representa os valores de pressão temporais em 201 timestamps.                             |
| **Label.csv**     | Contém os rótulos de cada amostra de válvula.            |

### Classes de Válvulas
<table>
  <thead>
    <tr>
      <th>Label</th>
      <th>Classe da Válvula</th>
      <th>Categoria</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color:#A8E6CF;">
      <td>0</td>
      <td>Normal</td>
      <td>Healthy</td>
    </tr>
    <tr style="background-color:#FFD3B6;">
      <td>1</td>
      <td>Isolated Failure: Charge Fault</td>
      <td>Falha única na câmara de Carga</td>
    </tr>
    <tr style="background-color:#FFD3B6;">
      <td>2</td>
      <td>Isolated Failure: Discharge Fault</td>
      <td>Falha única na câmara de Descarga</td>
    </tr>
    <tr style="background-color:#FFD3B6;">
      <td>3</td>
      <td>Isolated Failure: Friction Fault</td>
      <td>Falha única de fricção</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>4</td>
      <td>Simultaneous: Charge and Discharge</td>
      <td>Falha dupla (Carga + Descarga)</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>5</td>
      <td>Simultaneous: Charge and Friction</td>
      <td>Falha dupla (Carga + Fricção)</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>6</td>
      <td>Simultaneous: Discharge and Friction</td>
      <td>Falha dupla (Descarga + Fricção)</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>7</td>
      <td>Simultaneous: All Faults</td>
      <td>Falha tripla - as 3 falhas simultaneamente</td>
    </tr>
  </tbody>
</table>

### Pré-Processamento dos Dados
Os valores brutos de pressão apresentam grande variabilidade, podendo sofrer variações superiores a dez vezes entre timestamps consecutivos. Trabalhar com dados não normalizados dificulta o treinamento de redes neurais, pois intervalos muito amplos tornam a identificação de padrões menos eficiente. Portanto, a primeira etapa consistiu na normalização dos valores por timestamp utilizando a fórmula Z-Score:

$$
P_{N_i} = \frac{P_i - \overline{P_i}}{sd(P_i)}
$$

Aqui, \( P_i \) é o conjunto de valores de pressão das válvulas no timestamp \( i \),  
e \( \overline{P_i} \) e \( sd(P_i) \) são, respectivamente, a média e o desvio padrão de \( P_i \). O dataset já foi disponibilizado ao público normalizado, então não foi necessário nenhum tipo de pré-processamento posterior.

Após a normalização, é necessário estruturar a saída das redes neurais, já que elas não retornam diretamente a classe em formato textual. Como o problema possui 8 classes distintas, a camada final das redes contém 8 neurônios. Para representar essas classes, utilizou-se a codificação One Hot, em que apenas um dos neurônios é ativado (valor 1) para cada amostra, indicando a classe correspondente; todos os demais permanecem com valor 0. A tabela com as classes PRSOV codificadas em formato One Hot está apresentada a seguir.
![Tabela das Classes Hot Encoded](images/tabela_1_hotencoded.png) 

---