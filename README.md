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

## Dataset
[Link para o dataset](https://zenodo.org/records/7191171)  

O dataset é composto por dois arquivos principais:

| **Arquivo**      | **Descrição**                                                                 |
|------------------|-------------------------------------------------------------------------------|
| **Attribute.csv** | Contém os dados de 8.000 amostras de válvulas. Cada linha representa os valores de pressão padronizados em 201 timestamps: `pressure timestamp 1 | pressure timestamp 2 | ... | pressure timestamp 201`. |
| **Label.csv**     | Contém os rótulos de cada amostra de válvula. As classes são definidas como segue:  |

### Classes de Válvulas com Cores
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
      <td>Charge</td>
      <td>Single-Fault</td>
    </tr>
    <tr style="background-color:#FFD3B6;">
      <td>2</td>
      <td>Discharge</td>
      <td>Single-Fault</td>
    </tr>
    <tr style="background-color:#FFD3B6;">
      <td>3</td>
      <td>Friction</td>
      <td>Single-Fault</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>4</td>
      <td>Charge Discharge</td>
      <td>Multi-Fault</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>5</td>
      <td>Charge Friction</td>
      <td>Multi-Fault</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>6</td>
      <td>Discharge Friction</td>
      <td>Multi-Fault</td>
    </tr>
    <tr style="background-color:#FF8C94;">
      <td>7</td>
      <td>Charge Discharge Friction</td>
      <td>Multi-Fault</td>
    </tr>
  </tbody>
</table>

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
