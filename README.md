# Diagnóstico de Câncer de Mama com Dados Sintéticos e Aprendizado de Máquina

## Visão Geral

Este repositório apresenta o trabalho desenvolvido no contexto de um **Trabalho de Conclusão de Curso (TCC)** cujo objetivo é **auxiliar médicos no diagnóstico do câncer de mama** por meio da aplicação de **dados sintéticos** e **técnicas de aprendizado de máquina**, com foco em imagens de mamografias.

Com o avanço recente das abordagens de *Deep Learning*, especialmente das **Redes Neurais Convolucionais (CNNs)**, surge a possibilidade de identificar padrões e características sutis em imagens médicas que muitas vezes não são perceptíveis visualmente por especialistas humanos.

---

## Objetivo

Investigar se a utilização de **imagens sintéticas de alta qualidade** pode **ampliar conjuntos de dados médicos limitados** e melhorar o desempenho de modelos de classificação automática de câncer de mama, contribuindo para diagnósticos mais robustos e confiáveis.

---

## Metodologia

### Geração de Dados Sintéticos

Devido à escassez de dados médicos rotulados e de alta qualidade, foi desenvolvido um **conjunto de dados sintéticos** utilizando o modelo generativo:

* **StyleGAN2-ADA condicional**

Para garantir a qualidade e a fidelidade visual das imagens sintéticas em relação às imagens reais, foi aplicada uma filtragem baseada na métrica:

* **LPIPS (Learned Perceptual Image Patch Similarity)**

Essa etapa assegura **alta similaridade perceptual** entre os dados sintéticos e reais.

---

### Conjunto de Dados Reais

O conjunto final de dados reais é composto por **imagens de mamografias balanceadas** entre as classes:

* Benignas
* Malignas

Essas imagens foram utilizadas **exclusivamente na etapa de avaliação**, garantindo que os resultados reflitam desempenho em dados reais.

---

### Modelo de Classificação

Para a tarefa de classificação, foi utilizada a arquitetura:

* **EfficientNet-B0**

O modelo foi avaliado por meio de:

* **Validação cruzada estratificada de 5 dobras (5-fold)**

---

### Estratégia Experimental

Diferentes proporções de imagens sintéticas foram incorporadas ao conjunto de treino, mantendo a avaliação restrita apenas às imagens reais:

* 1:1 (sintéticas:reais)
* 2:1
* 3:1
* 4:1

Essa estratégia permitiu analisar o impacto progressivo da inclusão de dados sintéticos no desempenho do classificador.

---

## Resultados

Os experimentos demonstraram que a **inclusão de dados sintéticos** promoveu **melhoria consistente nas métricas de desempenho**.

### Melhor desempenho observado (proporção 2:1):

* **Acurácia média:** 0.651
* **F1-score:** 0.658
* **AUC:** 0.703

### Comparação com uso exclusivo de dados reais:

* **Acurácia:** 0.606
* **F1-score:** 0.573
* **AUC:** 0.665

Esses resultados indicam que dados sintéticos de alta qualidade podem **potencializar o aprendizado**, resultando em modelos com **maior capacidade de generalização**.

---

## Contribuições

* Demonstração do potencial de **dados sintéticos** para mitigar a escassez de dados médicos
* Melhoria no desempenho de classificadores para diagnóstico de câncer de mama
* Apoio ao desenvolvimento de **sistemas computacionais de suporte à decisão médica**

---

## Observações sobre os Dados

Devido a limitações de armazenamento e privacidade:

* O **dataset real** não é disponibilizado neste repositório
* As **imagens sintéticas completas** e **checkpoints de treinamento** também não são versionados

Instruções para obtenção ou reprodução dos dados podem ser fornecidas mediante solicitação.

---

## Licença

Este projeto é disponibilizado exclusivamente para **fins acadêmicos e de pesquisa**.

---

## Autor

**Rafael Palheta Tokairin**
Trabalho de Conclusão de Curso (TCC)
