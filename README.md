# PredictCost

Para o dataset de dados sintéticos sobre o custo de manobrar entre diferentes poses, verifique se você incluiu:

- [ ]  As estatísticas relevantes, como o número de exemplos.
- [ ]  Os detalhes das divisões de treino / validação / teste.
- [ ]  Uma explicação de quaisquer dados que foram excluídos e todos os passos de pré-processamento.
- [ ]  Um link para uma versão para download do conjunto de dados ou ambiente de simulação.
- [ ]  Para novos dados coletados, uma descrição completa do processo de coleta de dados, como instruções aos anotadores e métodos de controle de qualidade.

Verificação de data sete sintetico:

- [ ]  Detalhamento da geração do dataset:
    - [ ]  Método de construção dos atributos e do alvo:
        
        A pose inicial e final são escolhidas dentre um set de p
        

---

## dataset

Variáveis: 

- xi, yi, thetai, betai, xf, yf, thetaf, betaf
    - x e y em metros
    - theta e beta em radianos

Target:

Custo real: float

> Detalhes importantes pro artigo sobre o custo
> 
> 
> > Referência de como o custo para cada ação de manobra é normalmente encontrado de forma empírica ou utilizando otimização com múltiplos teste o que é um empírico repetido e comparado;
> > 

- Porque usar um algoritmo de machine learning para isso?
    - Pode ser usado para realizar teste de algoritmos de heurística de maneira muito mais rápida do que executar o planejador para múltiplas combinações para cara variação dos parâmetros da heurística

# Referências

- Algorithm 1 Function Hybrid A*
    
    [arxiv.org](https://arxiv.org/pdf/2111.06739)
    
- A truck-trailer mode
    
    [s40747-021-00330-z (1).pdf](attachment:28b024bb-74b9-495a-a5d2-3c9d1de135e7:s40747-021-00330-z_(1).pdf)
    

---

Modelos:

- **Algoritmo K-vizinhos mais próximos (kNN)**
    - Pros: Não precisa treinar
    - Contra: Para fazer a inferência dentro de milhares de dados vai demorar muito
- **Naive Bayes**
    - Pros:
        - é um algoritmo que é tranquilo de treinar,
        - tem poucos hiper parâmetros,
        - é um modelo bom para ser o baseline
    - Contras:
        - O desempenho do algoritmo é prejudicado quando há muitos atributos
        correlacionados
        - Para dados numéricos, assume uma distribuição Gaussiana que pode não
        caracterizar bem os atributos
- **Árvores de decisão**
    - Pros:
        - Tranquilo de treinar,
        - Rápido na predição
    - Contras:
        - overfitting com arvore muito profunda
- **Ensemble**
    - Usar talvez o Randon forest
- **Support Vector Machines (SVM)**

cantinho da bagunça

**Automotive:** Agent-based modeling can be employed to generate artificial data 
related to traffic flow, helping improve road and transport systems. The
 use of synthetic data can help car manufacturers avoid the costly and 
time-consuming process of obtaining real crash data for vehicle safety 
testing. Makers of autonomous vehicles can use synthetic data to train 
self-driving cars in navigating different scenarios.

Synthetic data and fabricated data are different. If you want to use synthetic data, you need to be transparent about how the data is generated and not tailor the synthetic data to support your hypothesis. Fabrication would be false data that is designed to support your hypothesis.

Remember that any statistical analysis will be a reflection of the process used to generate the data. It might be that the trend in the synthetic data reflects real data, but you would have to make that case by examining how the synthetic data is generated.

https://www.ibm.com/products/blog/synthetic-data-generation-building-trust-by-ensuring-privacy-and-quality

https://www.mdpi.com/2079-9292/13/17/3509

https://arxiv.org/html/2404.07503v1
