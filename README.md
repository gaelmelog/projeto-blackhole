# Simulação de Ray Tracing: Buraco Negro de Schwarzschild - ainda em teste

Este projeto realiza a visualização de um buraco negro estático (Schwarzschild) utilizando técnicas de **Ray Tracing** e integração numérica. O código simula o desvio da luz (lensing gravitacional) e os efeitos relativísticos no disco de acreção.

## 🌌 Conceitos de Física Aplicados

A simulação baseia-se na métrica de Schwarzschild, considerando os seguintes marcos teóricos:

### 1. Horizonte de Eventos e Órbita de Fótons
O raio de Schwarzschild ($r_s$) define o ponto de não retorno, enquanto a esfera de fótons ($r_{ph}$) define onde a luz pode orbitar o buraco negro:
* **Raio de Schwarzschild:** $r_s = \frac{2GM}{c^2}$
* **Esfera de Fótons:** $r_{ph} = 1.5 \cdot r_s$

### 2. Desvio Gravitacional (Redshift)
A luz que escapa das proximidades do buraco negro perde energia, alterando sua cor aparente:
$$z + 1 = \frac{1}{\sqrt{1 - \frac{r_s}{r}}}$$

### 3. Efeito Doppler Relativístico
Como o disco de acreção gira a velocidades relativísticas, aplicamos o fator de Doppler para ajustar o brilho (beaming) e a cor:
$$D = \frac{1}{\gamma (1 - \beta \cos \theta)}$$

## 🛠️ Tecnologias e Dependências
* **Linguagem:** Python 3
* **Bibliotecas:** * `NumPy`: Para cálculos matriciais e integração das geodésicas.
    * `Matplotlib`: Para geração e visualização da imagem final.

## 🚀 Como Executar
1. Ative o seu ambiente virtual:
   ```bash
   source venv/bin/activate
