# Detecção de Pessoas com TensorFlow e OpenCV

Este projeto utiliza o modelo **MobileNet-SSD** pré-treinado com **TensorFlow** e **OpenCV** para realizar a detecção de pessoas em tempo real a partir da câmera do computador. O código também inclui um alerta para indicar **sobrecarga de pessoas** quando 3 ou mais pessoas são detectadas.

## Requisitos

Antes de executar o código, certifique-se de ter as seguintes bibliotecas Python instaladas:

- Python 3.x
- `tensorflow`
- `opencv-python`
- `kagglehub`
- `numpy`

## Como rodar o código

Siga os passos abaixo para configurar o ambiente e rodar o código.

### 1. Clone o repositório

Clone o repositório em sua máquina local:

```bash
git clone https://github.com/Thiago-Pininga/DetecPessoas.git
cd DetecPessoas
```

### 2. Crie e ative um ambiente virtual

Crie um ambiente virtual para isolar as dependências do projeto:

```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

Com o ambiente virtual ativado, instale as dependências necessárias:

```bash
pip install tensorflow opencv-python numpy kagglehub
```

### 4. Execute o codigo
