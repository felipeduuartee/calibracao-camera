# Calibração de câmera

Trabalho da disciplina de Visão Computacional.

O programa estima os parâmetros intrínsecos e os coeficientes de distorção de uma câmera a partir de fotografias de um tabuleiro. Foram usadas 12 imagens em orientação retrato, todas com 1152x1536 pixels.

O tabuleiro possui 8x8 casas e 7x7 cantos internos. A área quadriculada mede 150x205 milímetros, sem incluir a moldura de madeira. Portanto, cada casa mede 18,75x25,625 milímetros.

A calibração foi feita com o modelo de cinco coeficientes do OpenCV: `k1`, `k2`, `p1`, `p2` e `k3`. Como o padrão de 7x7 cantos é simétrico e as casas são retangulares, o programa também seleciona automaticamente, para cada vista, qual eixo detectado corresponde ao lado de 25,625 milímetros. O erro RMS de reprojeção obtido foi de 0,4287 pixel.

## Compilação

Dependências:

    sudo apt install build-essential cmake libopencv-dev

Compilação:

    cmake -S . -B build
    cmake --build build -j

## Execução

As 12 imagens usadas na calibração estão em `imagens/calibracao`.

Para executar:

    ./build/calibra_camera imagens/calibracao

Os parâmetros estimados, as detecções dos cantos e a comparação antes e depois da correção são gravados em `resultados`.

## Resultado

Matriz intrínseca:

    [1048.8223,    0.0000, 570.2664]
    [   0.0000, 1064.5923, 782.9960]
    [   0.0000,    0.0000,   1.0000]

Coeficientes de distorção na ordem `[k1, k2, p1, p2, k3]`:

    [0.104262, -0.282980, 0.003527, -0.009270, 0.228913]

## Arquivos

- `src/`: código-fonte
- `imagens/calibracao/`: imagens usadas no experimento
- `resultados/calibracao.yml`: parâmetros intrínsecos, distorção e erros por imagem
- `resultados/deteccoes/`: cantos internos detectados nas 12 imagens
- `resultados/prancha_deteccoes.jpg`: visão conjunta das 12 detecções
- `resultados/comparacao_distorcao.jpg`: comparação visual antes e depois da correção
