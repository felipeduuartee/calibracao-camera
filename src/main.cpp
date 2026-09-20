#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "calibracao.hpp"

namespace fs = std::filesystem;

namespace {

bool ehImagem(const fs::path& caminho)
{
    std::string extensao = caminho.extension().string();

    std::transform(
        extensao.begin(),
        extensao.end(),
        extensao.begin(),
        [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        }
    );

    return extensao == ".jpg" ||
           extensao == ".jpeg" ||
           extensao == ".png" ||
           extensao == ".bmp" ||
           extensao == ".tif" ||
           extensao == ".tiff";
}

std::vector<fs::path> listaImagens(const fs::path& diretorio)
{
    if (!fs::is_directory(diretorio))
        throw std::runtime_error("Diretorio de imagens inexistente.");

    std::vector<fs::path> imagens;

    for (const auto& item : fs::directory_iterator(diretorio)) {
        if (item.is_regular_file() && ehImagem(item.path()))
            imagens.push_back(item.path());
    }

    std::sort(imagens.begin(), imagens.end());

    if (imagens.empty())
        throw std::runtime_error("Nenhuma imagem encontrada.");

    return imagens;
}

}

int main(int argc, char** argv)
{
    fs::path entrada =
        argc >= 2 ? argv[1] : "imagens/calibracao";

    fs::path saida =
        argc >= 3 ? argv[2] : "resultados";

    try {
        std::vector<fs::path> caminhos = listaImagens(entrada);

        if (caminhos.size() != 12)
            throw std::runtime_error("Sao esperadas exatamente 12 imagens.");

        std::vector<VistaCalibracao> vistas;
        vistas.reserve(caminhos.size());

        cv::Size tamanhoImagem;

        for (const fs::path& caminho : caminhos) {
            cv::Mat imagem = cv::imread(
                caminho.string(),
                cv::IMREAD_COLOR
            );

            if (imagem.empty())
                throw std::runtime_error("Falha ao abrir " + caminho.string());

            if (tamanhoImagem.empty())
                tamanhoImagem = imagem.size();
            else if (imagem.size() != tamanhoImagem)
                throw std::runtime_error("As imagens devem ter a mesma resolucao.");

            std::vector<cv::Point2f> cantos;

            if (!detectaCantos(imagem, cantos)) {
                throw std::runtime_error(
                    "Tabuleiro nao detectado em " + caminho.filename().string()
                );
            }

            vistas.push_back({
                caminho.filename().string(),
                imagem,
                cantos,
                montaPontosTabuleiro(false),
                false
            });

            std::cout
                << caminho.filename().string()
                << ": 49 cantos detectados\n";
        }

        determinaOrientacoes(vistas);

        ResultadoCalibracao resultado = calibraCamera(vistas);
        salvaResultados(resultado, vistas, saida.string());

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nErro RMS: " << resultado.erroRms << " pixel\n";
        std::cout << "\nMatriz da camera:\n" << resultado.matrizCamera << '\n';
        std::cout
            << "\nCoeficientes de distorcao:\n"
            << resultado.coeficientesDistorcao
            << '\n';
        std::cout << "\nResultados gravados em " << saida.string() << '\n';
    }
    catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
