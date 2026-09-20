#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct VistaCalibracao {
    std::string nomeArquivo;
    cv::Mat imagem;
    std::vector<cv::Point2f> cantosImagem;
    std::vector<cv::Point3f> pontosObjeto;
    bool eixoLongoEmX;
};

struct ResultadoCalibracao {
    cv::Size tamanhoImagem;
    cv::Mat matrizCamera;
    cv::Mat coeficientesDistorcao;
    std::vector<cv::Mat> vetoresRotacao;
    std::vector<cv::Mat> vetoresTranslacao;
    std::vector<double> errosPorImagem;
    double erroRms;
};

bool detectaCantos(
    const cv::Mat& imagem,
    std::vector<cv::Point2f>& cantos
);

std::vector<cv::Point3f> montaPontosTabuleiro(
    bool eixoLongoEmX
);

void determinaOrientacoes(
    std::vector<VistaCalibracao>& vistas
);

ResultadoCalibracao calibraCamera(
    const std::vector<VistaCalibracao>& vistas
);

void salvaResultados(
    const ResultadoCalibracao& resultado,
    const std::vector<VistaCalibracao>& vistas,
    const std::string& diretorioSaida
);

void salvaExperimentoProjecao(
    const ResultadoCalibracao& resultado,
    const std::vector<VistaCalibracao>& vistas,
    const std::string& diretorioSaida
);
