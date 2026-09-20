#include "calibracao.hpp"

#include <filesystem>
#include <limits>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

namespace {

const cv::Size tamanhoPadrao(7, 7);
const float ladoCurto = 150.0f / 8.0f;
const float ladoLongo = 205.0f / 8.0f;

void adicionaRotulo(cv::Mat& imagem, const std::string& texto)
{
    cv::putText(
        imagem,
        texto,
        cv::Point(20, 42),
        cv::FONT_HERSHEY_SIMPLEX,
        1.1,
        cv::Scalar(0, 255, 255),
        3,
        cv::LINE_AA
    );
}

}

bool detectaCantos(
    const cv::Mat& imagem,
    std::vector<cv::Point2f>& cantos
)
{
    if (imagem.empty())
        throw std::runtime_error("Imagem vazia.");

    cv::Mat cinza;
    cv::cvtColor(imagem, cinza, cv::COLOR_BGR2GRAY);

    int opcoes =
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE;

    bool encontrado = cv::findChessboardCorners(
        cinza,
        tamanhoPadrao,
        cantos,
        opcoes
    );

    if (!encontrado) {
        int opcoesPrecisas =
            cv::CALIB_CB_NORMALIZE_IMAGE |
            cv::CALIB_CB_EXHAUSTIVE |
            cv::CALIB_CB_ACCURACY;

        encontrado = cv::findChessboardCornersSB(
            cinza,
            tamanhoPadrao,
            cantos,
            opcoesPrecisas
        );
    }

    if (!encontrado)
        return false;

    cv::cornerSubPix(
        cinza,
        cantos,
        cv::Size(5, 5),
        cv::Size(-1, -1),
        cv::TermCriteria(
            cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
            60,
            1e-4
        )
    );

    return true;
}

std::vector<cv::Point3f> montaPontosTabuleiro(bool eixoLongoEmX)
{
    float passoX = eixoLongoEmX ? ladoLongo : ladoCurto;
    float passoY = eixoLongoEmX ? ladoCurto : ladoLongo;

    std::vector<cv::Point3f> pontos;
    pontos.reserve(tamanhoPadrao.area());

    for (int y = 0; y < tamanhoPadrao.height; ++y) {
        for (int x = 0; x < tamanhoPadrao.width; ++x) {
            pontos.emplace_back(
                static_cast<float>(x) * passoX,
                static_cast<float>(y) * passoY,
                0.0f
            );
        }
    }

    return pontos;
}

void determinaOrientacoes(std::vector<VistaCalibracao>& vistas)
{
    if (vistas.empty())
        throw std::runtime_error("Nenhuma vista para orientar.");

    const std::size_t quantidade = vistas.size();

    std::vector<std::vector<bool>> sementes;
    sementes.push_back(std::vector<bool>(quantidade, false));
    sementes.push_back(std::vector<bool>(quantidade, true));

    for (int periodo : {2, 3, 4}) {
        std::vector<bool> semente(quantidade);

        for (std::size_t i = 0; i < quantidade; ++i)
            semente[i] = static_cast<int>(i) % periodo == 0;

        sementes.push_back(semente);

        for (std::size_t i = 0; i < quantidade; ++i)
            semente[i] = !semente[i];

        sementes.push_back(semente);
    }

    double melhorErro = std::numeric_limits<double>::infinity();
    std::vector<bool> melhoresOrientacoes;

    for (std::vector<bool> orientacoes : sementes) {
        for (std::size_t i = 0; i < quantidade; ++i) {
            vistas[i].eixoLongoEmX = orientacoes[i];
            vistas[i].pontosObjeto = montaPontosTabuleiro(orientacoes[i]);
        }

        double erroAtual = calibraCamera(vistas).erroRms;
        bool alterou = true;

        while (alterou) {
            alterou = false;

            for (std::size_t i = 0; i < quantidade; ++i) {
                orientacoes[i] = !orientacoes[i];
                vistas[i].eixoLongoEmX = orientacoes[i];
                vistas[i].pontosObjeto = montaPontosTabuleiro(orientacoes[i]);

                double erroTeste = calibraCamera(vistas).erroRms;

                if (erroTeste + 1e-9 < erroAtual) {
                    erroAtual = erroTeste;
                    alterou = true;
                }
                else {
                    orientacoes[i] = !orientacoes[i];
                    vistas[i].eixoLongoEmX = orientacoes[i];
                    vistas[i].pontosObjeto = montaPontosTabuleiro(orientacoes[i]);
                }
            }
        }

        if (erroAtual < melhorErro) {
            melhorErro = erroAtual;
            melhoresOrientacoes = orientacoes;
        }
    }

    for (std::size_t i = 0; i < quantidade; ++i) {
        vistas[i].eixoLongoEmX = melhoresOrientacoes[i];
        vistas[i].pontosObjeto = montaPontosTabuleiro(melhoresOrientacoes[i]);
    }
}

ResultadoCalibracao calibraCamera(
    const std::vector<VistaCalibracao>& vistas
)
{
    if (vistas.empty())
        throw std::runtime_error("Nenhuma vista para calibracao.");

    std::vector<std::vector<cv::Point3f>> pontosObjeto;
    std::vector<std::vector<cv::Point2f>> pontosImagem;

    pontosObjeto.reserve(vistas.size());
    pontosImagem.reserve(vistas.size());

    for (const VistaCalibracao& vista : vistas) {
        pontosObjeto.push_back(vista.pontosObjeto);
        pontosImagem.push_back(vista.cantosImagem);
    }

    ResultadoCalibracao resultado;
    resultado.tamanhoImagem = vistas[0].imagem.size();
    resultado.matrizCamera = cv::Mat::eye(3, 3, CV_64F);
    resultado.coeficientesDistorcao = cv::Mat::zeros(5, 1, CV_64F);

    cv::Mat desviosIntrinsecos;
    cv::Mat desviosExtrinsecos;
    cv::Mat errosPorVista;

    resultado.erroRms = cv::calibrateCamera(
        pontosObjeto,
        pontosImagem,
        resultado.tamanhoImagem,
        resultado.matrizCamera,
        resultado.coeficientesDistorcao,
        resultado.vetoresRotacao,
        resultado.vetoresTranslacao,
        desviosIntrinsecos,
        desviosExtrinsecos,
        errosPorVista
    );

    resultado.errosPorImagem.reserve(errosPorVista.total());

    for (std::size_t i = 0; i < errosPorVista.total(); ++i)
        resultado.errosPorImagem.push_back(errosPorVista.at<double>(static_cast<int>(i)));

    return resultado;
}

void salvaResultados(
    const ResultadoCalibracao& resultado,
    const std::vector<VistaCalibracao>& vistas,
    const std::string& diretorioSaida
)
{
    fs::path saida(diretorioSaida);
    fs::path saidaDeteccoes = saida / "deteccoes";

    fs::create_directories(saidaDeteccoes);

    cv::FileStorage arquivo(
        (saida / "calibracao.yml").string(),
        cv::FileStorage::WRITE
    );

    if (!arquivo.isOpened())
        throw std::runtime_error("Nao foi possivel salvar a calibracao.");

    arquivo << "largura_imagem" << resultado.tamanhoImagem.width;
    arquivo << "altura_imagem" << resultado.tamanhoImagem.height;
    arquivo << "cantos_internos" << "[" << 7 << 7 << "]";
    arquivo << "area_quadriculada_mm" << "[" << 150.0 << 205.0 << "]";
    arquivo << "dimensoes_casa_mm" << "[" << ladoCurto << ladoLongo << "]";
    arquivo << "erro_rms_pixels" << resultado.erroRms;
    arquivo << "matriz_camera" << resultado.matrizCamera;
    arquivo << "coeficientes_distorcao" << resultado.coeficientesDistorcao;
    arquivo << "ordem_coeficientes" << "k1, k2, p1, p2, k3";
    arquivo << "vistas" << "[";

    for (std::size_t i = 0; i < vistas.size(); ++i) {
        arquivo << "{";
        arquivo << "arquivo" << vistas[i].nomeArquivo;
        arquivo << "eixo_longo_em_x" << vistas[i].eixoLongoEmX;
        arquivo << "erro_reprojecao_pixels" << resultado.errosPorImagem[i];
        arquivo << "vetor_rotacao" << resultado.vetoresRotacao[i];
        arquivo << "vetor_translacao_mm" << resultado.vetoresTranslacao[i];
        arquivo << "}";
    }

    arquivo << "]";
    arquivo.release();

    std::vector<cv::Mat> miniaturas;
    miniaturas.reserve(vistas.size());

    for (const VistaCalibracao& vista : vistas) {
        cv::Mat marcada = vista.imagem.clone();

        cv::drawChessboardCorners(
            marcada,
            tamanhoPadrao,
            vista.cantosImagem,
            true
        );

        fs::path nomeSaida =
            saidaDeteccoes /
            (fs::path(vista.nomeArquivo).stem().string() + "_cantos.jpg");

        if (!cv::imwrite(nomeSaida.string(), marcada))
            throw std::runtime_error("Nao foi possivel salvar uma deteccao.");

        cv::Mat miniatura;
        cv::resize(marcada, miniatura, cv::Size(288, 384));
        cv::putText(
            miniatura,
            vista.nomeArquivo,
            cv::Point(8, 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.65,
            cv::Scalar(0, 255, 255),
            2,
            cv::LINE_AA
        );
        miniaturas.push_back(miniatura);
    }

    std::vector<cv::Mat> linhas;

    for (std::size_t inicio = 0; inicio < miniaturas.size(); inicio += 4) {
        std::vector<cv::Mat> grupo(
            miniaturas.begin() + static_cast<std::ptrdiff_t>(inicio),
            miniaturas.begin() + static_cast<std::ptrdiff_t>(inicio + 4)
        );

        cv::Mat linha;
        cv::hconcat(grupo, linha);
        linhas.push_back(linha);
    }

    cv::Mat prancha;
    cv::vconcat(linhas, prancha);

    if (!cv::imwrite((saida / "prancha_deteccoes.jpg").string(), prancha)) {
        throw std::runtime_error("Nao foi possivel salvar a prancha.");
    }

    std::size_t indiceExemplo = vistas.size() / 2;
    const cv::Mat& original = vistas[indiceExemplo].imagem;

    cv::Mat novaMatriz = cv::getOptimalNewCameraMatrix(
        resultado.matrizCamera,
        resultado.coeficientesDistorcao,
        resultado.tamanhoImagem,
        1.0,
        resultado.tamanhoImagem
    );

    cv::Mat corrigida;
    cv::undistort(
        original,
        corrigida,
        resultado.matrizCamera,
        resultado.coeficientesDistorcao,
        novaMatriz
    );

    cv::Mat originalComRotulo = original.clone();
    cv::Mat corrigidaComRotulo = corrigida.clone();

    adicionaRotulo(originalComRotulo, "ORIGINAL");
    adicionaRotulo(corrigidaComRotulo, "SEM DISTORCAO");

    cv::Mat comparacao;
    cv::hconcat(originalComRotulo, corrigidaComRotulo, comparacao);

    if (!cv::imwrite(
            (saida / "comparacao_distorcao.jpg").string(),
            comparacao
        )) {
        throw std::runtime_error("Nao foi possivel salvar a comparacao.");
    }
}
