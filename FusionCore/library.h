#ifndef FUSIONCORE_LIBRARY_H
#define FUSIONCORE_LIBRARY_H

#include <vector>
#include <numeric>
#include <stdexcept>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    // Конструктор: данные + размерность
    Tensor(const std::vector<float>& data, const std::vector<int>& shape);

    // Сумма
    float sum() const;

    // Поэлементное сложение
    Tensor add(const Tensor& other) const;

    // Умножение на скаляр
    Tensor mul(float scalar) const;
};

#endif //FUSIONCORE_LIBRARY_H