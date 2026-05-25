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
    float mean() const;   // сраф ёпт

    // Поэлементное сложение
    Tensor add(const Tensor& other) const;

    // Умножение на скаляр
    Tensor mul(float scalar) const;
    Tensor mul(const Tensor& other) const;   // поэлементное умножение двух тензоров

    Tensor subtract(const Tensor& other) const;   // вычитание матриц
    Tensor divide(float scalar) const;    // деление на скаляр
    Tensor matmul(const Tensor& other) const;   // перемножение матриц
    Tensor transpose() const;  // транспонирование
    Tensor pow(int power) const;     // возведение матрицы в степень


    std::string toString() const;



    static Tensor zeros(const std::vector<int>& shape);   // заполнение нулями

};

#endif //FUSIONCORE_LIBRARY_H