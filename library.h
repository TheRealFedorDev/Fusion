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
    Tensor sum(int axis) const; // сумма по осям

    float mean() const;   // сраф ёпт
    Tensor mean(int axis) const; // сраф по осям (ого)


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
    Tensor pow(float power) const;
    Tensor divide(const Tensor& other) const;

    std::string toString() const;



    static Tensor zeros(const std::vector<int>& shape);   // заполнение нулями



    // операторы
    Tensor operator/(float scalar) const;
};

#endif //FUSIONCORE_LIBRARY_H