#include "library.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape)
    : data(data), shape(shape) {
    size_t expected = 1;
    for (int d : shape) expected *= d;
    if (data.size() != expected) {
        throw std::invalid_argument("Data size does not match shape");
    }
}

float Tensor::sum() const {
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

Tensor Tensor::add(const Tensor& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Tensor size mismatch in add");
    }
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] + other.data[i];
    }
    return Tensor(result, shape);
}

Tensor Tensor::mul(float scalar) const {
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] * scalar;
    }
    return Tensor(result, shape);
}

Tensor Tensor::mul(const Tensor& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Tensor size mismatch in mul");
    }
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] * other.data[i];
    }
    return Tensor(result, shape);
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (data.size() != other.data.size()) {
        throw std::invalid_argument("Tensor size mismatch in subtract");
    }
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] - other.data[i];
    }
    return Tensor(result, shape);
}

Tensor Tensor::divide(float scalar) const {
    if (scalar == 0.0f) throw std::invalid_argument("Division by zero");
    std::vector<float> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] / scalar;
    }
    return Tensor(result, shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() != 2 || other.shape.size() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors");
    }
    int m = shape[0];
    int k = shape[1];
    int k2 = other.shape[0];
    int n = other.shape[1];
    if (k != k2) {
        throw std::invalid_argument("Incompatible shapes for matmul");
    }
    std::vector<float> result(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += data[i * k + p] * other.data[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    return Tensor(result, {m, n});
}

Tensor Tensor::transpose() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("transpose requires 2D tensor");
    }
    int m = shape[0];
    int n = shape[1];
    std::vector<float> result(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j * m + i] = data[i * n + j];
        }
    }
    return Tensor(result, {n, m});
}

Tensor Tensor::pow(int power) const {
    std::vector<float> result(data.size());
    if (power == 0) {
        std::fill(result.begin(), result.end(), 1.0f);
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            result[i] = std::pow(data[i], static_cast<float>(power));
        }
    }
    return Tensor(result, shape);
}

std::string Tensor::toString() const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "], data=[";

    size_t max_show = 30;
    for (size_t i = 0; i < data.size() && i < max_show; ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(4) << data[i];
    }
    if (data.size() > max_show) {
        oss << ", ...";
    }
    oss << "])";
    return oss.str();

}

float Tensor::mean() const {
    return sum() / static_cast<float>(data.size());
}

Tensor Tensor::zeros(const std::vector<int>& shape) {
    size_t size = 1;
    for (int d : shape) size *= d;
    std::vector<float> data(size, 0.0f);
    return Tensor(data, shape);
}

namespace py = pybind11;

PYBIND11_MODULE(_fusion_core, m) {
    m.doc() = "Fusion: high-performance ML library with Python simplicity";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<float>&, const std::vector<int>&>(),
             py::arg("data"), py::arg("shape") = std::vector<int>{},
             "Create a Tensor from flat data and shape")
        .def("sum", &Tensor::sum, "Sum of all elements")
        .def("__add__", &Tensor::add, "Element-wise addition")
        .def("__mul__", [](const Tensor& self, const py::object& other) -> Tensor {
            if (py::isinstance<py::float_>(other) || py::isinstance<py::int_>(other)) {
                return self.mul(other.cast<float>());
            } else if (py::isinstance<Tensor>(other)) {
                return self.mul(other.cast<Tensor>());
            } else {
                throw std::invalid_argument("Unsupported operand type for *");
            }
            }, "Multiply by scalar or another tensor")
        .def("__sub__", &Tensor::subtract, "Element-wise subtraction")
        .def("__truediv__", &Tensor::divide, "Divide by scalar")
        .def("matmul", &Tensor::matmul, "Matrix multiplication")
        .def("transpose", &Tensor::transpose, "Transpose 2D tensor")
        .def("mean", &Tensor::mean, "Mean of all elements")
        .def("__pow__", &Tensor::pow, "Raise each element to an integer power")
        .def("__repr__", &Tensor::toString, "String representation of the tensor")
        .def("tolist", [](const Tensor& self) -> std::vector<float> {
            return self.data;
        }, "Returns data as Python list")
        .def_static("zeros", &Tensor::zeros, "Create a tensor filled with zeros")
        .def_readonly("shape", &Tensor::shape, "Shape of the tensor");

}