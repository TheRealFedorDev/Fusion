#include "library.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

Tensor::Tensor(const std::vector<float>& data, const std::vector<int>& shape)
    : data(data), shape(shape) {
    // Проверка на совпадение размера
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
        throw std::invalid_argument("Tensor size mismatch");
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

namespace py = pybind11;

PYBIND11_MODULE(_fusion_core, m) {
    m.doc() = "FusionCore: high-performance ML library with Python simplicity";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<float>&, const std::vector<int>&>(),
             py::arg("data"), py::arg("shape") = std::vector<int>{},
             "Create a Tensor from a flat list of data and a shape")
        .def("sum", &Tensor::sum, "Sum of all elements")
        .def("__add__", &Tensor::add, "Element-wise addition")
        .def("__mul__", &Tensor::mul, "Multiply by scalar")
        .def_readonly("shape", &Tensor::shape, "Shape of the tensor");
}