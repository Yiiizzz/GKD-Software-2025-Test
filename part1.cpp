#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// 矩阵类
class Matrix {
public:
    vector<vector<float>> data;
    int rows, cols;

    Matrix(int r, int c) {
        rows = r;
        cols = c;
        data = vector<vector<float>>(r, vector<float>(c, 0));
    }

    // 矩阵加法
    Matrix add(Matrix other) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // 矩阵乘法
    Matrix multiply(Matrix other) {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                float sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }
};

// ReLU函数
Matrix relu(Matrix m) {
    Matrix result(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            if (m.data[i][j] < 0) {
                result.data[i][j] = 0;
            } else {
                result.data[i][j] = m.data[i][j];
            }
        }
    }
    return result;
}

// Softmax函数
vector<float> softmax(Matrix m) {
    vector<float> vec; 

    // 如果是行向量
    if (m.rows == 1) {
        vec = m.data[0];
    }
    // 如果是列向量
    else if (m.cols == 1) {
        vec = vector<float>(m.rows);
        for (int i = 0; i < m.rows; i++) {
            vec[i] = m.data[i][0];
        }
    }

    int size = vec.size();
    vector<float> eval(size);
    float sum = 0;

    for (int i = 0; i < size; i++) {
        eval[i] = exp(vec[i]);
        sum += eval[i];
    }

    vector<float> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = eval[i] / sum;
    }
    return result;
}

// model类
class model {
public:
    Matrix w1, b1, w2, b2;

    model() : w1(784, 500), b1(1, 500), w2(500, 10), b2(1, 10) {}

    // forward函数
    vector<float> forward(Matrix input) {
        Matrix x1 = input.multiply(w1);
        Matrix x2 = x1.add(b1);
        Matrix x3 = relu(x2);
        Matrix x4 = x3.multiply(w2);
        Matrix x5 = x4.add(b2);
        vector<float> x6 = softmax(x5);

        return x6;
    }
};

int main() {
    model model;
    Matrix input(1, 784);
    
    vector<float> output = model.forward(input);

    for (int i = 0; i < output.size(); i++) {
        cout << output[i] << " ";
    }
    cout << endl;

    return 0;
}

