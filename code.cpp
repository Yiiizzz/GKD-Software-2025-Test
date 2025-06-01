#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "include/json.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
using namespace cv;
using json = nlohmann::json;
using namespace std;

// 矩阵类
template<typename T>
class Matrix {
public:
    vector<vector<T>> data;
    int rows, cols;

    Matrix(int r, int c) {
        rows = r;
        cols = c;
        data = vector<vector<T>>(r, vector<T>(c, 0));
    }

    // 矩阵加法
    Matrix<T> add(Matrix<T> other) {
        Matrix<T> result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // 矩阵乘法
    Matrix<T> multiply(Matrix<T> other, int num_threads = 8) {
    Matrix<T> result(rows, other.cols);

    auto work = [&](int start_col, int end_col) {
        for (int i = 0; i < rows; i++) {
            for (int j = start_col; j < end_col; j++){ 
                float sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
    };

    vector<thread> threads;
    int every = other.cols / num_threads;
    int left = other.cols % num_threads;
    int now = 0;

    for (int t = 0; t < num_threads; t++) {
        int start = now;
        int extra = 0;
        if (t < left) {
            extra = 1;
        }
        int end = start + every + extra;
        threads.push_back(thread(work, start, end));
        now = end;
    }

    for (auto thread : threads) {
        thread.join();
    }

    return result;
}

    void load_from_file(string filename) {
        ifstream file(filename, ios::binary);
        for (int i = 0; i < rows; i++){
           for (int j = 0; j < cols; j++){
              file.read(reinterpret_cast<char*>(&data[i][j]), sizeof(T));
           }
        }
        file.close();
    }
};

// ReLU函数
template<typename T>
Matrix<T> relu(Matrix<T> m) {
    Matrix<T> result(m.rows, m.cols);
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
template<typename T>
vector<float> softmax(Matrix<T> m) {
    vector<float> vec; 

    // 如果是行向量
    if (m.rows == 1) {
        vec.resize(m.cols);
        for (int i = 0; i < m.cols; ++i) {
        vec[i] = static_cast<float>(m.data[0][i]);
    }
}

    // 如果是列向量
    else if (m.cols == 1) {
        vec.resize(m.rows);
        for (int i = 0; i < m.rows; i++) {
            vec[i] = static_cast<float>(m.data[i][0]);
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
template<typename T>
class model {
public:
    Matrix<T> w1, b1, w2, b2;

    model() : w1(1, 1), b1(1, 1), w2(1, 1), b2(1, 1) {

        // 读取 meta.json
        ifstream meta_file("mnist-fc-plus/meta.json");
        json meta;
        meta_file >> meta;

        int w1_r = meta["fc1.weight"][0];
        int w1_c = meta["fc1.weight"][1];
        int b1_r = meta["fc1.bias"][0];
        int b1_c = meta["fc1.bias"][1];
        int w2_r = meta["fc2.weight"][0];
        int w2_c = meta["fc2.weight"][1];
        int b2_r = meta["fc2.bias"][0];
        int b2_c = meta["fc2.bias"][1];

        w1 = Matrix<T>(w1_r, w1_c);  w1.load_from_file("mnist-fc-plus/fc1.weight");
        b1 = Matrix<T>(b1_r, b1_c);  b1.load_from_file("mnist-fc-plus/fc1.bias");
        w2 = Matrix<T>(w2_r, w2_c);  w2.load_from_file("mnist-fc-plus/fc2.weight");
        b2 = Matrix<T>(b2_r, b2_c);  b2.load_from_file("mnist-fc-plus/fc2.bias");
    }

    // forward函数
    vector<float> forward(Matrix<T> input, int threads = 8) {
        Matrix<T> x1 = input.multiply(w1, threads);
        Matrix<T> x2 = x1.add(b1);
        Matrix<T> x3 = relu(x2);
        Matrix<T> x4 = x3.multiply(w2, threads);
        Matrix<T> x5 = x4.add(b2);
        vector<float> x6 = softmax(x5);

        return x6;
    }
};

int main() {
    model<double> model;

    cout << "w1 size: " << model.w1.rows << " x " << model.w1.cols << endl;
    cout << "b1 size: " << model.b1.rows << " x " << model.b1.cols << endl;
    cout << "w2 size: " << model.w2.rows << " x " << model.w2.cols << endl;
    cout << "b2 size: " << model.b2.rows << " x " << model.b2.cols << endl;

    string folder = "nums";
    vector<string> pnglist = {
    "nums/0.png", "nums/1.png", "nums/2.png", "nums/3.png",
    "nums/4.png", "nums/5.png", "nums/6.png", "nums/7.png",
    "nums/8.png", "nums/9.png"
};
for (int i = 0; i < pnglist.size(); i++) {
    string path = pnglist[i];

        Mat img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "can't read picture: " << path << endl;
        continue;
    }
  
        //缩放为28x28
        Mat png;
        resize(img, png, Size(28, 28));
  
        //拍扁
        Matrix<double> input(1, 784);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input.data[0][i * 28 + j] = png.at<uchar>(i, j) / 255.0f;
            }
        }


        auto start = std::chrono::high_resolution_clock::now();
        vector<float> output = model.forward(input,8);
        auto end = std::chrono::high_resolution_clock::now();
        // 计算耗时
        std::chrono::duration<double, std::milli> duration = end - start;

        cout << "picture: " << path << endl;

        cout << "forward time: " << duration.count() << " ms" << endl;

        // 输出概率向量
        cout << "probability vector: ";
        for (float prob : output) {
            cout << prob << " ";
        }
        cout << endl;
    }

    return 0;
}
