#include "ml/data/csv.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

pair<TensorPtr, TensorPtr> read_csv(const string& path, int y_col, bool header){

    ifstream file(path);
    string line;

    if (header) getline(file, line);

    vector<vector<float>> rows;

    while (getline(file, line)){
        vector<float> row;
        stringstream stream(line);
        string val;

        while (getline(stream, val, ',')){
            row.push_back(stof(val));
        }
        rows.push_back(row);
    }

    int n_rows = rows.size(); // number of rows
    int n_cols = rows[0].size(); // number of columns

    if (y_col == -1) y_col = n_cols - 1;

    auto x = make_shared<Tensor>(vector<int>{n_rows, n_cols-1});
    auto y = make_shared<Tensor>(vector<int>{n_rows, 1});

    for (int i = 0; i < n_rows; i++){

        int x_col = 0;

        for (int j = 0; j < n_cols; j++){

            if (j == y_col){
                y->data[i] = rows[i][j]; 
            } else {
                x->data[i * (n_cols - 1) + x_col] = rows[i][j];
                x_col++;
            }

        }
    }

    return {x, y};

}