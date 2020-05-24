#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>

template<typename T>
Matrix<T>::Matrix(unsigned rows_, unsigned cols_, const T initial_) {
    // Parameter Constructor
    mat.resize(rows_);
    for (unsigned i=0; i<mat.size(); ++i) {
        mat[i].resize(cols_, initial_);
    rows = rows_;
    cols = cols_;
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& rhs) {
    // Copy Constructor
    mat = rhs.mat;
    rows = rhs.get_rows();
    cols = rhs.get_cols();
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T> > lst, unsigned rows_,
    unsigned cols_, const T initial_): Matrix(lst.size(), lst.size()+1, initial_) {
    int i = 0;
    int j = 0;
    for (const auto& row : lst) {
        for (const auto& col : row) {
            mat[i][j] = col;
            ++j;
        }
        j = 0;
        ++i;
    }
}

template<typename T>
Matrix<T>::~Matrix() {}

template<typename T>
Matrix<T>& Matrix<T>::operator= (const Matrix<T>& rhs) {
// Assignment Operator
    if (&rhs != this) {
        unsigned new_rows = rhs.get_rows();
        unsigned new_cols = rhs.get_cols();

        mat.resize(new_rows);

        for (unsigned i=0; i<mat.size(); ++i)
            mat[i].resize(new_cols);

        for (unsigned i=0; i<new_rows; ++i)
        for (unsigned j=0; j<new_cols; ++j)
            mat[i][j] = rhs(i, j);

        rows = new_rows;
        cols = new_cols;
    }
    return *this;
}
    
// Addition of two matrices
template<typename T>
Matrix<T> Matrix<T>::operator+ (const Matrix<T>& rhs) {
    if (rows != rhs.rows || cols != rhs.cols) {
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] + rhs(i,j);

    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+= (const Matrix<T>& rhs) {
    if (rows != rhs.rows & cols != rhs.cols) {
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }

    unsigned rows = rhs.get_rows();
    unsigned cols = rhs.get_cols();

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        this->mat[i][j] += rhs(i,j);

    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator- (const Matrix<T>& rhs) {
    if (rows != rhs.rows & cols != rhs.cols) {
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }

    unsigned rows = rhs.get_rows();
    unsigned cols = rhs.get_cols();
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] - rhs(i,j);

    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-= (const Matrix<T>& rhs) {
    if(rows != rhs.rows || cols != rhs.cols){
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }

    unsigned rows = rhs.get_rows();
    unsigned cols = rhs.get_cols();

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        this->mat[i][j] -= rhs(i,j);

    return *this;
}

// Left multiplication of this matrix and another
template<typename T>
Matrix<T> Matrix<T>::operator* (const Matrix<T>& rhs) {
    if (this->cols != rhs.get_rows()) {
        std::cout << "\nmatrices are not congurent" << std::endl;
        std::cout << "\nthis->get_rows()= " << this->get_rows();
        std::cout << " rhs.get_cols()= " << rhs.get_cols();
        std::cout << "\nthis->get_cols()= " << this->get_cols();
        std::cout << " rhs.get_rows()= " << rhs.get_rows();
        exit(1);
    }

    Matrix result(rows, rhs.get_cols(), 0.0);

    for (unsigned i=0; i<this->get_rows(); ++i)
    for (unsigned j=0; j<rhs.get_cols(); ++j)
    for (unsigned k=0; k<this->cols; ++k)
        result.mat[i][j] += this->mat[i][k] * rhs(k,j);

    return result;
}

// Cumulative left multiplication of this matrix and another
template<typename T>
Matrix<T>& Matrix<T>::operator*= (const Matrix<T>& rhs) {
    if (rows != rhs.rows || cols != rhs.cols) {
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }

    Matrix result = (*this) * rhs;
    (*this) = result;

    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::transpose() {
    Matrix result(cols, rows, 0.0);    //flip dimensions

    for (unsigned i=0; i<cols; ++i)
    for (unsigned j=0; j<rows; ++j)
        result(i,j) = this->mat[j][i];

    return result;
}

template<typename T>
void Matrix<T>::swapRows(int i, int j) {
    std::swap(mat[i],mat[j]);
}

template<typename T>
void Matrix<T>::Pivot(int row, int col) {
    //create next pivot element
    std::cout << "Entering Pivot function " << "Row = " << row << " Col = " << col << std::endl;
    T max(0);
    for (int i = row; i < rows; ++i) {
        max = mat[i][col];
        for (int row = i+1; row < rows; ++row) {
            if (max < mat[row][col]) {
                std::swap(mat[i], mat[row]);
                max = mat[i][col];
            }
        }
    }
}



template<typename T>
void Matrix<T>::Permute(int col){
    std::cout << "Entering Permuted function - column col = " << col << std::endl;
    T max(0);
    for (int i = 0; i < rows; ++i) {
        max = mat[i][col];
        for (int row = i+1; row < rows; ++row) {
            if (max < mat[row][col]) {
                std::swap(mat[i], mat[row]);
                max = mat[i][col];
            }
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::PermutationMatrix(int Size, int row1, int row2) {
    Matrix P(Size, Size, T(0));

    for (int i = 0; i < Size; ++i)
        P.set_value(i,i,T(1));        //unity diagonal elements
    std::cout << "finished setting diagonal elements"<< std::endl;
    std::swap(P.mat[row1], P.mat[row2]);

    return P;
}

template<typename T>
Matrix<T> Matrix<T>::block_multiply(const Matrix<T>& rhs) {
    if (this->cols != rhs.rows)
        throw(std::runtime_error("operation doesn't support these dimensions\n"));

    Matrix<T> temp(this->rows, rhs.cols, 0.0);

    for (int i=0; i < this->rows; ++i)
    for (int j=0; j < rhs.cols; ++j)
    for (int k=0; k < this->cols; ++k)
        temp.mat[i][j] += this->mat[i][k] * rhs.mat[k][j];

    return temp;
}

template<typename T>
Matrix<T> Matrix<T>::Inverse() {
    std::cout << "Entering Inverse " << std::endl;
    Matrix<T> temp(*this);

    /* Augmenting Identity Matrix of Order row x col */
    Matrix<T> I(rows, cols, T(0));
    T a = T(0.0);
    T ratio = T(0);

    for(int i = 0; i < cols; ++i)
        I.mat[i][i] = T(1);

    for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
        if (i != j) {
            ratio = mat[j][i]/mat[i][i];
            for (int k = 0; k < cols; k++) {
                mat[j][k] -= ratio * mat[i][k];
                I.mat[j][k] -= ratio * I.mat[i][k];
            }
        }

    for (int i = 0; i < rows; ++i) {
        a = mat[i][i];
        for (int j = 0; j < rows; ++j) {
            mat[i][j] /= a;
            I.mat[i][j] /= a;
        }
    }
    return I;
}


template<typename T>
T Matrix<T>::Determinant()
{
    T dtr(0);
    Matrix<T> submat(rows, cols, T(0));

    for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
         submat.mat[i][j] = this->mat[i][j];

    dtr = determinant(submat, rows, cols);

    return dtr;
}


template<typename T>
T Matrix<T>::determinant(Matrix<T>& submat,int row, int col) {
    Matrix<T> Submat(rows,cols, T(0));
    T det(0);

    if (row == 2)
        det = submat.mat[0][0] * submat.mat[1][1] - submat.mat[1][0] * submat.mat[0][1];
    else
        for (int x = 0; x < row; ++x) {
            T subi = T(0); 
            for (int i = 1; i < row; ++i) {
                T subj = T(0);
                for (int j = 0; j < col; ++j) {
                    if (j == x)
                        continue;
                    Submat.mat[subi][subj] = submat.mat[i][j];
                    ++subj;
                }
                ++subi;
            }
            det = det + pow(-1, x) * mat[0][x] * determinant(Submat, row-1, col);
        }

    return det;
}

template<typename T>
vector<T> Matrix<T>::gauss_jordan(const char& pivoting) {
    if (cols != rows+1)
        throw(runtime_error("*this must be augmented square matrix to call this method"));

    int i;
    int j;
    int k;
    T scalar;
    Matrix<T> M = *this;

    std::cout << "Read matrix as:\n";
    std::cout << M;
    std::cout << '\n';

    switch (pivoting) {
        case 'y': {
            std::cout << "Beginning Gauss elimination with pivoting\n";
            for (k=0; k<rows; ++k) {
                int i_max = k;
                int v_max = M.mat[i_max][k];

                for (i=0; i<rows; ++i)
                    if (fabs(M.mat[i][k]) > v_max) {
                        v_max = M.mat[i][k];
                        i_max = i;
                    }
                if (i_max != k)
                    M.swapRows(k, i_max);

                // annihilate below
                for (i = k+1; i<rows; ++i) {
                    scalar = M.mat[i][k] / M.mat[k][k];
                    for (j = k; j < cols; ++j)
                        M.mat[i][j] -= scalar * M.mat[k][j];
                }
                // annihilate above
                if (k > 0)
                    for (i = k-1; i >= 0; --i) {
                        scalar = M.mat[i][k] / M.mat[k][k];
                        for (j = k; j < cols; ++j)
                            M.mat[i][j] -= scalar * M.mat[k][j];
                    }
                
                std::cout << "Step " << k+1 << '\n' << M << '\n';
                
                // rescale pivot if not 1 (to get identity matrix)
                if (M.mat[k][k] != 1) {
                    T temp = M.mat[k][k];
                    for (i = 0; i < cols; ++i)
                        M.mat[k][i] *= 1 / temp;
                }
            }
            break;
        }
        case 'n': {
            std::cout << "Beginning Gauss elimination without pivoting\n";
            for (k=0; k<rows; ++k) {
                for (i = k+1; i<rows; ++i) {
                    scalar = M.mat[i][k] / M.mat[k][k];
                    for (j = k; j < cols; ++j)
                        M.mat[i][j] -= scalar*M.mat[k][j];
                }
                // annihilate above
                if (k > 0)
                    for (i = k-1; i >= 0; --i) {
                        scalar = M.mat[i][k] / M.mat[k][k];
                        for (j = k; j < cols; ++j)
                            M.mat[i][j] -= scalar*M.mat[k][j];
                    }
                
                // rescale pivot if not 1 (to get identity matrix)
                if (M.mat[k][k] != 1) {
                    T temp = M.mat[k][k];
                    for (i = 0; i < cols; ++i)
                        M.mat[k][i] *= 1 / temp;
                }
            }
            break;
        }
    }

    std::cout << "Reduced to row echelon form:\n";
    std::cout << M;

    std::cout << "\nThe solution is reached for the following coefficients:\n";
    vector<T> soln;
    for (i = 0; i < rows; ++i) {
        std::cout << "X" << i + 1 << " = " << M.mat[i][rows] << '\n';
        soln.push_back(M.mat[i][rows]);
    }     
    return soln;
}

template<typename T>
vector<T> Matrix<T>::jacobi(const T& tolerance, const int& max_iter) {
    Matrix<T> M = *this;
    vector<T> x;    // solution vector
    vector<T> x_old;
    vector<T> b;    // augmented coefficients
    // these variables abide by Mx = b

    int iter = 0;

    for (int i=0; i < rows; ++i) {
        x.push_back(T(0));
        x_old.push_back(x[i]);
        b.push_back(M[i][cols-1]);
    }

    do {
        for (int i = 0; i < rows; ++i) {
            x_old[i] = x[i];

            T bi = b[i];
            T ci = M[i][i];

            for (int j = 0; j < rows; ++j)
                if (j != i)
                    bi -= M[i][j]*x_old[j];

            x[i] = bi/ci;        
        }

        T diff = T(0);
        for (int i = 0; i < rows; ++i) {
            diff += x[i] - x_old[i];
            printf("%7.4f\t", x[i]);
            if (i == rows-1)
                std::cout << '\n';
        }

        if (fabs(diff) < tolerance)
            break;

        ++iter;
    }
    while (iter < max_iter);

    std::cout << "\nSolution with tolerance " << tolerance << " after " << iter << " trials is:\n";
    for (int i = 0; i < rows; ++i)
        std::cout << "X" << i + 1 << " = " << std::setprecision(10) << x[i] << '\n';
    std::cout << "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    return x;
}

template<typename T>
vector<T> Matrix<T>::gauss_seidel(const T& tolerance, const int& max_iter) {
    Matrix<T> M = *this;
    vector<T> x;
    vector<T> x_old;
    vector<T> b;
    int iter = 0;

    for (int i=0; i < rows; ++i) {
        x.push_back(T(0));
        x_old.push_back(T(0));
        b.push_back(M[i][cols-1]);
    }

    do {
        for (int i=0; i < rows; ++i)
            x_old[i] = x[i];

        for (int i=0; i < rows; ++i) {
            T bi = b[i];
            T ci = M[i][i];

            for (int j=0; j < i; ++j)
                bi -= M[i][j] * x[j];
            
            for (int j = i+1; j < rows; ++j)
                bi -= M[i][j] * x_old[j];

            x[i] = bi/ci;
        }

        T diff = T(0);

        for (int i=0; i < rows; ++i) {
            diff += x[i] - x_old[i];
            printf("%7.4f\t", x[i]);
            if (i == rows-1)
                std::cout << '\n';
        }
        if (fabs(diff) < tolerance)
            break;
        ++iter;
    }
    while (iter < max_iter);

    std::cout << "\nSolution with tolerance " << tolerance << " after " << iter << " trials is:\n";
    for (int i = 0; i < rows; ++i)
        std::cout << "X" << i + 1 << " = " << std::setprecision(10) << x[i] << '\n';
    std::cout << "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    return x;
}

template<typename T>
Matrix<T> Matrix<T>::operator+ (const T& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] + rhs;

    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator- (const T& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] - rhs;

    return result;
}

// Matrix/scalar multiplication
template<typename T>
Matrix<T> Matrix<T>::operator* (const T& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat;

    return result;
}

// Matrix/scalar division
template<typename T>
Matrix<T> Matrix<T>::operator/ (const T& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] / rhs;

    return result;
}

// Multiply a matrix with a vector
template<typename T>
vector<T> Matrix<T>::operator* (const vector<T>& rhs) {
    vector<T> result(rhs.size(), 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result[i] = this->mat[i][j] * rhs[j];

    return result;
}

template<typename T>
bool Matrix<T>::operator== (const Matrix<T>& rhs) {
    if (this->rows != rhs.get_rows() || this->cols != rhs.get_cols())
        return false;

    for (int i=0; i < rows; ++i)
    for (int j=0; j < cols; ++j)
        if (this->mat[i][j] != rhs.mat[i][j]) 
            return false;
    return true;
}

template<typename T>
bool Matrix<T>::circa(const Matrix<T>& rhs, const T& tolerance) {
    // test for approximate equality
    if (this->rows != rhs.get_rows() || this->cols != rhs.get_cols())
        return false;

    for (int i=0; i < rows; ++i)
    for (int j=0; j < cols; ++j)
        if (fabs(this->mat[i][j] - rhs.mat[i][j]) > tolerance)
            return false;
    return true;
}

template<typename T>
vector<T> Matrix<T>::diag_vec() {
    // Obtain a vector of the diagonal elements
    vector<T> result(rows, 0.0);

    for (unsigned i=0; i<rows; ++i)
        result[i] = this->mat[i][i];

    return result;
}

template<typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) {
    return this->mat[row][col];
}

template<typename T>
const T& Matrix<T>::operator()(const unsigned& row, const unsigned& col) const {
    return this->mat[row][col];
}

template<typename T>
vector<T>& Matrix<T>::operator[] (const unsigned& elem) {
    return this->mat[elem];
}

template<typename T>
const vector<T>& Matrix<T>::operator[] (const unsigned& elem) const {
    return this->mat[elem];
}


template<typename T>
unsigned Matrix<T>::get_rows() const {
    return this->rows;
}

template<typename T>
unsigned Matrix<T>::get_cols() const {
    return this->cols;
}

template<typename U>
ostream& operator<<(ostream &os, Matrix<U> rhs){
    int row = rhs.get_rows();
    int col = rhs.get_cols();

    for (int i = 0; i<row; ++i) {
        for (int j = 0; j<col; ++j)
            os << /*setw(9) <<*/ std::setprecision(3) << rhs.mat[i][j] << " ";
        std::cout << '\n';
    }
    return os;
}

#endif