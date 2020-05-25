#ifndef MATRIX_HPP
#define MATRIX_HPP

/* need C++11 or higher to compile */

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <initializer_list>
//aflksjdf;laksjdfl;js
template <typename Type> 
class Matrix {
    private:
        std::vector<std::vector<Type> > mat;
        unsigned rows;
        unsigned cols;
        Type det;

    public:
        Matrix() {}
        Matrix(unsigned _rows = 0, unsigned _cols = 0, const Type _initial = 0);
        Matrix(const Matrix<Type>& rhs);
        Matrix(std::initializer_list<std::initializer_list<Type> >, unsigned _rows = 0, unsigned _cols = 0, const Type _initial = 0);
        virtual ~Matrix();

        // Access the individual elements
        Type& operator() (const unsigned& row, const unsigned& col);
        const Type& operator() (const unsigned& row, const unsigned& col) const;
        std::vector<Type>& operator[] (const unsigned& el);
        const std::vector<Type>& operator[] (const unsigned& el) const;

        Matrix<Type> operator+ (const Matrix<Type>& rhs);
        Matrix<Type> operator- (const Matrix<Type>& rhs);
        Matrix<Type> operator* (const Matrix<Type>& rhs);
        Matrix<Type>& operator= (const Matrix<Type>& rhs);
        Matrix<Type>& operator+= (const Matrix<Type>& rhs);
        Matrix<Type>& operator-= (const Matrix<Type>& rhs);
        Matrix<Type>& operator*= (const Matrix<Type>& rhs);
        bool operator== (const Matrix<Type>& rhs);
        bool circa(const Matrix<Type>& rhs, const Type& tolerance);

        // Matrix/scalar operations
        Matrix<Type> operator+ (const Type& rhs);
        Matrix<Type> operator- (const Type& rhs);
        Matrix<Type> operator* (const Type& rhs);
        Matrix<Type> operator/ (const Type& rhs);

        // Matrix/vector operations
        std::vector<Type> operator* (const std::vector<Type>& rhs);
        std::vector<Type> diag_vec();

        Matrix<Type> transpose();
        void swapRows(int i, int j);
        void Permute(int col);
        void Pivot(int row, int col);
        Matrix<Type> PermutationMatrix(int Size, int row1, int row2);
        int maxCol(int col);
        Matrix<Type> Inverse();
        Matrix<Type> block_multiply(const Matrix<Type>& rhs);
        Type Determinant();
        Type determinant(Matrix<Type>& rhs, int row, int col);
        std::vector<Type> gauss_jordan(const char& pivoting);
        std::vector<Type> jacobi(const Type& tolerance, const int& max_iter = 100);
        std::vector<Type> gauss_seidel(const Type& tolerance, const int& max_iter = 100);

        void set_value(int i, int j, const Type& value);
        unsigned get_rows() const;
        unsigned get_cols() const;

    template<typename U>
    friend std::ostream& operator<<(std::ostream &os, Matrix<U> rhs);
};

template<typename Type>
Matrix<Type>::Matrix(unsigned rows_, unsigned cols_, const Type initial_) {
    mat.resize(rows_);
    for (unsigned i=0; i<mat.size(); ++i) {
        mat[i].resize(cols_, initial_);
        rows = rows_;
        cols = cols_;
    }
}

template<typename Type>
Matrix<Type>::Matrix(const Matrix<Type>& rhs) {
    mat = rhs.mat;
    rows = rhs.get_rows();
    cols = rhs.get_cols();
}

template<typename Type>
Matrix<Type>::Matrix(std::initializer_list<std::initializer_list<Type> > lst, unsigned rows_,
    unsigned cols_, const Type initial_): Matrix(lst.size(), lst.size()+1, initial_) {
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

template<typename Type>
Matrix<Type>::~Matrix() {}

template<typename Type>
Matrix<Type>& Matrix<Type>::operator= (const Matrix<Type>& rhs) {
    if (this != &rhs) {
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
    
template<typename Type>
Matrix<Type> Matrix<Type>::operator+ (const Matrix<Type>& rhs) {
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

template<typename Type>
Matrix<Type>& Matrix<Type>::operator+= (const Matrix<Type>& rhs) {
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

template<typename Type>
Matrix<Type> Matrix<Type>::operator- (const Matrix<Type>& rhs) {
    if (rows != rhs.rows & cols != rhs.cols) {
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }

    unsigned rows = rhs.get_rows();
    unsigned cols = rhs.get_cols();
    Matrix result(rows, cols, static_cast<Type>(0));

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] - rhs(i,j);

    return result;
}

template<typename Type>
Matrix<Type>& Matrix<Type>::operator-= (const Matrix<Type>& rhs) {
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

template<typename Type>
Matrix<Type> Matrix<Type>::operator* (const Matrix<Type>& rhs) {
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

template<typename Type>
Matrix<Type>& Matrix<Type>::operator*= (const Matrix<Type>& rhs) {
    if (rows != rhs.rows || cols != rhs.cols) {
        std::cout << "matrices are not same size" << std::endl;
        exit(1);
    }

    Matrix result = (*this) * rhs;
    (*this) = result;

    return *this;
}

template<typename Type>
Matrix<Type> Matrix<Type>::transpose() {
    Matrix result(cols, rows, static_cast<Type>(0));

    for (unsigned i=0; i<cols; ++i)
    for (unsigned j=0; j<rows; ++j)
        result(i,j) = this->mat[j][i];

    return result;
}

template<typename Type>
void Matrix<Type>::swapRows(int i, int j) {
    std::swap(mat[i],mat[j]);
}

template<typename Type>
void Matrix<Type>::Pivot(int row, int col) {
    std::cout << "Entering Pivot function " << "Row = " << row << " Col = " << col << std::endl;
    Type max(0);
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



template<typename Type>
void Matrix<Type>::Permute(int col){
    std::cout << "Entering Permuted function - column col = " << col << std::endl;
    Type max(0);
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

template<typename Type>
Matrix<Type> Matrix<Type>::PermutationMatrix(int Size, int row1, int row2) {
    Matrix P(Size, Size, Type(0));

    for (int i = 0; i < Size; ++i)
        P.set_value(i,i,Type(1));
    std::cout << "finished setting diagonal elements"<< std::endl;
    std::swap(P.mat[row1], P.mat[row2]);

    return P;
}

template<typename Type>
Matrix<Type> Matrix<Type>::block_multiply(const Matrix<Type>& rhs) {
    if (this->cols != rhs.rows)
        throw(std::runtime_error("operation doesn't support these dimensions\n"));

    Matrix<Type> temp(this->rows, rhs.cols, 0.0);

    for (int i=0; i < this->rows; ++i)
    for (int j=0; j < rhs.cols; ++j)
    for (int k=0; k < this->cols; ++k)
        temp.mat[i][j] += this->mat[i][k] * rhs.mat[k][j];

    return temp;
}

template<typename Type>
Matrix<Type> Matrix<Type>::Inverse() {
    std::cout << "Entering Inverse " << std::endl;
    Matrix<Type> temp(*this);

    Matrix<Type> I(rows, cols, Type(0));
    Type a = Type(0.0);
    Type ratio = Type(0);

    for(int i = 0; i < cols; ++i)
        I.mat[i][i] = Type(1);

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


template<typename Type>
Type Matrix<Type>::Determinant()
{
    Type dtr(0);
    Matrix<Type> submat(rows, cols, static_cast<Type>(0));

    for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
        submat.mat[i][j] = this->mat[i][j];

    dtr = determinant(submat, rows, cols);

    return dtr;
}


template<typename Type>
Type Matrix<Type>::determinant(Matrix<Type>& submat,int row, int col) {
    Matrix<Type> Submat(rows,cols, static_cast<Type>(0));
    Type det(0);

    if (row == 2)
        det = submat.mat[0][0] * submat.mat[1][1] - submat.mat[1][0] * submat.mat[0][1];
    else
        for (int x = 0; x < row; ++x) {
            Type subi = Type(0); 
            for (int i = 1; i < row; ++i) {
                Type subj = Type(0);
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

template<typename Type>
std::vector<Type> Matrix<Type>::gauss_jordan(const char& pivoting) {
    if (cols != rows+1)
        throw(std::runtime_error("*this must be augmented square matrix to call this method"));

    int i;
    int j;
    int k;
    Type scalar;
    Matrix<Type> M = *this;

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
                    Type temp = M.mat[k][k];
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
                    Type temp = M.mat[k][k];
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
    std::vector<Type> soln;
    for (i = 0; i < rows; ++i) {
        std::cout << "X" << i + 1 << " = " << M.mat[i][rows] << '\n';
        soln.push_back(M.mat[i][rows]);
    }     
    return soln;
}

template<typename Type>
std::vector<Type> Matrix<Type>::jacobi(const Type& tolerance, const int& max_iter) {
    Matrix<Type> M = *this;
    std::vector<Type> x;    // solution vector
    std::vector<Type> x_old;
    std::vector<Type> b;    // augmented coefficients
    // these variables abide by Mx = b

    int iter = 0;

    for (int i=0; i < rows; ++i) {
        x.push_back(Type(0));
        x_old.push_back(x[i]);
        b.push_back(M[i][cols-1]);
    }

    do {
        for (int i = 0; i < rows; ++i) {
            x_old[i] = x[i];

            Type bi = b[i];
            Type ci = M[i][i];

            for (int j = 0; j < rows; ++j)
                if (j != i)
                    bi -= M[i][j]*x_old[j];

            x[i] = bi/ci;        
        }

        Type diff = Type(0);
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

template<typename Type>
std::vector<Type> Matrix<Type>::gauss_seidel(const Type& tolerance, const int& max_iter) {
    Matrix<Type> M = *this;
    std::vector<Type> x;
    std::vector<Type> x_old;
    std::vector<Type> b;
    int iter = 0;

    for (int i=0; i < rows; ++i) {
        x.push_back(Type(0));
        x_old.push_back(Type(0));
        b.push_back(M[i][cols-1]);
    }

    do {
        for (int i=0; i < rows; ++i)
            x_old[i] = x[i];

        for (int i=0; i < rows; ++i) {
            Type bi = b[i];
            Type ci = M[i][i];

            for (int j=0; j < i; ++j)
                bi -= M[i][j] * x[j];
            
            for (int j = i+1; j < rows; ++j)
                bi -= M[i][j] * x_old[j];

            x[i] = bi/ci;
        }

        Type diff = Type(0);

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

template<typename Type>
Matrix<Type> Matrix<Type>::operator+ (const Type& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] + rhs;

    return result;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator- (const Type& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] - rhs;

    return result;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator* (const Type& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat;

    return result;
}

template<typename Type>
Matrix<Type> Matrix<Type>::operator/ (const Type& rhs) {
    Matrix result(rows, cols, 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result(i,j) = this->mat[i][j] / rhs;

    return result;
}

template<typename Type>
std::vector<Type> Matrix<Type>::operator* (const std::vector<Type>& rhs) {
    std::vector<Type> result(rhs.size(), 0.0);

    for (unsigned i=0; i<rows; ++i)
    for (unsigned j=0; j<cols; ++j)
        result[i] = this->mat[i][j] * rhs[j];

    return result;
}

template<typename Type>
bool Matrix<Type>::operator== (const Matrix<Type>& rhs) {
    if (this->rows != rhs.get_rows() || this->cols != rhs.get_cols())
        return false;

    for (int i=0; i < rows; ++i)
    for (int j=0; j < cols; ++j)
        if (this->mat[i][j] != rhs.mat[i][j]) 
            return false;
    return true;
}

template<typename Type>
bool Matrix<Type>::circa(const Matrix<Type>& rhs, const Type& tolerance) {
    if (this->rows != rhs.get_rows() || this->cols != rhs.get_cols())
        return false;

    for (int i=0; i < rows; ++i)
    for (int j=0; j < cols; ++j)
        if (fabs(this->mat[i][j] - rhs.mat[i][j]) > tolerance)
            return false;
    return true;
}

template<typename Type>
std::vector<Type> Matrix<Type>::diag_vec() {
    std::vector<Type> result(rows, 0.0);

    for (unsigned i=0; i<rows; ++i)
        result[i] = this->mat[i][i];

    return result;
}

template<typename Type>
Type& Matrix<Type>::operator() (const unsigned& row, const unsigned& col) {
    return this->mat[row][col];
}

template<typename Type>
const Type& Matrix<Type>::operator() (const unsigned& row, const unsigned& col) const {
    return this->mat[row][col];
}

template<typename Type>
std::vector<Type>& Matrix<Type>::operator[] (const unsigned& elem) {
    return this->mat[elem];
}

template<typename Type>
const std::vector<Type>& Matrix<Type>::operator[] (const unsigned& elem) const {
    return this->mat[elem];
}

template<typename Type>
void Matrix<Type>::set_value(int i, int j, const Type& value) {
    mat[i][j] = value;
}

template<typename Type>
unsigned Matrix<Type>::get_rows() const {
    return this->rows;
}

template<typename Type>
unsigned Matrix<Type>::get_cols() const {
    return this->cols;
}

template<typename U>
std::ostream& operator<<(std::ostream& os, Matrix<U> rhs){
    int row = rhs.get_rows();
    int col = rhs.get_cols();

    for (int i = 0; i<row; ++i) {
        for (int j = 0; j<col; ++j)
            os << std::setprecision(3) << rhs.mat[i][j] << " ";
        std::cout << '\n';
    }
    return os;
}

#endif