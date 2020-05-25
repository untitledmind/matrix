#ifndef MATRIX_H
#define MATRIX_H

/* need C++11 or higher to compile */

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <initializer_list>

template <typename Type> 
class Matrix {
    private:
        std::vector<std::vector<Type> > mat;
        unsigned rows;
        unsigned cols;
        Type d;	//determinant

    public:
        Matrix() {}
        Matrix(unsigned _rows = 0, unsigned _cols = 0, const Type _initial = 0);
        Matrix(const Matrix<Type>& rhs);
        Matrix(std::initializer_list<std::initializer_list<Type> >, unsigned _rows = 0, unsigned _cols = 0, const Type _initial = 0);
        virtual ~Matrix();

        Matrix<Type>& operator=(const Matrix<Type>& rhs);
        Matrix<Type> operator+(const Matrix<Type>& rhs);
        Matrix<Type>& operator+=(const Matrix<Type>& rhs);
        Matrix<Type> operator-(const Matrix<Type>& rhs);
        Matrix<Type>& operator-=(const Matrix<Type>& rhs);
        Matrix<Type> operator*(const Matrix<Type>& rhs);
        Matrix<Type>& operator*=(const Matrix<Type>& rhs);
        bool operator== (const Matrix<Type>& rhs);
        bool circa(const Matrix<Type>& rhs, const Type& tolerance);

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

        // Matrix/scalar operations
        Matrix<Type> operator+ (const Type& rhs);
        Matrix<Type> operator- (const Type& rhs);
        Matrix<Type> operator* (const Type& rhs);
        Matrix<Type> operator/ (const Type& rhs);

        // Matrix/vector operations
        std::vector<Type> operator* (const std::vector<Type>& rhs);
        std::vector<Type> diag_vec();

        // Access the individual elements
        Type& operator() (const unsigned& row, const unsigned& col);
        const Type& operator() (const unsigned& row, const unsigned& col) const;
        std::vector<Type>& operator[] (const unsigned& el);
        const std::vector<Type>& operator[] (const unsigned& el) const;

        void set_value(int i, int j, const Type& value);

        unsigned get_rows() const;
        unsigned get_cols() const;

    template<typename U>
    friend std::ostream& operator<<(std::ostream &os, Matrix<U> rhs);
};

#include "matrix.cpp"

#endif
