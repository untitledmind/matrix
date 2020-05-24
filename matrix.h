#ifndef MATRIX_H
#define MATRIX_H

/* need C++11 or higher to compile */

#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
#include <initializer_list>
using namespace std;

template <typename T> 
class Matrix {
 private:
 vector<vector<T> > mat;
  unsigned rows;
  unsigned cols;
  T d;	//determinant

 public:
    // Matrix() {}
  Matrix(unsigned _rows = 0, unsigned _cols = 0, const T _initial = 0);
//  Matrix(unsigned _rows, unsigned _cols, const T* _initial);
  Matrix(const Matrix<T>& rhs);
  Matrix(initializer_list<initializer_list<T> >, unsigned _rows = 0, unsigned _cols = 0, const T _initial = 0);
  virtual ~Matrix();

  // Operator overloading, for "standard" mathematical matrix operations
  Matrix<T>& operator=(const Matrix<T>& rhs);

  // Matrix mathematical operations
  Matrix<T> operator+(const Matrix<T>& rhs);
  Matrix<T>& operator+=(const Matrix<T>& rhs);
  Matrix<T> operator-(const Matrix<T>& rhs);
  Matrix<T>& operator-=(const Matrix<T>& rhs);
  Matrix<T> operator*(const Matrix<T>& rhs);
  Matrix<T>& operator*=(const Matrix<T>& rhs);
  bool operator== (const Matrix<T>& rhs);
  bool circa(const Matrix<T>& rhs, const T& tolerance);

//Auxiliary Matrix functions
  Matrix<T> transpose();
  void swapRows(int i, int j);
  void Permute(int col);						//permute rows of matrix in descending order
  void Pivot(int row, int col);						//create next privot element
  Matrix<T> PermutationMatrix(int Size, int row1, int row2);		//creates a permutation matrix
  int maxCol(int col);
  Matrix<T> Inverse();
  Matrix<T> block_multiply(const Matrix<T>& rhs);
  T Determinant();			//compute the determinant of this matrix
  T determinant(Matrix<T>& rhs, int row, int col);
  vector<T> gauss_jordan(const char& pivoting);
  vector<T> jacobi(const T& tolerance, const int& max_iter = 100);
  vector<T> gauss_seidel(const T& tolerance, const int& max_iter = 100);

  // Matrix/scalar operations
  Matrix<T> operator+(const T& rhs);
  Matrix<T> operator-(const T& rhs);
  Matrix<T> operator*(const T& rhs);
  Matrix<T> operator/(const T& rhs);

  // Matrix/vector operations
  std::vector<T> operator*(const std::vector<T>& rhs);
  std::vector<T> diag_vec();

  // Access the individual elements
  T& operator()(const unsigned& row, const unsigned& col);
  const T& operator()(const unsigned& row, const unsigned& col) const;
  vector<T>& operator[](const unsigned& el);
  const vector<T>& operator[](const unsigned& el) const;

  //set the individual elements
  void set_value(int i, int j, const T& value){mat[i][j] = value;}

  // Access the row and column sizes
  unsigned get_rows() const;
  unsigned get_cols() const;

  template<typename U>
  friend ostream& operator<<(ostream &os, Matrix<U> rhs);
};

#include "matrix.cpp"

#endif
