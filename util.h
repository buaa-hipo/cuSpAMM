#pragma once
#pragma once
// #include <mkl.h>
// #include <tbb/parallel_for.h>
#include <functional>
// #include "igl_timer.h"
// #include <Eigen/Sparse>
#include <set>
#include <vector>
#include <tuple>
#include <iostream>

inline double benchmarkTimer(std::function<void()> op) {
	igl::Timer t;
	t.start();
	op();
	t.stop();
	return t.getElapsedTimeInMicroSec() / 100.0;
};

// generate a sparse matrix with fixed number of entries per row
template <typename T, int _Options = Eigen::RowMajor>
Eigen::SparseMatrix<T>
generate_sparse_matrix(int m, int n, int entry_per_row)
{
	Eigen::SparseMatrix<T, _Options> A(m, n);
	std::vector<Eigen::Triplet<double>> trip;
	trip.reserve(m * entry_per_row);
	for (int i = 0; i < m; ++i) {
		std::set<int> col_nums;
		while (col_nums.size() < entry_per_row) {
			col_nums.insert(int(rand() % n));
		}

		for (auto c : col_nums) {
			trip.push_back(Eigen::Triplet<double>(i, c, rand() / ((T)RAND_MAX)));
		}

	}
	A.setFromTriplets(trip.begin(), trip.end());
	A.makeCompressed();
	return A;
};

// generate a sparse matrix where the entry per row is not fixed, but the average entry per row is set by user
template <typename T, int _Options = Eigen::RowMajor>
Eigen::SparseMatrix<T>
generate_sparse_matrix_average(int m, int n, int entry_per_row)
{
	Eigen::SparseMatrix<T, _Options> A(m, n);
	std::set<std::pair<int, int>> positions;
	std::vector<Eigen::Triplet<double>> trip;
	trip.reserve(m * entry_per_row);
	while (positions.size() < entry_per_row * m) {
		positions.insert({ int(rand() % m), int(rand() % n) });
	}
	for (auto p : positions) {
		trip.push_back(Eigen::Triplet<double>(p.first, p.second, rand() / ((T)RAND_MAX)));
	}
	A.setFromTriplets(trip.begin(), trip.end());
	A.makeCompressed();
	return A;
};


// creates a mkl csr matrix from eigen sparse matrix
void create_mkl_csr_matrix(Eigen::SparseMatrix<double, Eigen::RowMajor>& M, sparse_matrix_t* A) {
	sparse_status_t status;
	status = mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, M.rows(), M.cols(), M.outerIndexPtr(), M.outerIndexPtr() + 1, M.innerIndexPtr(), M.valuePtr());
};


// computes AtA in mkl using multi thread
void PROFILE_MKL_MULTI_SYRK(std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>>& MATRIX_VECTOR, sparse_matrix_t& R) {
	sparse_status_t status;
	std::vector<sparse_matrix_t> matrix_vectors, result_vectors;
	struct matrix_descr symmetric_type;
	auto elapsed_prep = benchmarkTimer([&] {
		symmetric_type.type = SPARSE_MATRIX_TYPE_DIAGONAL;
		symmetric_type.diag = SPARSE_DIAG_UNIT;
		for (unsigned int i = 0; i < MATRIX_VECTOR.size(); i++) {
			sparse_matrix_t tmp;
			MKL_INT nnz = MATRIX_VECTOR[i].nonZeros();
			create_mkl_csr_matrix(MATRIX_VECTOR[i], &tmp);
			matrix_vectors.push_back(tmp); \
		}
		sparse_matrix_t tmp;
		matrix_vectors.push_back(tmp);
		for (int i = 0; i < 1; i++) {
			sparse_matrix_t tmp;
			result_vectors.push_back(tmp);
		}
		status = mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, matrix_vectors[0], matrix_vectors[1], symmetric_type, &result_vectors[0], SPARSE_STAGE_NNZ_COUNT);
		status = mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, matrix_vectors[0], matrix_vectors[1], symmetric_type, &result_vectors[0], SPARSE_STAGE_FINALIZE_MULT_NO_VAL);
		});
	std::cout << "MKL PREP TIME " << elapsed_prep * 100 / 1000000 << " sec\n";
	mkl_set_num_threads_local(0);
	auto elapsed = benchmarkTimer([&]() {
		for (int j = 0; j < 100; j++) {
			// std::cout << "loop: " << j << "\n";
			status = mkl_sparse_sypr(SPARSE_OPERATION_TRANSPOSE, matrix_vectors[0], matrix_vectors[1], symmetric_type, &result_vectors[0], SPARSE_STAGE_FINALIZE_MULT);
			// std::cout << "status: " << status << "\n";
		}
		});
	std::cout << "MKL_MULTI SYRK: " << elapsed << " us\n";
	R = result_vectors[0];
	mkl_free_buffers();
};


sparse_status_t mkl_export_csr(const sparse_matrix_t A,
	sparse_index_base_t* indexing, MKL_INT* num_rows, MKL_INT* num_cols,
	MKL_INT** row_start, MKL_INT** row_end, MKL_INT** col_indx,
	double** values) {
	return mkl_sparse_d_export_csr(A, indexing, num_rows, num_cols,
		row_start, row_end,
		col_indx, values);
};



// extract info from mkl_sparse_t
static std::tuple<int, int, int, std::vector<int>, std::vector<int>, std::vector<double>> extract_value(sparse_matrix_t& mkl_matrix)
{
	sparse_status_t status;
	sparse_index_base_t indexing;
	MKL_INT num_rows, num_cols;
	MKL_INT* row_start, * row_end, * col_indx;
	double* values;
	status = mkl_export_csr(mkl_matrix, &indexing, &num_rows, &num_cols,
		&row_start, &row_end,
		&col_indx, &values);
	int count = 0;
	std::vector<int> outerindex;
	std::vector<int> innerindex;
	std::vector<double> valueptr;
	outerindex.reserve(num_rows + 1);
	innerindex.reserve(num_rows);
	valueptr.reserve(num_rows);
	for (int i = 0; i < num_rows; ++i) {
		outerindex.push_back(row_start[i]);
		std::vector<std::pair<int, double>> sorted_index;
		for (int j = row_start[i]; j < row_end[i]; ++j) {
			sorted_index.push_back({ col_indx[j], values[count] });
			++count;
		}
		sort(sorted_index.begin(), sorted_index.end());
		for (auto p : sorted_index) {
			innerindex.push_back(std::get<0>(p));
			valueptr.push_back(std::get<1>(p));
		}
	}
	outerindex.push_back(count);
	return { num_rows, num_cols, count, outerindex, innerindex, valueptr };
}


// construct a map sparse matrix
static Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> ConstructSparseMatrix(int rowCount, int colCount, int nonZeroCount, double* nonZeroArray, int* rowIndex, int* colIndex)
{
	Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> spMap(rowCount, colCount, nonZeroCount, rowIndex, colIndex, nonZeroArray, 0);
	return spMap;
}