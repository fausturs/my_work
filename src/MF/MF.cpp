#include "MF.hpp"

#include <iostream>
#include <iterator>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cassert>
/*
    some local function
 */


/*
    member value or member function of class MF
 */
std::ostringstream MF::MF_log;
const MF::element_tp MF::empty_mark = 0;

void MF::initialize(size_t k, int rand_seed, element_tp rand_range, element_tp lambda, element_tp initial_learning_rate, size_t max_iter_num, element_tp epsilon)
{
    this->k = k;
    this->rand_range = rand_range;
    this->lambda = lambda;
    this->initial_learning_rate = initial_learning_rate;
    this->max_iter_num = max_iter_num;
    this->epsilon = epsilon;
    
    if (rand_seed == -1)
        mt = std::mt19937( std::random_device{}() );
    else
        mt = std::mt19937( rand_seed );
    
    clear();
}

MF::element_tp MF::predict(size_t i, size_t j) const
{
    auto begin_it_u = u.begin() + i*k;
    auto end_it_u   = begin_it_u + k;
    auto begin_it_v = v.begin() + j*k;
    auto predict_value = std::inner_product(begin_it_u, end_it_u,
                                            begin_it_v,
                                            static_cast<element_tp>(0)
                                            );
    return predict_value;
}

void MF::save(const std::string& path) const
{
    std::ofstream myout(path);
    assert(myout);
    
    myout<<rows_num<<" "<<cols_num<<" "<<k<<" ";
    std::copy(u.begin(), u.end(), std::ostream_iterator<element_tp>(myout, " "));
    std::copy(v.begin(), v.end(), std::ostream_iterator<element_tp>(myout, " "));
}

void MF::load(const std::string& path)
{
    std::ifstream myin(path);
    assert(myin);
    
    clear();
    myin>>rows_num>>cols_num>>k;
    u.reserve(rows_num*k);
    v.reserve(cols_num*k);
    std::copy_n(std::istream_iterator<element_tp>(myin), rows_num*k, std::back_inserter(u));
    std::copy_n(std::istream_iterator<element_tp>(myin), cols_num*k, std::back_inserter(v));
}

void MF::clear()
{
    u.clear();
    v.clear();
}

void MF::print()
{
    std::cout<<"Matrix u:"<<std::endl;
    std::copy(u.begin(), u.end(), std::ostream_iterator<element_tp>(std::cout, "\t"));
    std::cout<<std::endl;
    std::cout<<"Matrix v:"<<std::endl;
    std::copy(v.begin(), v.end(), std::ostream_iterator<element_tp>(std::cout, "\t"));
    std::cout<<std::endl;
}


void MF::generate_u_v(const dense_matrix_tp& A)
{
    // calculate rows_num and cols_num with different type
    rows_num = A.size();
    cols_num = A.begin()->size();
    //
    clear();
    u.reserve(rows_num*k);
    v.reserve(cols_num*k);
    // gen_rand() generate a real number uniformly from [-rand_range, rand_range],
    // and use mt as the random device
    std::uniform_real_distribution<element_tp> urd(-rand_range, rand_range);
    auto gen_rand = std::bind(urd, std::ref(mt));
    
    std::generate_n(std::back_inserter(u), rows_num*k, gen_rand);
    std::generate_n(std::back_inserter(v), cols_num*k, gen_rand);
}

void MF::generate_u_v(const sparse_matrix_tp& A)
{
    // calculate rows_num and cols_num
    rows_num = cols_num = 0;
    for (const auto& row : A)
    {
        rows_num = std::max(rows_num, row.first+1);
        for (const auto& ele : row.second) cols_num = std::max(cols_num, ele.first+1);
    }
    //
    clear();
    u.reserve(rows_num*k);
    v.reserve(cols_num*k);
    // gen_rand() generate a real number uniformly from [-rand_range, rand_range],
    // and use mt as the random device
    std::uniform_real_distribution<element_tp> urd(-rand_range, rand_range);
    auto gen_rand = std::bind(urd, std::ref(mt));
    
    std::generate_n(std::back_inserter(u), rows_num*k, gen_rand);
    std::generate_n(std::back_inserter(v), cols_num*k, gen_rand);
}

MF::element_tp MF::get_Aij(const MF::dense_matrix_tp& A, size_t i, size_t j) const
{
    return A[i][j];
}
MF::element_tp MF::get_Aij(const MF::sparse_matrix_tp& A, size_t i, size_t j) const
{
    if (A.find(i)==A.end()) return MF::empty_mark;
    auto& row_i = A.at(i);
    if (row_i.find(j)==row_i.end()) return MF::empty_mark;
    return row_i.at(j);
}
