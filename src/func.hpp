#ifndef FUNC_HPP
#define FUNC_HPP

#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <numeric>
#include <algorithm>
#include <functional>


#include "Date.hpp"
#include "tools.hpp"
#include "predictor.hpp"
#include "sparse_tensor.hpp"

extern std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
extern std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;

void read_all(wjy::Date data_date);
void create_tensor();





//          evaluation functions
template <typename D, typename T, typename V>
T mean_square_error(const D& d,const wjy::predictor<T, V >& predictor)
{
    T MSE = 0;
    for (auto & entry : d)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        T temp = (value - predictor( index ));
        MSE += temp*temp;
    }
    return MSE / d.size();
}

template <typename D, typename T, typename V>
T mean_absolute_percentage_error(const D& d, const wjy::predictor<T, V >& predictor)
{
    T MAPE = 0;
    for (auto & entry : d)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        T temp = std::abs( (value - predictor( index ))/value );
        MAPE += temp;
    }
    return MAPE * 100 / d.size();
}

template <typename D, typename T, typename V>
T r_square(const D& d, const wjy::predictor<T, V>& predictor)
{
    T rs = 0, mean = 0, ss_tot = 0, ss_res = 0;
    for (auto & entry : d)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        mean += value;
        T temp = (value - predictor( index ));
        ss_res += temp*temp;
    }
    mean /= d.size();
    for (auto & entry : d) ss_tot += (entry.second - mean)*(entry.second - mean);
    return 1-(ss_res/ss_tot);
}

template <typename D, typename T, typename V>
std::vector< std::pair<std::string, T> > all_evaluations(const D& d,const wjy::predictor<T, V >& predictor)
{
    std::vector< std::pair<std::string, T> > ans;
    ans.emplace_back("MSE", mean_square_error(d, predictor));
    ans.emplace_back("MAPE", mean_absolute_percentage_error(d, predictor));
    ans.emplace_back("R-Square", r_square(d, predictor));
    return ans;
}

//  generate negative entries
template<typename T>
wjy::sparse_tensor<T, 3> generate_negative_entries(const wjy::sparse_tensor<T, 3>& t, int n = 5)
{
    wjy::sparse_tensor<T, 3> negative_tensor;
    auto dims = wjy::dims_of_sparse_tensor(t);
    auto t1 = wjy::flatten_sparse_tensor(t, 0);
    auto skill_popularity = wjy::flatten_sparse_tensor(t1, 0);
    auto position_popularity = wjy::flatten_sparse_tensor(t1, 1);
    T sum_s = 0, sum_p = 0;
    for (auto & pop : skill_popularity) sum_s += pop.second;
    for (auto & pop : position_popularity) sum_p += pop.second;
    std::mt19937 mt(1);
    std::uniform_int_distribution<size_t> uid(0, dims[0]-1);
    std::uniform_real_distribution<double> urd(0, 1);

    for (size_t position=0; position<dims[1]; position++)
    {
        T pop = position_popularity[{position}];
        double p_p = 1 / ( 1 + std::exp(-pop/sum_p*5000) );
        for (size_t skill=0; skill<dims[2]; skill++)
        {
            if (t1.count({position, skill})!=0) continue;
            pop = skill_popularity[{skill}];
            double p_s = 1 / ( 1 + std::exp(-pop/sum_s*5000) );
            double p = (p_s-0.5) * (p_p-0.5);
            if (urd(mt)>p) continue;
            wjy::sparse_tensor_index<3> index;
            for (int i=0; i<n; i++)
            {
                index[0]=uid(mt); index[1]=position; index[2]=skill;
                negative_tensor[index] = 0;
            }
        }
    }
    return negative_tensor;
}

//for test
void test();


#endif
