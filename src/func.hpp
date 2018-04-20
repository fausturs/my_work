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
std::vector< std::pair<std::string, T> > all_evaluations(const D& d,const wjy::predictor<T, V >& predictor)
{
    std::vector< std::pair<std::string, T> > ans;
    ans.emplace_back("MSE", mean_square_error(d, predictor));
    ans.emplace_back("MAPE", mean_absolute_percentage_error(d, predictor));
    ans.emplace_back("R-Square", r_square(d, predictor));
    return ans;
}

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

//for test
void test();


#endif
