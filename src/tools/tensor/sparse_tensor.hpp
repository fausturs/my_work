#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP

#include <vector>
#include <array>
#include <unordered_map>

namespace wjy {
    
    template<size_t dim>
    using sparse_tensor_index = std::array<size_t, dim>;

    // hash function
    template<size_t dim>
    class sparse_tensor_index_hash{
        std::hash<std::string> string_hash;
    public:
        size_t operator()(const sparse_tensor_index<dim>& indexes) const
        {
            std::string st_index;
            for (auto i : indexes) st_index += std::to_string(i)+",";
            return string_hash(st_index);
        }
    };
    
    template <class T, size_t dim>
    using sparse_tensor = std::unordered_map< sparse_tensor_index<dim> , T, sparse_tensor_index_hash<dim> >;

}

#endif
