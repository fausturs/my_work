#include <iostream>
#include <fstream>
#include <utility>

#include "MF.hpp"
#include "TF.hpp"
#include "MySQL.hpp"

void print_tensor(TF::sparse_tensor_tp & A, size_t dim_1, size_t dim_2, size_t dim_3)
{
    for (int i=0; i<dim_1; i++)
    {
        bool flag1 = A.count(i);
        for (int j=0; j<dim_2; j++)
        {
            bool flag2 = flag1 && A[i].count(j);
            for (int k=0; k<dim_3; k++)
            {
                bool flag3 = flag2 && A[i][j].count(k);
                if (flag3)
                    std::cout<< A[i][j][k] <<" ";
                else
                    std::cout<< 0 <<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}


int main(int args, const char* argv[])
{
	/*
    std::ifstream myin("a.txt");
    size_t dim_1, dim_2, dim_3;
    int temp;
    myin>>dim_1>>dim_2>>dim_3;
    TF::sparse_tensor_tp A;
    for (int i=0; i<dim_1; i++)
        for (int j=0; j<dim_2; j++)
            for (int k=0; k<dim_3; k++)
            {
                myin>>temp;
                if (temp!=TF::empty_mark) A[i][j][k]=temp;
            }
    //
    TF tf;
    tf.initialize(2, 2, 2);
    tf.train(A, std::cout);
	*/
	MySQL sql("127.0.0.1", "wjy", "", "lagou_data");
	sql.connect();
	sql.query("select id,position from clean_positions limit 0,20;");
	std::cout<<sql.get_error_info();
	auto & table = sql.get_result_table();
	for (auto & row : table)
	{
		for (auto & x : row) std::wcout<< string_to_wstring(x);
		std::cout<<"\n";
	}
	std::cout<<sql()<<std::endl;    
    std::cout<<"hello world!"<<std::endl;
    return 0;
}
