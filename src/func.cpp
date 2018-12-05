#include "func.hpp"

#include <iostream>
#include <fstream>
#include <locale>
#include <codecvt>
#include <string>
#include <list>
#include <cassert>

#include "sparse_tensor.hpp"


std::unordered_map< std::string, size_t >   skill_to_id, company_to_id, position_to_id;
std::vector< std::string >                  id_to_skill, id_to_company, id_to_position;

std::unordered_map< std::string, size_t >   demand_level;
std::unordered_map< int, std::pair<std::string, std::string> >  jdid_company_position;
std::unordered_map< int, int > jdid_posttime;

std::unordered_map< std::string, int > skill_counter, company_counter, position_counter;
//

std::pair< std::string, std::string > get_company_position(int id);
std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd, const std::unordered_map< std::string, size_t >& skill_list, const std::unordered_map< std::string, size_t >& demand_level);

void read_raw_data();
void prepare_filter();
bool skill_fliter(const std::string& skill);
bool company_fliter(const std::string& company);
bool position_fliter(const std::string& position);
void save_new_data(const wjy::sparse_tensor<double, 4> t);

void create_tensor()
{
    read_raw_data();
    std::clog<<"read raw data finish!"<<std::endl;
    prepare_filter();
    std::clog<<"prepare filter finish!"<<std::endl<<"create tensor now"<<std::endl;
    
    std::ifstream myin("../data/raw_data/position_cut.txt");
    assert(myin);
    wjy::sparse_tensor<double, 4> tensor;
    wjy::sparse_tensor_index<4> index = {0, 0, 0, 0};
    std::string st_jd;
    while (std::getline(myin, st_jd))
    {
        auto job_description = wjy::split(st_jd, {',','/'});
        int id = std::atoi( job_description[0].c_str() );
        std::pair<std::string, std::string> company_position;
		if (jdid_company_position.find(id) == jdid_company_position.end()) continue;
		company_position = jdid_company_position[id]; 
        int year = jdid_posttime[id] - 2013;// start in 2013
        if (year >4 || year<0) continue; //to 2017
        index[3] = year;
        // these two keys must in the map.
        if (company_position.first == "" || company_position.second=="") continue;
        if (company_to_id.count(company_position.first)==0) continue;
        if (position_to_id.count(company_position.second)==0) continue;
        index[0] = company_to_id[company_position.first];
        index[1] = position_to_id[company_position.second];
        if ((company_fliter(company_position.first) && position_fliter(company_position.second))==false)  continue;
        auto demand_level_list = parse_jd_demand(job_description, skill_to_id, demand_level);
        for (auto& skill_level : demand_level_list)
        {
            index[2] = skill_to_id[skill_level.first];
            if (!skill_fliter(skill_level.first)) continue;
            tensor[ index ] = skill_level.second;
        }
		//if (id > 30) break;
    }
	myin.close();
    //
    std::clog<<"saving now"<<std::endl;
    save_new_data(tensor);
}




/****************************************************************************************/
/*                      private functions&var for this file                             */
/****************************************************************************************/

//wjy::MySQL sql("127.0.0.1", "wjy", "", "lagou_data");


// it's a (something, something_id) pair list. something_id helps to build the tensor.
void read_some_list(const std::string& path, std::unordered_map< std::string, size_t >& some_to_id, std::vector< std::string >& id_to_some)
{
    std::ifstream myin(path);
    // std::cout<<path<<"\n";
    assert(myin);
    some_to_id.clear();
    id_to_some.clear();
    
    std::string something;
    size_t n = 0;
    while (std::getline(myin, something))
    {
		if (some_to_id.find(something) != some_to_id.end()) continue;
        some_to_id[something] = n++;
        id_to_some.push_back(something);
    }
	myin.close();
}

void read_skill_list(const std::string& path)
{
    return read_some_list(path, skill_to_id, id_to_skill);
}

void read_company_list(const std::string& path)
{
    return read_some_list(path, company_to_id, id_to_company);
}

void read_position_list(const std::string& path)
{
    return read_some_list(path, position_to_id, id_to_position);
}

void read_demand_level(const std::string& path = "../data/demand_level.txt")
{
    std::ifstream myin(path);
    assert(myin);
    demand_level.clear();
    std::string word;
    int level;
    while (myin>>word>>level) demand_level[word] = level;
	myin.close();
}

void read_jdid_company_position(const std::string& path = "../data/raw_data/jdid_company_position.txt")
{
    std::ifstream myin(path);
    assert(myin);
    jdid_company_position.clear();
    std::string st;
    while(std::getline(myin, st))
	{
		auto sts = wjy::split(st, {'?'});
        //if (sts.size()!=3) std::clog<<sts[0]<<std::endl;
		int id = std::atoi(sts[0].c_str());
        //if (id<=0) std::clog<<st<<std::endl;
		//assert(id>0 && sts.size()==3);
        if (sts.size()!=3 || id<=0) continue;
		jdid_company_position[id] = std::make_pair(sts[1], sts[2]);
	}
	myin.close();
}

void read_jdid_posttime(const std::string& path = "../data/raw_data/jdid_posttime.txt")
{
    std::ifstream myin(path);
    assert(myin);
    jdid_posttime.clear();
    std::string st;
    while(std::getline(myin, st))
    {
        auto sts = wjy::split(st, {','});
        //if (sts.size()!=2) std::clog<<sts[0]<<std::endl;
        int id = std::atoi(sts[0].c_str());
        //if (id<=0) std::clog<<st<<std::endl;
        //assert(id>0 && sts.size()==2);
        if (sts.size()!=2 || id<=0)  continue;
        jdid_posttime[id] = std::atoi( sts[1].substr(0, 4).c_str() );
    }
    myin.close();
}

void read_raw_data()
{
    read_skill_list("../data/raw_data/skill_list.txt");
    read_company_list("../data/raw_data/company_list.txt");
    read_position_list("../data/raw_data/position_list.txt");
    read_demand_level();
    read_jdid_company_position();
    read_jdid_posttime();
}

void read_all(wjy::Date data_date)
{
    auto date = data_date.to_string("");
    read_skill_list("../data/"+date+"/skill_list_"+date+".txt");
    read_company_list("../data/"+date+"/company_list_"+date+".txt");
    read_position_list("../data/"+date+"/position_list_"+date+".txt");
//    read_demand_level();
//    read_jdid_company_position();
}


// input a jd, output all skills and it's demand.
std::list< std::pair<std::string, int> > parse_jd_demand(const std::vector< std::string >& jd, const std::unordered_map< std::string, size_t >& skill_list, const std::unordered_map< std::string, size_t >& demand_level)
{
    std::list< std::pair<std::string, int> > skill_level_list;
	std::unordered_set< std::string > skills;
    auto i = jd.size(), j = i;
    do{
        i--;
        auto skill = skill_list.find(jd[i]);
        if (skill == skill_list.end() || skills.find(skill->first)!=skills.end()) continue;
        if (j>i) // two skills may have the same level of demand
            for (j=i-1; j>0; j--)   //jd[0] is jdid, so here j in [1, i-1]
			{
                if (demand_level.find(jd[j]) != demand_level.end()) break;
			}
		if (j==0) break;
        skill_level_list.push_back( std::make_pair(jd[i], demand_level.at(jd[j])) );
		skills.insert(skill->first);
    }while (i!=1);// jd[0] is the id for this jd
    return skill_level_list;
}

void prepare_filter()
{
    skill_counter.clear();
    company_counter.clear();
    position_counter.clear();
    for (auto &cp : jdid_company_position)
    {
        auto & com = cp.second.first;
        auto & pos = cp.second.second;
        company_counter[com]++;
        position_counter[pos]++;
    }
    std::ifstream myin("../data/raw_data/position_cut.txt");
    assert(myin);
    std::string st_jd;
    while (std::getline(myin, st_jd))
    {
        auto job_description = wjy::split(st_jd, {',','/'});
        for (auto & word : job_description)
            if (skill_to_id.find(word)!=skill_to_id.end()) skill_counter[word]++;
    }
    myin.close();
}
bool skill_fliter(const std::string& skill)
{
    return (skill_counter[skill] >= 10);
}
bool company_fliter(const std::string& company)
{
    return (company_counter[company] >= 20);
}
bool position_fliter(const std::string& position)
{
    return (position_counter[position] >= 10);
}
void save_new_data(const wjy::sparse_tensor<double, 4> t)
{
    std::unordered_map< std::string, size_t > skill_to_new_id, company_to_new_id, position_to_new_id;
    auto save_something = [](auto fliter, auto & old_map, auto & new_map, const auto & path){
        std::ofstream myout(path);
        assert(myout);
        size_t n = 0;
        for (auto & p : old_map)
        {
            if (!fliter(p.first)) continue;
            new_map[p.first] = n++;
            myout<<p.first<<std::endl;
        }
        myout.close();
    };
    auto today = wjy::Date{}.to_string("");
    auto path = "../data/"+today+"/";
    save_something(skill_fliter, skill_to_id, skill_to_new_id, path+"skill_list_"+today+".txt");
    save_something(company_fliter, company_to_id, company_to_new_id, path+"company_list_"+today+".txt");
    save_something(position_fliter, position_to_id, position_to_new_id, path+"position_list_"+today+".txt");
    
    wjy::sparse_tensor<double, 4> new_tensor;
    wjy::sparse_tensor_index<4> new_index;
    for (auto & p : t)
    {
        auto & old_index = p.first;
        new_index[0] = company_to_new_id[ id_to_company[ old_index[0] ] ];
        new_index[1] = position_to_new_id[ id_to_position[ old_index[1] ] ];
        new_index[2] = skill_to_new_id[ id_to_skill[ old_index[2] ] ];
        new_index[3] = old_index[3];
        new_tensor[new_index] = p.second;
    }
    wjy::save_sparse_tensor(new_tensor, path+"tensor_dim4_"+today+".txt");
}


/*
std::pair< std::string, std::string > get_company_position(int id)
{
	std::cout<<id<<std::endl;
    if (!sql) sql.connect();
    std::string q = "select compname,position from clean_positions where id = \'"+ std::to_string(id) + "\';";
    sql.query(q);
    auto rslt = sql.get_result_table();
    return std::make_pair( rslt[0][0], rslt[0][1] );
}
*/


std::vector< std::vector< std::pair<std::string, int> > > print_top_k_company(int k)
{
    read_jdid_company_position();
    read_jdid_posttime();

    std::vector< std::vector< std::pair<std::string, int> > > ans;

    std::vector< std::unordered_map< std::string , int > > counters(5);

    for (auto & p : jdid_company_position)
    {
        auto & id = p.first;
        auto & company = p.second.first;
        auto & position = p.second.second;

        auto year = jdid_posttime[id];
        if (year<2013 || 2017<year) continue;
        year = year - 2013;

        counters[year][company] += 1;
    }

    for (int i=0; i<5; i++)
    {
        auto & counter = counters[i];
        std::vector< std::pair<std::string, int> > v(counter.begin(), counter.end());
        std::sort(v.begin(), v.end(), [](auto&x, auto&y){return x.second > y.second;});
        if (k > 0) v.resize(k);
        ans.push_back(v);
    }

    // wjy::Date date(2018, 9, 6);
    // read_all(date);

    // std::vector< std::vector< std::pair<std::string, int> > > ans;

    // auto  tensors = wjy::split_sparse_tensor(tensor, 3);
    // for (int i=0; i<5; i++)
    // {
    //     std::unordered_map< std::string , int > company_counter, position_counter;

    //     auto t = wjy::flatten_sparse_tensor(tensors[i], 2);
    //     for (auto & entry : t)
    //     {
    //         auto & index = entry.first;
    //         auto company = id_to_company[index[0]];
    //         auto position = id_to_position[index[1]];
    //         company_counter[company] += 1;
    //         position_counter[position] += 1;
    //     }
    //     auto & c = company_counter;

    //     std::vector< std::pair<std::string, int> > v(c.begin(), c.end());

    //     std::sort(v.begin(), v.end(), [](auto&x, auto&y){return x.second > y.second;});
    //     if (k > 0) v.resize(k);
    //     // std::cout<<2013+i<<std::endl;
    //     // for (auto & p : v)
    //     //     std::cout<<p.first<<" "<<p.second<<std::endl;
    //     ans.push_back(v);
    // }
    return ans;
}

std::vector< std::vector< std::pair<std::string, int> > > print_top_k_position_of_company(const std::string& company_name, int k)
{
    read_jdid_company_position();
    read_jdid_posttime();

    std::vector< std::vector< std::pair<std::string, int> > > ans;

    std::vector< std::unordered_map< std::string , int > > counters(5);

    for (auto & p : jdid_company_position)
    {
        auto & id = p.first;
        auto & company = p.second.first;
        auto & position = p.second.second;

        if (company_name != company) continue;

        auto year = jdid_posttime[id];
        if (year<2013 || 2017<year) continue;
        year = year - 2013;

        counters[year][position] += 1;
    }

    for (int i=0; i<5; i++)
    {
        auto & counter = counters[i];
        std::vector< std::pair<std::string, int> > v(counter.begin(), counter.end());
        std::sort(v.begin(), v.end(), [](auto&x, auto&y){return x.second > y.second;});
        if (k > 0) v.resize(k);
        ans.push_back(v);
    }

    // for (int i=0; i<5; i++)
    // {
    //     std::unordered_map< std::string , int > position_counter;

    //     auto t = wjy::flatten_sparse_tensor(tensors[i], 2);
    //     for (auto & entry : t)
    //     {
    //         auto & index = entry.first;
    //         if (index[0] != company_id) continue;
    //         auto position = id_to_position[index[1]];
    //         position_counter[position] += 1;
    //     }
    //     auto & c = position_counter;

    //     std::vector< std::pair<std::string, int> > v(c.begin(), c.end());

    //     std::sort(v.begin(), v.end(), [](auto&x, auto&y){return x.second > y.second;});
    //     if (k > 0) v.resize(k);
    //     // std::cout<<2013+i<<std::endl;
    //     // for (auto & p : v)
    //     //     std::cout<<p.first<<" "<<p.second<<std::endl;
    //     ans.push_back(v);
    // }
    return ans;
}

std::vector< std::vector<double> > company_skills_count(std::size_t company_id, const wjy::sparse_tensor<double, 4> & tensor)
{
    std::vector< std::vector<double> > counter(5);
    for (auto & c : counter) c.resize(680, 0);
    for (auto & entry : tensor)
    {
        auto & index = entry.first;
        auto & value = entry.second;
        if (index[0]==company_id) counter[ index[3] ][ index[2] ] += value;
    }
    return counter;
}
void print_companies_skills_count(const std::vector< std::string >& companies, std::ostream& myout )
{
    read_all(wjy::Date(2018, 9, 6));
    wjy::sparse_tensor<double, 4> tensor;
    wjy::load_sparse_tensor(tensor, "../data/20180906/tensor_dim4_20180906.txt");

    for (auto & company : companies)
    {
        std::size_t company_id = company_to_id[company];
        auto result = company_skills_count(company_id, tensor);
        for (auto & row : result)
        {
            std::string spliter = "";
            for (auto & x : row) 
            {
                myout<<spliter<<x;
                spliter=" ";
            }
            myout<<std::endl;
        }
        // myout<<std::endl;
    }

}


void category_of_company()
{
    std::string tensor_path = "../data/20180906/tensor_dim4_20180906.txt";
    std::string save_path = "../data/20180906/";
    wjy::sparse_tensor<double ,4> tensor;
    wjy::load_sparse_tensor(tensor, tensor_path);
    auto tensors = wjy::split_sparse_tensor(wjy::flatten_sparse_tensor(tensor, 3), 0);
    for (int i=20; i<=100; i+=20)
    {
        auto c = k_means_for_tensors(tensors, i, 200);
        std::ofstream myout(save_path+std::to_string(i)+"_category_of_company.txt");
        myout<<c.size()<<std::endl;
        for (auto & x : c) myout<<x<<" ";
        myout.close();
    }
}


//for test
void test()
{
	//read_all();
	//create_tensor();
}




