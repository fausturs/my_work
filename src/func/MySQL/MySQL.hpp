//
//  MySQL.hpp
//  mysql_lab
//
//  Created by wjy on 16/4/10.
//  Copyright © 2016年 wjy. All rights reserved.
//
//  在工程文件中 要添加如下一些东西
//  Header Search Paths             添加了/usr/local/mysql/include /usr/local/include
//  Library Search Paths            添加了/usr/local/mysql/lib     /usr/local/lib
//  Other Linker Flages             添加了-lmysqlclient -lz -lm

#ifndef MySQL_hpp
#define MySQL_hpp

#include <stdio.h>
#include <string>
#include <vector>

#include "mysql.h"

class MySQL{
//
    MYSQL mysql;
    //一些登陆数据库必须的变量
    std::string ip;
    std::string usr_name;
    std::string pwd;
    std::string database_name;
    //用来标志是否已经连接到了数据库
    bool is_connected;
    
    //用来零时储存一个表
    std::vector<std::string> table_head;
    std::vector< std::vector<std::string> > table;
    //储存错误信息 任何一次数据库的操作 都会刷新这个值
    std::string error_info;
    

public:
    MySQL() = delete;
    MySQL(const MySQL &) = delete;
    MySQL& operator=(const MySQL&) = delete;
	MySQL(MySQL&&) = delete;
	MySQL& operator=(MySQL&&) = delete;
    //仅有的构造方式
    MySQL(std::string ip,std::string usr_name,std::string pwd,std::string database_name);
    ~MySQL();

    //连接与断开
    int connect();
    int disconnect();
    //
    bool is_connect();
    
    //接受一个mysql命令  执行成功返回0 执行失败返回1
    //若有别的返回值  如查询命令需要返回一个表 则以其他形式返回
    int query(std::string st);
    //这两个函数返回mysql查询的结果  一个返回表头  一个返回表内容
    const std::vector<std::string>& get_result_table_head() const;
    const std::vector< std::vector<std::string> >& get_result_table() const;
    const std::string& get_error_info() const;
    //结果表的行和列的数量
    unsigned long get_table_rows();
    unsigned long get_table_cols();
    
    //返回的bool值表示是否已经连接到一个数据库
    bool operator ()();
    operator bool();
    //
    const std::vector< std::string >& operator [](unsigned long i) const;
};


#endif /* MySQL_hpp */
