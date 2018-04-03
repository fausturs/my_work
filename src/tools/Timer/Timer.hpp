//
//  Timer.hpp
//  test_c++
//
//  Created by wjy on 16/7/10.
//  Copyright © 2016年 wjy. All rights reserved.
//

/*
 Example
 with the head file #include <unistd.h>
 
 Timer timer;
 timer.start();
 sleep(100);//100s
 timer.end();
 
 Timer t1;
 t1.start();
 std::cout<<timer.get_duration_ns()<<" ns"<<std::endl;
 std::cout<<timer.get_duration_us()<<" us"<<std::endl;
 std::cout<<timer.get_duration_ms()<<" ms"<<std::endl;
 std::cout<<timer.get_duration_s()<<" s"<<std::endl;
 std::cout<<timer.get_duration_minute()<<" min"<<std::endl;
 t1.end();
 
 std::cout<<t1.get_duration_ns()<<std::endl;
 
 the output may looks like:
 100000966997 ns
 100000966 us
 100000 ms
 100 s
 1 min
 255533
 */

#ifndef Timer_hpp
#define Timer_hpp

#include <chrono>
#include <ratio>
namespace wjy
{
    class Timer
    {
        std::chrono::high_resolution_clock::time_point start_time,end_time;
        std::chrono::nanoseconds duration;
        
    public:
        
        void start();
        void end();
        
        long long get_duration_ns();
        long long get_duration_us();
        long long get_duration_ms();
        long long get_duration_s();
        long long get_duration_minute();
        long long get_duration_hour();
        
    };
    
}
#endif /* Timer_hpp */
