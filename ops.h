#include <iostream>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

template <typename Func, typename... Args>
double measure_exec_time(Func&& func, Args&&... args) {
    std::vector<double> du;
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();  
        func(std::forward<Args>(args)...);  
        auto end = std::chrono::high_resolution_clock::now();    
        std::chrono::duration<double> elap = end - start;
        auto elaps = elap.count();
        std::cout << elaps << "s" << std::endl;
        du.push_back(elaps);
    }
    auto time_taken = std::accumulate(du.begin()+1, du.end(), 0.0) / (du.size()-1);      
    std::cout << "Execution Time: " << time_taken << " s" << std::endl;
    return time_taken;        
}