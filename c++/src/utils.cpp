#include "utils.h"
#include <algorithm>

char *cmdOptionParser(char** begin, char** end, const std::string& option){
    char **itr = std::find(begin, end, option);
    
    if ( (option.size() >= 2) && (option.substr(0,2)== "--")){
    
        if (itr != end) return *itr;
        else return 0;

    }
    else
        if ( (itr != end) && ((itr+1) != end) ) return *(itr+1);
        else return 0;
}

bool cmdOptionExist(char** begin, char** end, const std::string& option){
    return cmdOptionParser(begin, end, option) != 0;
}
