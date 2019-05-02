#include <stdexcept>

class NoGenerator: public std::runtime_error
{
    public:
        NoGenerator(std::string msg): std::runtime_error(msg){}

};
