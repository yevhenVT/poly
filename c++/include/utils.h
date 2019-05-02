#ifndef UTILS_H
#define UTILS_H

#include <string>

char *cmdOptionParser(char** begin, char** end, const std::string& option); // parse command line to find option value
bool cmdOptionExist(char** begin, char** end, const std::string& option); // check if option exist in command line arguments

#endif
