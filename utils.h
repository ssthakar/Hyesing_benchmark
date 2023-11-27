#ifndef utils_H
#define utils_H
#include <string>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
//- for Dictionary class
#include <map>

//read in text file store in jagged vector // stores in all numerical data
template<typename T>
std::vector<std::vector<T>> readIn(std::string s,const char delim)
{
  std::vector<std::vector<T>> vec;
  std::ifstream file(s); 
  if(file.is_open()) 
  {
    std::string line; //outer string, stores in the entire line
    while(std::getline(file,line)) 
    {
      std::stringstream ss; 
      ss<<line; 
      std::vector<T> temp_vec;
      std::string word;
      double val;
      while(std::getline(ss,word,delim)) // inner string stream, stores in the word
      {
        if (std::stringstream(word)>> val) // converting from string to double
        {
          temp_vec.push_back(val);
        }
      }
      vec.push_back(temp_vec); //populate row
    }
  }
  else 
  {
    std::cout<<"file not found"<<std::endl;
  }
  //deal with empty rows with zero size
  for(size_t i=0;i<vec.size();)
  {
    if(vec[i].size()==0)
      vec.erase(vec.begin()+i);
    else i++;
  }
  return vec; //return the formatted storage
};


// overload of the readin function, formats vector to read in specific 
// number of cols
template<typename  T>
std::vector<std::vector<T>> readIn(std::string s,size_t cols,const char delim)
{
  std::vector<std::vector<T>> vec;
  std::ifstream file(s);
  if(file.is_open())
  {
    std::string line;
    while(std::getline(file,line))
    {
      std::stringstream ss;
      ss<<line;
      std::vector<T> temp_vec;
      std::string word;
      double val;
      while(std::getline(ss,word,delim))
      {
        if (std::stringstream(word)>> val)
        {
          temp_vec.push_back(val);
        }
      }
      vec.push_back(temp_vec);
    }
  }
  else 
  {
    std::cout<<"file not found"<<std::endl;
  }
  // remove any rows with less than the cols specified in the function arg
  for(size_t i=0;i<vec.size();)
  {
    if(vec[i].size()<cols)
      vec.erase(vec.begin()+i);
    else i++;
  }
  return vec;
};

// template function to write out 2D vector to file
template<typename T>
void writeOut(std::string fileName, std::vector<std::vector<T>> vec,const char delim)
{
  std::ofstream fileObj(fileName);
  for(size_t i=0;i<vec.size();i++)
  {
    for(size_t j=0;j<vec[i].size();j++)
    {
      fileObj<<vec[i][j]<<delim;
    }
    fileObj<<"\n";
  }
}

//- unorderd mem map to load in parameters needed in the training without recompling
class Dictionary 
{
  public:
    Dictionary(std::string fileName)
    {
      readFromFile(fileName);
    }
    // Function to add or update a key-value pair
    void set
    (
      const std::string& key, 
      const std::string& value
    ) 
    {
      data[key] = value;
    }
    
    // function to get value from Dictionary
    template <typename ValueType>
    ValueType get(const std::string& key) const 
    {
      auto it = data.find(key);
      if (it != data.end()) 
      {
        ValueType result;
        std::istringstream iss(it->second);
        iss >> result;
        return result;
      }
      return ValueType(); // Default value if key not found or conversion fails
    }
    
    // Function to read key-value pairs from a file
    void readFromFile(const std::string& filename) 
    {
      std::ifstream file(filename);
      if (file.is_open()) 
      {
        std::string line;
        while (std::getline(file, line)) 
        {
          std::istringstream iss(line);
          std::string key, value;
          if (iss >> key >> value) 
          {
            set(key, value);
          }
        }
        file.close();
      } 
      else 
      {
        std::cerr << "Unable to open file: " << filename << std::endl;
      }
    }

private:
    std::map<std::string, std::string> data;
};

//- TODO include matplotlib functionality here for plotting stuff




#endif
