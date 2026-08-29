#ifndef _GibbsComplex_PARSER_h
#define _GibbsComplex_PARSER_h

#include <iostream>
#include <string>

#include "antsCommandLineParser.h"

class GibbsComplex_PARSER : public itk::ants::CommandLineParser
{
public:
    GibbsComplex_PARSER( int argc , char * argv[] );
    ~GibbsComplex_PARSER();

    std::string getInputImageName();
    std::string getOutputImageName();
    std::string getOutputMagnitudeName();
    int         getPEDir();
    bool        getDoPOCS();
    int         getPOCSIters();
    float       getPOCSTol();
    float       getPFFactor();      // <= 0 when not supplied
    std::string getPFSide();        // "" when not supplied
    bool        getForcePF();
    float       getZeroTol();
    int         getNsh();
    int         getMinW();
    int         getMaxW();
    int         getNumberOfCores();
    bool        getDisableITKThreads();

private:
    void CreateParserandFillText(int argc , char * argv[] );
    void InitializeCommandLineOptions();
    bool checkIfAllRequiredParamsAreEntered();
};

#endif
