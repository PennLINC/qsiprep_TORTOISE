#include "gibbs_complex_parser.h"
#include <algorithm>
#include <ctype.h>

GibbsComplex_PARSER::GibbsComplex_PARSER( int argc , char * argv[] )
{
    CreateParserandFillText(argc,argv);
    this->Parse(argc,argv);

    if( argc == 1 )
    {
        this->PrintMenu( std::cout, 5, false );
        exit(EXIT_FAILURE);
    }

    if(checkIfAllRequiredParamsAreEntered()==0)
    {
        std::cout<<"Not all the required Parameters are entered! Exiting!"<<std::endl;
        exit(EXIT_FAILURE);
    }
}

GibbsComplex_PARSER::~GibbsComplex_PARSER()
{
}

void GibbsComplex_PARSER::CreateParserandFillText(int argc, char* argv[])
{
    this->SetCommand( argv[0] );

    std::string commandDescription = std::string( "Partial-Fourier-aware Gibbs ringing correction for complex-valued DWIs. Reads a complex-valued (COMPLEX64) 4D NIFTI, optionally restores the un-acquired partial-Fourier k-space region with POCS, and applies the Kellner et al. local subvoxel-shift method to the complex data instead of its magnitude. For magnitude-only data, use the Gibbs command instead." );

    this->SetCommandDescription( commandDescription );
    this->InitializeCommandLineOptions();
}

void GibbsComplex_PARSER::InitializeCommandLineOptions()
{
    typedef itk::ants::CommandLineParser::OptionType OptionType;

    {
        std::string description = std::string( "Full path to the input complex-valued 4D NIFTI (COMPLEX64 or COMPLEX128). REQUIRED." );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'i');
        option->SetLongName( "input");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Full path to the output complex-valued 4D NIFTI (COMPLEX64). REQUIRED." );
        OptionType::Pointer option = OptionType::New();
        option->SetShortName( 'o');
        option->SetLongName( "output");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Optional. Full path for an additional magnitude-only output NIFTI, written in the same pass." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "output_magnitude");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Phase encoding direction. 0: horizontal, 1: vertical. Same convention as the Gibbs command. Default: 1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pe_dir");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Run POCS partial-Fourier reconstruction before unringing (0/1). Skipped automatically if the data does not look zero-filled. Default: 1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pocs");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Maximum number of POCS iterations (int). Default: 10" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pocs_iters");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "POCS relative-change stopping tolerance (float). Default: 1e-4" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pocs_tol");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Override the detected partial-Fourier factor (float, e.g. 0.75 or 0.875). By default the factor is detected from the k-space zero band." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pf_factor");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Override the detected truncated side of k-space: low or high. low means the most-negative-ky lines are missing." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "pf_side");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Proceed even when the declared and detected partial-Fourier geometry disagree, using the declared values (0/1). Default: 0" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "force_pf");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Normalised-energy threshold below which a k-space line counts as empty (float). Default: 1e-6" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "zero_tol");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Number of subvoxel shifts for the unringing (int). Default: 25" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "nsh");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Minimum window size for the unringing (int). Default: 1" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "minW");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Maximum window size for the unringing (int). Default: 3" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "maxW");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Number of cores to use (int). Caps both the ITK and OpenMP thread counts." );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "ncores");
        option->SetDescription( description );
        this->AddOption( option );
    }
    {
        std::string description = std::string( "Pin ITK to a single thread so the ITK and OpenMP layers cannot multiply (0/1). Default: 0" );
        OptionType::Pointer option = OptionType::New();
        option->SetLongName( "disable_itk_threads");
        option->SetDescription( description );
        this->AddOption( option );
    }
}

std::string GibbsComplex_PARSER::getInputImageName()
{
    OptionType::Pointer option = this->GetOption( "input");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

std::string GibbsComplex_PARSER::getOutputImageName()
{
    OptionType::Pointer option = this->GetOption( "output");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

std::string GibbsComplex_PARSER::getOutputMagnitudeName()
{
    OptionType::Pointer option = this->GetOption( "output_magnitude");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

int GibbsComplex_PARSER::getPEDir()
{
    OptionType::Pointer option = this->GetOption( "pe_dir");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 1;
}

bool GibbsComplex_PARSER::getDoPOCS()
{
    OptionType::Pointer option = this->GetOption( "pocs");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return true;
}

int GibbsComplex_PARSER::getPOCSIters()
{
    OptionType::Pointer option = this->GetOption( "pocs_iters");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 10;
}

float GibbsComplex_PARSER::getPOCSTol()
{
    OptionType::Pointer option = this->GetOption( "pocs_tol");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
        return 1e-4;
}

float GibbsComplex_PARSER::getPFFactor()
{
    OptionType::Pointer option = this->GetOption( "pf_factor");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
        return -1.;
}

std::string GibbsComplex_PARSER::getPFSide()
{
    OptionType::Pointer option = this->GetOption( "pf_side");
    if(option->GetNumberOfFunctions())
        return option->GetFunction(0)->GetName();
    else
        return std::string("");
}

bool GibbsComplex_PARSER::getForcePF()
{
    OptionType::Pointer option = this->GetOption( "force_pf");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return false;
}

float GibbsComplex_PARSER::getZeroTol()
{
    OptionType::Pointer option = this->GetOption( "zero_tol");
    if(option->GetNumberOfFunctions())
        return (atof(option->GetFunction(0)->GetName().c_str()));
    else
        return 1e-6;
}

int GibbsComplex_PARSER::getNsh()
{
    OptionType::Pointer option = this->GetOption( "nsh");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 25;
}

int GibbsComplex_PARSER::getMinW()
{
    OptionType::Pointer option = this->GetOption( "minW");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 1;
}

int GibbsComplex_PARSER::getMaxW()
{
    OptionType::Pointer option = this->GetOption( "maxW");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 3;
}

int GibbsComplex_PARSER::getNumberOfCores()
{
    OptionType::Pointer option = this->GetOption( "ncores");
    if(option->GetNumberOfFunctions())
        return (int)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return 0;
}

bool GibbsComplex_PARSER::getDisableITKThreads()
{
    OptionType::Pointer option = this->GetOption( "disable_itk_threads");
    if(option->GetNumberOfFunctions())
        return (bool)(atoi(option->GetFunction(0)->GetName().c_str()));
    else
        return false;
}

bool GibbsComplex_PARSER::checkIfAllRequiredParamsAreEntered()
{
    if(this->getInputImageName()==std::string(""))
    {
        std::cout<<"Input image name not entered...Exiting..."<<std::endl;
        return 0;
    }
    if(this->getOutputImageName()==std::string(""))
    {
        std::cout<<"Output image name not entered...Exiting..."<<std::endl;
        return 0;
    }
    std::string side = this->getPFSide();
    if(side!=std::string("") && side!=std::string("low") && side!=std::string("high"))
    {
        std::cout<<"pf_side must be either low or high...Exiting..."<<std::endl;
        return 0;
    }
    return 1;
}
