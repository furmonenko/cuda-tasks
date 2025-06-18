#include "input_args_parser.h"
#include <iostream>
#include "../utils/version.h"

namespace cuda_filter
{

    InputArgsParser::InputArgsParser(int argc, char **argv)
        : m_argc(argc), m_argv(argv)
    {
    }

    InputSource InputArgsParser::stringToInputSource(const std::string &str)
    {
        if (str == "webcam")
            return InputSource::WEBCAM;
        if (str == "image")
            return InputSource::IMAGE;
        if (str == "video")
            return InputSource::VIDEO;
        if (str == "synthetic")
            return InputSource::SYNTHETIC;
        throw std::runtime_error("Invalid input source: " + str);
    }

    SyntheticPattern InputArgsParser::stringToSyntheticPattern(const std::string &str)
    {
        if (str == "checkerboard")
            return SyntheticPattern::CHECKERBOARD;
        if (str == "gradient")
            return SyntheticPattern::GRADIENT;
        if (str == "noise")
            return SyntheticPattern::NOISE;
        throw std::runtime_error("Invalid synthetic pattern: " + str);
    }

    ToneMappingAlgorithm InputArgsParser::stringToToneMappingAlgorithm(const std::string &str)
    {
        if (str == "reinhard")
            return ToneMappingAlgorithm::REINHARD;
        if (str == "drago")
            return ToneMappingAlgorithm::DRAGO;
        if (str == "mantiuk")
            return ToneMappingAlgorithm::MANTIUK;
        throw std::runtime_error("Invalid tone mapping algorithm: " + str);
    }

    FilterOptions InputArgsParser::parseArgs()
    {
        cxxopts::Options options("cuda-webcam-filter", "Real-time webcam filter with CUDA acceleration");

        setupOptions(options);

        auto result = options.parse(m_argc, m_argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("version"))
        {
            std::cout << "CUDA Webcam Filter version " << CUDA_WEBCAM_FILTER_VERSION << std::endl;
            exit(0);
        }

        FilterOptions filterOptions;

        // Parse input source
        std::string inputType = result["input"].as<std::string>();
        filterOptions.inputSource = stringToInputSource(inputType);
        filterOptions.inputPath = result["path"].as<std::string>();

        if (filterOptions.inputSource == InputSource::SYNTHETIC)
        {
            std::string patternType = result["synthetic"].as<std::string>();
            filterOptions.syntheticPattern = stringToSyntheticPattern(patternType);
        }
        else if (filterOptions.inputSource == InputSource::WEBCAM)
        {
            filterOptions.deviceId = result["device"].as<int>();
        }

        filterOptions.filterType = result["filter"].as<std::string>();
        filterOptions.kernelSize = result["kernel-size"].as<int>();
        filterOptions.sigma = result["sigma"].as<float>();
        filterOptions.intensity = result["intensity"].as<float>();
        filterOptions.preview = result.count("preview") > 0;

        // HDR tone mapping parameters
        filterOptions.exposure = result["exposure"].as<float>();
        filterOptions.gamma = result["gamma"].as<float>();
        filterOptions.saturation = result["saturation"].as<float>();
        std::string toneAlgorithm = result["tone-algorithm"].as<std::string>();
        filterOptions.toneMappingAlgorithm = stringToToneMappingAlgorithm(toneAlgorithm);

        return filterOptions;
    }

    void InputArgsParser::setupOptions(cxxopts::Options &options)
    {
        options.add_options()
            ("i,input", "Input source: 'webcam', 'image', 'video', or 'synthetic'", cxxopts::value<std::string>()->default_value("webcam"))
            ("p,path", "Path to input image or video file (when not using webcam)", cxxopts::value<std::string>()->default_value("test_image.jpg"))
            ("s,synthetic", "Synthetic pattern type: 'checkerboard', 'gradient', 'noise'", cxxopts::value<std::string>()->default_value("checkerboard"))
            ("d,device", "Camera device ID", cxxopts::value<int>()->default_value("0"))
            ("f,filter", "Filter type: blur, sharpen, edge, emboss, hdr", cxxopts::value<std::string>()->default_value("blur"))
            ("k,kernel-size", "Kernel size for filters", cxxopts::value<int>()->default_value("3"))
            ("sigma", "Sigma value for Gaussian blur", cxxopts::value<float>()->default_value("1.0"))
            ("intensity", "Filter intensity", cxxopts::value<float>()->default_value("1.0"))
            ("e,exposure", "HDR exposure value", cxxopts::value<float>()->default_value("1.0"))
            ("g,gamma", "HDR gamma correction", cxxopts::value<float>()->default_value("2.2"))
            ("saturation", "HDR saturation adjustment", cxxopts::value<float>()->default_value("1.0"))
            ("tone-algorithm", "Tone mapping algorithm: reinhard, drago, mantiuk", cxxopts::value<std::string>()->default_value("reinhard"))
            ("preview", "Show original video alongside filtered")
            ("h,help", "Print usage")
            ("v,version", "Print version information");
    }

} // namespace cuda_filter
