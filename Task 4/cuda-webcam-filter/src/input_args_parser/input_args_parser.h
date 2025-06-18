#pragma once

#include <string>
#include <cxxopts.hpp>
#include "../utils/filter_utils.h"

namespace cuda_filter
{

    enum class InputSource
    {
        WEBCAM,
        IMAGE,
        VIDEO,
        SYNTHETIC
    };

    enum class SyntheticPattern
    {
        CHECKERBOARD,
        GRADIENT,
        NOISE
    };


    struct FilterOptions
    {
        InputSource inputSource;
        std::string inputPath;
        SyntheticPattern syntheticPattern;
        int deviceId;
        std::string filterType;
        int kernelSize;
        float sigma;
        float intensity;
        bool preview;
        
        // HDR tone mapping parameters
        float exposure;
        float gamma;
        float saturation;
        ToneMappingAlgorithm toneMappingAlgorithm;
        
        // Pipeline parameters
        std::string pipelineFilters;     // Comma-separated filter list
        bool enablePipeline;             // Enable pipeline mode
        bool enableMultiStream;          // Enable multi-stream processing
        bool enableTransitions;          // Enable filter transitions
        float transitionDuration;       // Transition duration in seconds
        bool showPerformanceMetrics;    // Show detailed performance info
    };

    class InputArgsParser
    {
    public:
        InputArgsParser(int argc, char **argv);

        FilterOptions parseArgs();

    private:
        int m_argc;
        char **m_argv;

        void setupOptions(cxxopts::Options &options);
        InputSource stringToInputSource(const std::string &str);
        SyntheticPattern stringToSyntheticPattern(const std::string &str);
        ToneMappingAlgorithm stringToToneMappingAlgorithm(const std::string &str);
    };

} // namespace cuda_filter
