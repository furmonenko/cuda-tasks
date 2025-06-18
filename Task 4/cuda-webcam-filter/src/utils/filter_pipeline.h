#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include "filter_utils.h"

namespace cuda_filter
{

    // Forward declarations
    class PerformanceMonitor;
    class TransitionEngine;

    // Filter stage configuration
    struct FilterStage
    {
        FilterType type;
        float intensity;
        float exposure;      // For HDR filters
        float gamma;         // For HDR filters
        float saturation;    // For HDR filters
        ToneMappingAlgorithm algorithm; // For HDR filters
        cv::Mat kernel;      // For convolution filters
        cudaStream_t stream; // CUDA stream for this stage
        bool enabled;

        FilterStage(FilterType t, float i = 1.0f) 
            : type(t), intensity(i), exposure(1.0f), gamma(2.2f), 
              saturation(1.0f), algorithm(ToneMappingAlgorithm::REINHARD), 
              stream(nullptr), enabled(true) {}
    };

    // Performance monitoring class
    class PerformanceMonitor
    {
    private:
        std::vector<double> stageTimes;
        std::vector<double> stageHistory;
        double totalPipelineTime;
        double lastFrameTime;
        int frameCount;
        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point stageStartTime;

    public:
        PerformanceMonitor();
        ~PerformanceMonitor();

        void startFrame();
        void endFrame();
        void startStage(int stageId);
        void endStage(int stageId);
        
        double getTotalTime() const { return totalPipelineTime; }
        double getStageTime(int stageId) const;
        double getFrameRate() const;
        
        void reset();
        void visualizeTimings(cv::Mat& display, int x = 10, int y = 10);
        void printStatistics() const;
    };

    // Transition effects between filters
    class TransitionEngine
    {
    public:
        enum class TransitionType
        {
            NONE,
            WIPE_LEFT_TO_RIGHT,
            WIPE_TOP_TO_BOTTOM,
            FADE,
            BLEND
        };

        static void applyWipeTransition(const cv::Mat& filter1Result, 
                                       const cv::Mat& filter2Result,
                                       cv::Mat& output, 
                                       float progress);
        
        static void applyFadeTransition(const cv::Mat& filter1Result,
                                       const cv::Mat& filter2Result,
                                       cv::Mat& output,
                                       float progress);

        static void applyBlendTransition(const cv::Mat& filter1Result,
                                        const cv::Mat& filter2Result,
                                        cv::Mat& output,
                                        float progress);
    };

    // Main filter pipeline class
    class FilterPipeline
    {
    private:
        std::vector<std::unique_ptr<FilterStage>> stages;
        std::vector<cv::Mat> intermediateBuffers;
        std::vector<cv::Mat> cpuBuffers;
        std::vector<cv::Mat> gpuBuffers;
        
        std::unique_ptr<PerformanceMonitor> monitor;
        std::unique_ptr<TransitionEngine> transitionEngine;
        
        // CUDA streams management
        std::vector<cudaStream_t> streams;
        bool useMultipleStreams;
        
        // Transition state
        bool transitionActive;
        int transitionFromStage;
        int transitionToStage;
        float transitionProgress;
        TransitionEngine::TransitionType transitionType;
        
        // Performance comparison
        bool enableCpuGpuComparison;
        
        void initializeBuffers(int width, int height, int channels);
        void cleanupStreams();
        void processStageGPU(int stageIdx, const cv::Mat& input, cv::Mat& output);
        void processStageGPU_MultiStream(int stageIdx, const cv::Mat& input, cv::Mat& output);
        void processStageCPU(int stageIdx, const cv::Mat& input, cv::Mat& output);

    public:
        FilterPipeline();
        ~FilterPipeline();

        // Pipeline management
        void addFilter(FilterType type, float intensity = 1.0f);
        void addHDRFilter(ToneMappingAlgorithm algorithm, float exposure = 1.0f, 
                         float gamma = 2.2f, float saturation = 1.0f);
        void removeFilter(int index);
        void clearPipeline();
        
        // Filter configuration
        void setFilterEnabled(int index, bool enabled);
        void setFilterIntensity(int index, float intensity);
        void setHDRParameters(int index, float exposure, float gamma, float saturation);
        
        // Processing
        void processFrame(const cv::Mat& input, cv::Mat& output);
        void processFrameComparison(const cv::Mat& input, cv::Mat& cpuOutput, cv::Mat& gpuOutput);
        
        // Transitions
        void startTransition(int fromStage, int toStage, 
                           TransitionEngine::TransitionType type = TransitionEngine::TransitionType::WIPE_LEFT_TO_RIGHT);
        void updateTransition(float progress); // 0.0 to 1.0
        void endTransition();
        
        // Configuration
        void setMultiStreamMode(bool enable) { useMultipleStreams = enable; }
        void setCpuGpuComparison(bool enable) { enableCpuGpuComparison = enable; }
        
        // Information
        int getStageCount() const { return static_cast<int>(stages.size()); }
        FilterType getStageType(int index) const;
        bool isStageEnabled(int index) const;
        
        // Performance monitoring
        PerformanceMonitor* getMonitor() { return monitor.get(); }
        void resetPerformanceCounters();
        
        // Debug and visualization
        void visualizePerformance(cv::Mat& display);
        void printPipelineInfo() const;
    };

} // namespace cuda_filter