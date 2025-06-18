#include "filter_pipeline.h"
#include "filter_utils.h"
#include "../kernels/kernels.h"
#include <plog/Log.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace cuda_filter
{

//=============================================================================
// PerformanceMonitor Implementation
//=============================================================================

PerformanceMonitor::PerformanceMonitor()
    : totalPipelineTime(0.0), lastFrameTime(0.0), frameCount(0)
{
    stageTimes.clear();
    stageHistory.clear();
}

PerformanceMonitor::~PerformanceMonitor() = default;

void PerformanceMonitor::startFrame()
{
    startTime = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::endFrame()
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    lastFrameTime = duration.count() / 1000.0; // Convert to milliseconds
    totalPipelineTime = lastFrameTime;
    frameCount++;
}

void PerformanceMonitor::startStage(int stageId)
{
    if (stageId >= static_cast<int>(stageTimes.size()))
    {
        stageTimes.resize(stageId + 1, 0.0);
    }
    stageStartTime = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::endStage(int stageId)
{
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - stageStartTime);
    if (stageId < static_cast<int>(stageTimes.size()))
    {
        stageTimes[stageId] = duration.count() / 1000.0; // Convert to milliseconds
    }
}

double PerformanceMonitor::getStageTime(int stageId) const
{
    if (stageId >= 0 && stageId < static_cast<int>(stageTimes.size()))
    {
        return stageTimes[stageId];
    }
    return 0.0;
}

double PerformanceMonitor::getFrameRate() const
{
    return lastFrameTime > 0.0 ? 1000.0 / lastFrameTime : 0.0;
}

void PerformanceMonitor::reset()
{
    stageTimes.clear();
    stageHistory.clear();
    totalPipelineTime = 0.0;
    lastFrameTime = 0.0;
    frameCount = 0;
}

void PerformanceMonitor::visualizeTimings(cv::Mat& display, int x, int y)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1);
    
    // Overall performance
    ss << "Pipeline FPS: " << getFrameRate();
    cv::putText(display, ss.str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    
    ss.str("");
    ss << "Total Time: " << totalPipelineTime << "ms";
    cv::putText(display, ss.str(), cv::Point(x, y + 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    
    // Individual stage times
    for (size_t i = 0; i < stageTimes.size(); ++i)
    {
        ss.str("");
        ss << "Stage " << i << ": " << stageTimes[i] << "ms";
        cv::putText(display, ss.str(), cv::Point(x, y + 50 + static_cast<int>(i) * 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }
}

void PerformanceMonitor::printStatistics() const
{
    PLOG_INFO << "=== Pipeline Performance Statistics ===";
    PLOG_INFO << "Frame Rate: " << getFrameRate() << " FPS";
    PLOG_INFO << "Total Pipeline Time: " << totalPipelineTime << " ms";
    
    for (size_t i = 0; i < stageTimes.size(); ++i)
    {
        PLOG_INFO << "Stage " << i << " Time: " << stageTimes[i] << " ms";
    }
}

//=============================================================================
// TransitionEngine Implementation
//=============================================================================

void TransitionEngine::applyWipeTransition(const cv::Mat& filter1Result, 
                                          const cv::Mat& filter2Result,
                                          cv::Mat& output, 
                                          float progress)
{
    if (filter1Result.size() != filter2Result.size() || filter1Result.type() != filter2Result.type())
    {
        PLOG_ERROR << "Input images must have same size and type for transition";
        filter1Result.copyTo(output);
        return;
    }
    
    output.create(filter1Result.size(), filter1Result.type());
    
    int wipePosition = static_cast<int>(progress * filter1Result.cols);
    
    // Left part: filter1
    if (wipePosition > 0)
    {
        cv::Rect leftRect(0, 0, std::min(wipePosition, filter1Result.cols), filter1Result.rows);
        filter1Result(leftRect).copyTo(output(leftRect));
    }
    
    // Right part: filter2
    if (wipePosition < filter1Result.cols)
    {
        cv::Rect rightRect(wipePosition, 0, filter1Result.cols - wipePosition, filter1Result.rows);
        filter2Result(rightRect).copyTo(output(rightRect));
    }
}

void TransitionEngine::applyFadeTransition(const cv::Mat& filter1Result,
                                          const cv::Mat& filter2Result,
                                          cv::Mat& output,
                                          float progress)
{
    cv::addWeighted(filter1Result, 1.0f - progress, filter2Result, progress, 0, output);
}

void TransitionEngine::applyBlendTransition(const cv::Mat& filter1Result,
                                           const cv::Mat& filter2Result,
                                           cv::Mat& output,
                                           float progress)
{
    // Circular blend effect
    cv::Point2f center(filter1Result.cols / 2.0f, filter1Result.rows / 2.0f);
    float maxRadius = std::sqrt(center.x * center.x + center.y * center.y);
    float currentRadius = progress * maxRadius;
    
    output.create(filter1Result.size(), filter1Result.type());
    filter1Result.copyTo(output);
    
    cv::Mat mask = cv::Mat::zeros(filter1Result.size(), CV_8UC1);
    cv::circle(mask, center, static_cast<int>(currentRadius), cv::Scalar(255), -1);
    
    filter2Result.copyTo(output, mask);
}

//=============================================================================
// FilterPipeline Implementation
//=============================================================================

FilterPipeline::FilterPipeline()
    : useMultipleStreams(false), transitionActive(false), transitionFromStage(-1),
      transitionToStage(-1), transitionProgress(0.0f), 
      transitionType(TransitionEngine::TransitionType::NONE),
      enableCpuGpuComparison(false)
{
    monitor = std::make_unique<PerformanceMonitor>();
    transitionEngine = std::make_unique<TransitionEngine>();
    
    PLOG_INFO << "FilterPipeline initialized";
}

FilterPipeline::~FilterPipeline()
{
    cleanupStreams();
    PLOG_INFO << "FilterPipeline destroyed";
}

void FilterPipeline::addFilter(FilterType type, float intensity)
{
    auto stage = std::make_unique<FilterStage>(type, intensity);
    
    // Create CUDA stream for this stage
    if (useMultipleStreams)
    {
        cudaStream_t stream;
        cudaError_t error = cudaStreamCreate(&stream);
        if (error != cudaSuccess)
        {
            PLOG_ERROR << "Failed to create CUDA stream: " << cudaGetErrorString(error);
            stage->stream = nullptr;
        }
        else
        {
            stage->stream = stream;
        }
    }
    
    // Create kernel for convolution filters
    if (type != FilterType::HDR_TONEMAPPING)
    {
        stage->kernel = FilterUtils::createFilterKernel(type, 3, intensity);
    }
    
    stages.push_back(std::move(stage));
    
    PLOG_INFO << "Added filter to pipeline. Total stages: " << stages.size();
}

void FilterPipeline::addHDRFilter(ToneMappingAlgorithm algorithm, float exposure, 
                                 float gamma, float saturation)
{
    auto stage = std::make_unique<FilterStage>(FilterType::HDR_TONEMAPPING, 1.0f);
    stage->algorithm = algorithm;
    stage->exposure = exposure;
    stage->gamma = gamma;
    stage->saturation = saturation;
    
    // Create CUDA stream for this stage
    if (useMultipleStreams)
    {
        cudaStream_t stream;
        cudaError_t error = cudaStreamCreate(&stream);
        if (error != cudaSuccess)
        {
            PLOG_ERROR << "Failed to create CUDA stream: " << cudaGetErrorString(error);
            stage->stream = nullptr;
        }
        else
        {
            stage->stream = stream;
        }
    }
    
    stages.push_back(std::move(stage));
    
    PLOG_INFO << "Added HDR filter to pipeline. Total stages: " << stages.size();
}

void FilterPipeline::removeFilter(int index)
{
    if (index >= 0 && index < static_cast<int>(stages.size()))
    {
        // Clean up CUDA stream
        if (stages[index]->stream != nullptr)
        {
            cudaStreamDestroy(stages[index]->stream);
        }
        
        stages.erase(stages.begin() + index);
        PLOG_INFO << "Removed filter from pipeline. Total stages: " << stages.size();
    }
}

void FilterPipeline::clearPipeline()
{
    cleanupStreams();
    stages.clear();
    intermediateBuffers.clear();
    cpuBuffers.clear();
    gpuBuffers.clear();
    
    PLOG_INFO << "Pipeline cleared";
}

void FilterPipeline::setFilterEnabled(int index, bool enabled)
{
    if (index >= 0 && index < static_cast<int>(stages.size()))
    {
        stages[index]->enabled = enabled;
    }
}

void FilterPipeline::setFilterIntensity(int index, float intensity)
{
    if (index >= 0 && index < static_cast<int>(stages.size()))
    {
        stages[index]->intensity = intensity;
        
        // Recreate kernel for convolution filters
        if (stages[index]->type != FilterType::HDR_TONEMAPPING)
        {
            stages[index]->kernel = FilterUtils::createFilterKernel(
                stages[index]->type, 3, intensity);
        }
    }
}

void FilterPipeline::setHDRParameters(int index, float exposure, float gamma, float saturation)
{
    if (index >= 0 && index < static_cast<int>(stages.size()) && 
        stages[index]->type == FilterType::HDR_TONEMAPPING)
    {
        stages[index]->exposure = exposure;
        stages[index]->gamma = gamma;
        stages[index]->saturation = saturation;
    }
}

void FilterPipeline::initializeBuffers(int width, int height, int channels)
{
    int numStages = static_cast<int>(stages.size());
    
    intermediateBuffers.clear();
    cpuBuffers.clear();
    gpuBuffers.clear();
    
    // Create intermediate buffers for each stage
    for (int i = 0; i <= numStages; ++i)
    {
        cv::Mat buffer(height, width, CV_8UC(channels));
        intermediateBuffers.push_back(buffer);
        
        if (enableCpuGpuComparison)
        {
            cv::Mat cpuBuffer(height, width, CV_8UC(channels));
            cv::Mat gpuBuffer(height, width, CV_8UC(channels));
            cpuBuffers.push_back(cpuBuffer);
            gpuBuffers.push_back(gpuBuffer);
        }
    }
}

void FilterPipeline::cleanupStreams()
{
    for (auto& stage : stages)
    {
        if (stage->stream != nullptr)
        {
            cudaStreamDestroy(stage->stream);
            stage->stream = nullptr;
        }
    }
}

void FilterPipeline::processFrame(const cv::Mat& input, cv::Mat& output)
{
    if (stages.empty())
    {
        input.copyTo(output);
        return;
    }
    
    monitor->startFrame();
    
    // Initialize buffers if needed
    if (intermediateBuffers.empty() || 
        intermediateBuffers[0].size() != input.size() ||
        intermediateBuffers[0].type() != input.type())
    {
        initializeBuffers(input.cols, input.rows, input.channels());
    }
    
    // Copy input to first buffer
    input.copyTo(intermediateBuffers[0]);
    
    // Process each stage
    for (int i = 0; i < static_cast<int>(stages.size()); ++i)
    {
        if (!stages[i]->enabled)
        {
            // Skip disabled stage - copy input to output
            intermediateBuffers[i].copyTo(intermediateBuffers[i + 1]);
            continue;
        }
        
        monitor->startStage(i);
        
        if (useMultipleStreams && stages[i]->stream != nullptr)
        {
            processStageGPU_MultiStream(i, intermediateBuffers[i], intermediateBuffers[i + 1]);
        }
        else
        {
            processStageGPU(i, intermediateBuffers[i], intermediateBuffers[i + 1]);
        }
        
        monitor->endStage(i);
    }
    
    // Synchronize all streams before final output
    if (useMultipleStreams)
    {
        for (const auto& stage : stages)
        {
            if (stage->stream != nullptr)
            {
                cudaStreamSynchronize(stage->stream);
            }
        }
    }
    
    // Handle transitions
    if (transitionActive && transitionFromStage >= 0 && transitionToStage >= 0 &&
        transitionFromStage < static_cast<int>(stages.size()) && 
        transitionToStage < static_cast<int>(stages.size()))
    {
        cv::Mat fromResult, toResult;
        
        // Process with from-stage only
        processStageGPU(transitionFromStage, input, fromResult);
        
        // Process with to-stage only  
        processStageGPU(transitionToStage, input, toResult);
        
        // Apply transition
        switch (transitionType)
        {
        case TransitionEngine::TransitionType::WIPE_LEFT_TO_RIGHT:
            TransitionEngine::applyWipeTransition(fromResult, toResult, output, transitionProgress);
            break;
        case TransitionEngine::TransitionType::FADE:
            TransitionEngine::applyFadeTransition(fromResult, toResult, output, transitionProgress);
            break;
        case TransitionEngine::TransitionType::BLEND:
            TransitionEngine::applyBlendTransition(fromResult, toResult, output, transitionProgress);
            break;
        default:
            intermediateBuffers.back().copyTo(output);
            break;
        }
    }
    else
    {
        // Normal processing - copy final result
        intermediateBuffers.back().copyTo(output);
    }
    
    monitor->endFrame();
}

void FilterPipeline::processStageGPU(int stageIdx, const cv::Mat& input, cv::Mat& output)
{
    const auto& stage = stages[stageIdx];
    
    if (stage->type == FilterType::HDR_TONEMAPPING)
    {
        int algorithmIndex = static_cast<int>(stage->algorithm);
        applyHDRToneMappingGPU(input, output, stage->exposure, stage->gamma, 
                              stage->saturation, algorithmIndex);
    }
    else
    {
        applyFilterGPU(input, output, stage->kernel);
    }
}

void FilterPipeline::processStageGPU_MultiStream(int stageIdx, const cv::Mat& input, cv::Mat& output)
{
    const auto& stage = stages[stageIdx];
    
    if (stage->stream == nullptr)
    {
        // Fallback to regular processing if no stream available
        processStageGPU(stageIdx, input, output);
        return;
    }
    
    // Use the stage's dedicated CUDA stream for processing
    if (stage->type == FilterType::HDR_TONEMAPPING)
    {
        int algorithmIndex = static_cast<int>(stage->algorithm);
        applyHDRToneMappingGPU_Stream(input, output, stage->exposure, stage->gamma, 
                                     stage->saturation, algorithmIndex, stage->stream);
    }
    else
    {
        applyFilterGPU_Stream(input, output, stage->kernel, stage->stream);
    }
    
    // Note: Synchronization is handled at pipeline level
}

void FilterPipeline::processStageCPU(int stageIdx, const cv::Mat& input, cv::Mat& output)
{
    const auto& stage = stages[stageIdx];
    
    if (stage->type == FilterType::HDR_TONEMAPPING)
    {
        int algorithmIndex = static_cast<int>(stage->algorithm);
        applyHDRToneMappingCPU(input, output, stage->exposure, stage->gamma, 
                              stage->saturation, algorithmIndex);
    }
    else
    {
        FilterUtils::applyFilterCPU(input, output, stage->kernel);
    }
}

void FilterPipeline::processFrameComparison(const cv::Mat& input, cv::Mat& cpuOutput, cv::Mat& gpuOutput)
{
    if (stages.empty())
    {
        input.copyTo(cpuOutput);
        input.copyTo(gpuOutput);
        return;
    }
    
    // Process GPU pipeline
    processFrame(input, gpuOutput);
    
    // Process CPU pipeline for comparison
    if (cpuBuffers.empty() || 
        cpuBuffers[0].size() != input.size() ||
        cpuBuffers[0].type() != input.type())
    {
        initializeBuffers(input.cols, input.rows, input.channels());
    }
    
    input.copyTo(cpuBuffers[0]);
    
    for (int i = 0; i < static_cast<int>(stages.size()); ++i)
    {
        if (!stages[i]->enabled)
        {
            cpuBuffers[i].copyTo(cpuBuffers[i + 1]);
            continue;
        }
        
        processStageCPU(i, cpuBuffers[i], cpuBuffers[i + 1]);
    }
    
    cpuBuffers.back().copyTo(cpuOutput);
}

void FilterPipeline::startTransition(int fromStage, int toStage, 
                                    TransitionEngine::TransitionType type)
{
    if (fromStage >= 0 && fromStage < static_cast<int>(stages.size()) &&
        toStage >= 0 && toStage < static_cast<int>(stages.size()) &&
        fromStage != toStage)
    {
        transitionActive = true;
        transitionFromStage = fromStage;
        transitionToStage = toStage;
        transitionType = type;
        transitionProgress = 0.0f;
        
        PLOG_INFO << "Started transition from stage " << fromStage << " to stage " << toStage;
    }
}

void FilterPipeline::updateTransition(float progress)
{
    if (transitionActive)
    {
        transitionProgress = std::clamp(progress, 0.0f, 1.0f);
    }
}

void FilterPipeline::endTransition()
{
    transitionActive = false;
    transitionFromStage = -1;
    transitionToStage = -1;
    transitionProgress = 0.0f;
    transitionType = TransitionEngine::TransitionType::NONE;
    
    PLOG_INFO << "Transition ended";
}

FilterType FilterPipeline::getStageType(int index) const
{
    if (index >= 0 && index < static_cast<int>(stages.size()))
    {
        return stages[index]->type;
    }
    return FilterType::IDENTITY;
}

bool FilterPipeline::isStageEnabled(int index) const
{
    if (index >= 0 && index < static_cast<int>(stages.size()))
    {
        return stages[index]->enabled;
    }
    return false;
}

void FilterPipeline::resetPerformanceCounters()
{
    monitor->reset();
}

void FilterPipeline::visualizePerformance(cv::Mat& display)
{
    monitor->visualizeTimings(display, 10, display.rows - 150);
}

void FilterPipeline::printPipelineInfo() const
{
    PLOG_INFO << "=== Filter Pipeline Information ===";
    PLOG_INFO << "Number of stages: " << stages.size();
    PLOG_INFO << "Multi-stream mode: " << (useMultipleStreams ? "Enabled" : "Disabled");
    PLOG_INFO << "CPU/GPU comparison: " << (enableCpuGpuComparison ? "Enabled" : "Disabled");
    
    for (size_t i = 0; i < stages.size(); ++i)
    {
        const auto& stage = stages[i];
        std::string typeStr;
        
        switch (stage->type)
        {
        case FilterType::BLUR: typeStr = "BLUR"; break;
        case FilterType::SHARPEN: typeStr = "SHARPEN"; break;
        case FilterType::EDGE_DETECTION: typeStr = "EDGE_DETECTION"; break;
        case FilterType::EMBOSS: typeStr = "EMBOSS"; break;
        case FilterType::HDR_TONEMAPPING: typeStr = "HDR_TONEMAPPING"; break;
        case FilterType::IDENTITY: typeStr = "IDENTITY"; break;
        default: typeStr = "UNKNOWN"; break;
        }
        
        PLOG_INFO << "Stage " << i << ": " << typeStr 
                  << ", Intensity: " << stage->intensity
                  << ", Enabled: " << (stage->enabled ? "Yes" : "No");
    }
}

} // namespace cuda_filter