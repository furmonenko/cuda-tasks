#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>
#include "input_args_parser/input_args_parser.h"
#include "utils/input_handler.h"
#include "utils/filter_utils.h"
#include "utils/filter_pipeline.h"
#include "kernels/kernels.h"
#include <sstream>
#include <vector>

std::vector<std::string> splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter))
    {
        if (!token.empty())
        {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

void setupFilterPipeline(cuda_filter::FilterPipeline& pipeline, const cuda_filter::FilterOptions& options)
{
    if (!options.enablePipeline || options.pipelineFilters.empty())
    {
        // Single filter mode - add the specified filter
        cuda_filter::FilterType filterType = cuda_filter::FilterUtils::stringToFilterType(options.filterType);
        
        if (filterType == cuda_filter::FilterType::HDR_TONEMAPPING)
        {
            pipeline.addHDRFilter(options.toneMappingAlgorithm, options.exposure, options.gamma, options.saturation);
        }
        else
        {
            pipeline.addFilter(filterType, options.intensity);
        }
        
        PLOG_INFO << "Single filter mode: " << options.filterType;
    }
    else
    {
        // Pipeline mode - parse comma-separated filters
        std::vector<std::string> filterNames = splitString(options.pipelineFilters, ',');
        
        for (const auto& filterName : filterNames)
        {
            cuda_filter::FilterType filterType = cuda_filter::FilterUtils::stringToFilterType(filterName);
            
            if (filterType == cuda_filter::FilterType::HDR_TONEMAPPING)
            {
                pipeline.addHDRFilter(options.toneMappingAlgorithm, options.exposure, options.gamma, options.saturation);
            }
            else
            {
                pipeline.addFilter(filterType, options.intensity);
            }
        }
        
        PLOG_INFO << "Pipeline mode with " << filterNames.size() << " filters: " << options.pipelineFilters;
    }
    
    // Configure pipeline settings
    pipeline.setMultiStreamMode(options.enableMultiStream);
    pipeline.setCpuGpuComparison(true); // Always enable for performance comparison
    
    if (options.enableMultiStream)
    {
        PLOG_INFO << "Multi-stream processing enabled";
    }
    
    if (options.showPerformanceMetrics)
    {
        PLOG_INFO << "Detailed performance metrics enabled";
    }
}

int main(int argc, char **argv)
{
    // Initialize logger
    plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    // Parse command line arguments
    cuda_filter::InputArgsParser parser(argc, argv);
    cuda_filter::FilterOptions options = parser.parseArgs();

    // Initialize input handler
    cuda_filter::InputHandler inputHandler(options);
    if (!inputHandler.isOpened())
    {
        PLOG_ERROR << "Failed to initialize input source";
        return -1;
    }

    // Create and setup filter pipeline
    cuda_filter::FilterPipeline pipeline;
    setupFilterPipeline(pipeline, options);
    
    // Print pipeline information
    pipeline.printPipelineInfo();

    cv::Mat frame, pipelineOutput, cpuOutput, gpuOutput;
    
    // Transition variables
    bool transitionActive = false;
    int transitionFromStage = 0;
    int transitionToStage = 1;
    float transitionTime = 0.0f;
    auto transitionStartTime = std::chrono::high_resolution_clock::now();

    PLOG_INFO << "Press 'ESC' to exit, 'T' for transition demo, 'P' for pipeline info, '1'-'5' to select transition target";

    while (true)
    {
        // Capture frame
        if (!inputHandler.readFrame(frame))
        {
            PLOG_ERROR << "Failed to read frame";
            break;
        }

        // Update transition progress
        if (transitionActive)
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - transitionStartTime);
            float progress = elapsed.count() / (options.transitionDuration * 1000.0f);
            
            if (progress >= 1.0f)
            {
                pipeline.endTransition();
                transitionActive = false;
                PLOG_INFO << "Transition completed";
            }
            else
            {
                pipeline.updateTransition(progress);
            }
        }

        // Process frame through pipeline
        if (pipeline.getStageCount() > 0)
        {
            if (options.enablePipeline && pipeline.getStageCount() > 1)
            {
                // Multi-stage pipeline mode
                pipeline.processFrame(frame, pipelineOutput);
                
                // Add pipeline info text
                std::string pipelineText = "Pipeline (" + std::to_string(pipeline.getStageCount()) + " stages)";
                std::string fpsText = "FPS: " + std::to_string(static_cast<int>(pipeline.getMonitor()->getFrameRate()));
                std::string timeText = "Time: " + std::to_string(pipeline.getMonitor()->getTotalTime()).substr(0, 5) + "ms";
                
                cv::putText(pipelineOutput, pipelineText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                cv::putText(pipelineOutput, fpsText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                cv::putText(pipelineOutput, timeText, cv::Point(10, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                
                // Show performance metrics if enabled
                if (options.showPerformanceMetrics)
                {
                    pipeline.visualizePerformance(pipelineOutput);
                }
                
                // Display result
                if (options.preview)
                {
                    inputHandler.displaySideBySide(frame, pipelineOutput);
                }
                else
                {
                    inputHandler.displayFrame(pipelineOutput);
                }
            }
            else
            {
                // Single filter mode with CPU/GPU comparison
                pipeline.processFrameComparison(frame, cpuOutput, gpuOutput);
                
                // Add performance text
                std::string cpuText = "CPU FPS: " + std::to_string(static_cast<int>(pipeline.getMonitor()->getFrameRate()));
                std::string gpuText = "GPU Pipeline FPS: " + std::to_string(static_cast<int>(pipeline.getMonitor()->getFrameRate()));
                
                cv::putText(cpuOutput, cpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                cv::putText(gpuOutput, gpuText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                
                // Create side-by-side comparison
                cv::Mat combined;
                cv::hconcat(cpuOutput, gpuOutput, combined);
                cv::putText(combined, "CPU Version", cv::Point(10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                cv::putText(combined, "GPU Pipeline", cv::Point(combined.cols / 2 + 10, combined.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
                
                // Display result
                if (options.preview)
                {
                    inputHandler.displaySideBySide(frame, combined);
                }
                else
                {
                    inputHandler.displayFrame(combined);
                }
            }
        }
        else
        {
            // No filters - display original
            inputHandler.displayFrame(frame);
        }

        // Handle keyboard input
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) // ESC
        {
            break;
        }
        else if (key == 't' || key == 'T') // Transition demo
        {
            if (pipeline.getStageCount() >= 2 && !transitionActive)
            {
                transitionFromStage = 0;
                transitionToStage = 1;
                pipeline.startTransition(transitionFromStage, transitionToStage, 
                                        options.enableTransitions ? cuda_filter::TransitionEngine::TransitionType::WIPE_LEFT_TO_RIGHT
                                                                  : cuda_filter::TransitionEngine::TransitionType::FADE);
                transitionActive = true;
                transitionStartTime = std::chrono::high_resolution_clock::now();
                PLOG_INFO << "Started transition demo from stage " << transitionFromStage << " to " << transitionToStage;
            }
            else if (pipeline.getStageCount() < 2)
            {
                PLOG_WARNING << "Need at least 2 stages for transition demo";
            }
        }
        else if (key == 'p' || key == 'P') // Pipeline info
        {
            pipeline.printPipelineInfo();
            pipeline.getMonitor()->printStatistics();
        }
        else if (key >= '1' && key <= '5') // Stage selection for transitions
        {
            int stageIndex = key - '1';
            if (stageIndex < pipeline.getStageCount())
            {
                transitionToStage = stageIndex;
                PLOG_INFO << "Set transition target to stage " << stageIndex;
            }
        }
    }

    // Print final performance statistics
    PLOG_INFO << "Final Pipeline Performance:";
    pipeline.getMonitor()->printStatistics();
    
    PLOG_INFO << "Application terminated";
    return 0;
}
