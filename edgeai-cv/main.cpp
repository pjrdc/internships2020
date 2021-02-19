// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core.hpp"
#include "utils.hpp"
#include "tracker.hpp"
#include "descriptor.hpp"
#include "distance.hpp"
#include "detector.hpp"
#include "image_reader.hpp"
#include "pedestrian_tracker_demo.hpp"

#include <opencv2/core.hpp>

#include <iostream>
#include <utility>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <gflags/gflags.h>

#include <unordered_map>
#include <chrono>

using namespace InferenceEngine;
using ImageWithFrameIndex = std::pair<cv::Mat, int>;

std::unique_ptr<PedestrianTracker>
CreatePedestrianTracker(const std::string& reid_model,
                        const std::string& reid_weights,
                        const InferenceEngine::Core & ie,
                        const std::string & deviceName,
                        bool should_keep_tracking_info) {
    TrackerParams params;

    if (should_keep_tracking_info) {
        params.drop_forgotten_tracks = false;
        params.max_num_objects_in_track = -1;
    }

    std::unique_ptr<PedestrianTracker> tracker(new PedestrianTracker(params));

    // Load reid-model.
    std::shared_ptr<IImageDescriptor> descriptor_fast =
        std::make_shared<ResizedImageDescriptor>(
            cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
    std::shared_ptr<IDescriptorDistance> distance_fast =
        std::make_shared<MatchTemplateDistance>();

    tracker->set_descriptor_fast(descriptor_fast);
    tracker->set_distance_fast(distance_fast);

    if (!reid_model.empty() && !reid_weights.empty()) {
        CnnConfig reid_config(reid_model, reid_weights);
        reid_config.max_batch_size = 16;

        std::shared_ptr<IImageDescriptor> descriptor_strong =
            std::make_shared<DescriptorIE>(reid_config, ie, deviceName);

        if (descriptor_strong == nullptr) {
            THROW_IE_EXCEPTION << "[SAMPLES] internal error - invalid descriptor";
        }
        std::shared_ptr<IDescriptorDistance> distance_strong =
            std::make_shared<CosDistance>(descriptor_strong->size());

        tracker->set_descriptor_strong(descriptor_strong);
        tracker->set_distance_strong(distance_strong);
    } else {
        std::cout << "WARNING: Either reid model or reid weights "
            << "were not specified. "
            << "Only fast reidentification approach will be used." << std::endl;
    }

    return tracker;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    return true;
}

int main_work(int argc, char **argv) {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }


    // Reading command line parameters.

    auto detlog_out = FLAGS_out;
  
    auto custom_cpu_library = FLAGS_l;
    auto path_to_custom_layers = FLAGS_c;
    bool should_use_perf_counter = FLAGS_pc;

    bool should_print_out = FLAGS_r;

    bool should_show = !FLAGS_no_show;
    int delay = FLAGS_delay;
    if (!should_show)
        delay = -1;
    should_show = (delay >= 0);

    bool should_save_det_log = !detlog_out.empty();
  
    std::string det_model = "C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.4.287\\deployment_tools\\tools\\model_downloader\\intel\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.xml";
    std::string det_weights = "C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.4.287\\deployment_tools\\tools\\model_downloader\\intel\\person-detection-retail-0013\\FP32\\person-detection-retail-0013.bin";

    std::string reid_model = "C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.4.287\\deployment_tools\\tools\\model_downloader\\intel\\person-reidentification-retail-0248\\FP32\\person-reidentification-retail-0248.xml";
    std::string reid_weights = "C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.4.287\\deployment_tools\\tools\\model_downloader\\intel\\person-reidentification-retail-0248\\FP32\\person-reidentification-retail-0248.bin";

    //human pose
    std::string human_pose_model = "C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.4.287\\deployment_tools\\tools\\model_downloader\\intel\\human-pose-estimation-0001\\FP32\\human-pose-estimation-0001.xml";
    std::string human_pose_weights = "C:\\Program Files (x86)\\IntelSWTools\\openvino_2020.4.287\\deployment_tools\\tools\\model_downloader\\intel\\human-pose-estimation-0001\\FP32\\human-pose-estimation-0001.bin";

    auto detector_mode = "CPU";
    auto reid_mode = "CPU";
    auto human_pose_detector_mode = "CPU";

    std::vector<std::string> devices{detector_mode, reid_mode};
    InferenceEngine::Core ie =
        LoadInferenceEngine(
            devices, custom_cpu_library, path_to_custom_layers,
            should_use_perf_counter);

    DetectorConfig detector_confid(det_model, det_weights);
    ObjectDetector pedestrian_detector(detector_confid, ie, detector_mode);

    bool should_keep_tracking_info = should_save_det_log || should_print_out;
    std::unique_ptr<PedestrianTracker> tracker =
        CreatePedestrianTracker(reid_model, reid_weights, ie, reid_mode,
                                should_keep_tracking_info);

    DetectorConfig human_pose_config(human_pose_model);
    ObjectDetector human_pose_detector(human_pose_config, ie, human_pose_detector_mode);
  
    // Opening video.
    cv::VideoCapture cap;

    cap.open("C:\\Users\\Seq\\Documents\\Intel\\OpenVINO\\omz_demos_build\\intel64\\Release\\Safety_Full_Hat_and_Vest.mp4");
    cv::Mat frame;

    double video_fps = 25;
  
    int frame_idx = 0;

    cv::Size graphSize{static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60};
    Presenter presenter(FLAGS_u, 10, graphSize);
 
    for (;;) {
        cap.read(frame);
        frame_idx = frame_idx + 1;

        pedestrian_detector.submitFrame(frame, frame_idx);
        pedestrian_detector.waitAndFetchResults();

        TrackedObjects detections = pedestrian_detector.getResults();
    
        human_pose_detector.submitFrame(frame, frame_idx);
        human_pose_detector.waitAndFetchResults();

        TrackedObjects humanPoseDetections = human_pose_detector.getResults();

        // timestamp in milliseconds
        uint64_t cur_timestamp = static_cast<uint64_t >(1000.0 / video_fps * frame_idx);
        tracker->Process(frame, detections, cur_timestamp);

        if (should_show) {
            // Drawing colored "worms" (tracks).
            frame = tracker->DrawActiveTracks(frame);

            // Drawing all detected objects on a frame by BLUE COLOR
            for (const auto &detection : detections) {
                cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
            }

            // Drawing tracked detections only by RED color and print ID and detection
            // confidence level.
            for (const auto &detection : tracker->TrackedDetections()) {
                cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
                std::string text = std::to_string(detection.object_id) +
                    " conf: " + std::to_string(detection.confidence);
                cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
                            1.0, cv::Scalar(0, 0, 255), 3);
            }
          
            //OVERALL PERSON COUNT
            std::string opc_count = "Laczna liczba osob: " + std::to_string(tracker->Count());
            cv::putText(frame, opc_count, cv::Point(5, 100), cv::FONT_ITALIC, 0.7, cv::Scalar(0, 0, 255), 3);

            // LIVE PERSON COUNT
            std::unordered_map<size_t, std::vector<cv::Point>> active_track_ids = tracker->GetActiveTracks();
            std::string lpc_count = "Liczba osob: " + std::to_string(active_track_ids.size());
            cv::putText(frame, lpc_count, cv::Point(5, 150), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 3);
          
            cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
            cv::imshow("dbg", frame);
            char k = cv::waitKey(delay);
            if (k == 27)
                break;
        }

        if (should_save_det_log && (frame_idx % 100 == 0)) {
            DetectionLog log = tracker->GetDetectionLog(true);
            SaveDetectionLogToTrajFile(detlog_out, log);
        }
    }

    if (should_keep_tracking_info) {
        DetectionLog log = tracker->GetDetectionLog(true);

        if (should_save_det_log)
            SaveDetectionLogToTrajFile(detlog_out, log);
        if (should_print_out)
            PrintDetectionLog(log);
    }
    if (should_use_perf_counter) {
        pedestrian_detector.PrintPerformanceCounts(getFullDeviceName(ie, FLAGS_d_det));
        tracker->PrintReidPerformanceCounts(getFullDeviceName(ie, FLAGS_d_reid));
    }
    return 0;
}

int main(int argc, char **argv) {
    try {
        main_work(argc, argv);
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "Execution successful" << std::endl;

    return 0;
}
