// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"
#include <memory>
#include <iostream>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/framework/logging.h" 


int main()
{
    // {
    //     using namespace tensorflow;
    //     Session * session;
    //     Status status = NewSession(SessionOptions(), &session);
    //     if (!status.ok()) {
    //         std::cerr << status.ToString() << std::endl;
    //         return 1;
    //     } else {
    //         std::cout << "Session created successfully" << std::endl;
    //     }
    //     return 0;
    // }
    // return 0;



    std::string modelPath = "model";
    const auto savedModelBundle = std::make_unique<tensorflow::SavedModelBundleLite>();

    // Create dummy options.
    tensorflow::SessionOptions sessionOptions;
    tensorflow::RunOptions runOptions;

    // Load the model bundle.
    const auto loadResult = tensorflow::LoadSavedModel(
            sessionOptions,
            runOptions,
            modelPath, //std::string containing path of the model bundle
            { "serve" },
            savedModelBundle.get());

    // Check if loading was okay.
    TF_CHECK_OK(loadResult);


    // Provide input data.
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 32, 32, 3}));
    for (std::size_t i = 0 ; i < 32 * 32 * 3 ; ++i) {
        tensor.flat<float>()(i) = 1.0;
    }

    std::vector<std::pair<std::string, tensorflow::Tensor>> feedInputs = { {"serving_default_conv2d_input", tensor} };
    std::vector<std::string> fetches = { "StatefulPartitionedCall:0" };
    
    // We need to store the results somewhere.
    std::vector<tensorflow::Tensor> outputs;

    // Let's run the model...
    auto status = savedModelBundle->GetSession()->Run(feedInputs, fetches, {}, &outputs);
    TF_CHECK_OK(status);

    // ... and print out it's predictions.
    for (const auto& record : outputs) {
        LOG(INFO) << record.DebugString();
    }
}