#include <iostream>
#include <string>
#include <vector>
#include <torch/script.h>

int main() {
    // Load the saved PyTorch model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("model.pth");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Initialize variables for user input and chatbot response
    std::string input_string;
    std::vector<int64_t> input_tensor_values;
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor input_tensor, output_tensor;
    std::string output_string;

    // Enter a loop to continually prompt the user for input and generate responses
    while (true) {
        // Get user input
        std::cout << "> ";
        std::getline(std::cin, input_string);

        // Convert user input to a PyTorch tensor
        input_tensor_values.clear();
        for (const auto& word : input_string) {
            input_tensor_values.push_back(word);
        }
        input_tensor = torch::tensor(input_tensor_values, torch::kLong);

        // Pass the input tensor through the model to get the output tensor
        inputs.clear();
        inputs.push_back(input_tensor);
        try {
            output_tensor = module.forward(inputs).toTensor();
        } catch (const c10::Error& e) {
            std::cerr << "Error running the model\n";
            continue;
        }

        // Convert the output tensor to a string and print it
        output_string.clear();
        for (int64_t i = 0; i < output_tensor.size(0); i++) {
            output_string.push_back(static_cast<char>(output_tensor[i].item<int64_t>()));
        }
        std::cout << output_string << "\n";
    }

    return 0;
}

// TO FIX ERROR:

// The error message "cannot open source file 'torch/script.h'" indicates that the C++ compiler cannot find the header file for the PyTorch script module. To fix this, you need to make sure that you have installed the PyTorch C++ API and have properly set up your build environment to include the PyTorch libraries and headers.

// Here are the steps to install the PyTorch C++ API on Linux:

// Install PyTorch using the instructions for your platform on the PyTorch website: https://pytorch.org/get-started/locally/

// Install the PyTorch C++ API using the following command:

// pip install torch-cpp
// Set up your build environment to include the PyTorch libraries and headers. This typically involves setting the following environment variables:

// LD_LIBRARY_PATH=/path/to/pytorch/lib:$LD_LIBRARY_PATH
// CPLUS_INCLUDE_PATH=/path/to/pytorch/include:$CPLUS_INCLUDE_PATH
// Make sure to replace "/path/to/pytorch" with the actual path to your PyTorch installation directory.

// If you are using a different platform, the installation steps may be different. Please refer to the PyTorch documentation for platform-specific instructions: https://pytorch.org/cppdocs/installing.html