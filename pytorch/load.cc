#include <iostream>
#include <memory>
#include <Eigen/Eigen>

#include <torch/script.h>

class ExternalModelTorch {
public:
  ExternalModelTorch(char const * filename)
    : _module(torch::jit::load(filename))
  {
  }


  // no batch
  Eigen::VectorXd forward(Eigen::VectorXd const & vec) {
    torch::Tensor t = torch::empty({vec.rows()});
    double * data = t.data_ptr<double>();

    {
      Eigen::Map<Eigen::VectorXd> ef(data, t.size(0));
      ef = vec;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(std::move(t));

    Eigen::VectorXd out;
    {
      at::Tensor output = _module.forward(inputs).toTensor();
      Eigen::Map<Eigen::VectorXd> ef(output.data_ptr<double>(), output.size(0));
      out = ef;
    }
    return out;
  }

private:
  torch::jit::script::Module _module;
};


int main(int argc, char const * argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example <script>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << module.num_slots() << std::endl;
  std::cout << "ok\n";

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({3, 32, 32}));

  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output << std::endl;
}
