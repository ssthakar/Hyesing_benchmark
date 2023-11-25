// main file, run code here
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/meshgrid.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <fstream>
#include <iostream>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/torch.h>
#include "nn_main.h"
#include "utils.h"

//- loads in python like indexing of tensors
using namespace torch::indexing;

//- util functions to write tensors to file to later plot using matplotlib
void writeTensorToFile(const torch::Tensor& tensor, const std::string& filename) {
    // Check if the tensor is 2D
  if (tensor.ndimension() == 2) 
  {
    // Get the sizes of the tensor
    int64_t numRows = tensor.size(0);
    int64_t numCols = tensor.size(1);

    // Open the file for writing
    std::ofstream outputFile(filename);

    // Check if the file is opened successfully
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // Iterate over the tensor elements and write them to the file
    for (int64_t i = 0; i < numRows; ++i) {
        for (int64_t j = 0; j < numCols; ++j) {
            // Write each element to the file
            outputFile << tensor.index({i, j}).item<float>() << " ";
        }
        outputFile << std::endl; // Move to the next row in the file
    }

    // Close the file
    outputFile.close();
  }
  if(tensor.ndimension() == 1)
  {
    int64_t  numRows = tensor.size(0);
    std::ofstream outputFile(filename);
    // Check if the file is opened successfully
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }
     // Iterate over the tensor elements and write them to the file
    for(int64_t i = 0; i < numRows; ++i) 
    {
      // Write each element to the file
      outputFile << tensor.index({i}).item<float>() << "\n";
    }
    outputFile << std::endl; // Move to the next row in the file
    }

}

//- util functions to write tensors to file to later plot using matplotlib
void writeTensorToFile(torch::Tensor& tensor, torch::Tensor& additionalTensor, const std::string& filename) {
    // Check if both tensors are 2D and have compatible sizes
    if (tensor.ndimension() != 2 || additionalTensor.ndimension() != 1 || tensor.size(0) != additionalTensor.size(0)) {
        std::cerr << "Error: Incompatible tensors or unsupported dimensions." << std::endl;
        return;
    }

    // Get the sizes of the tensor
    int64_t numRows = tensor.size(0);
    int64_t numCols = tensor.size(1);

    // Open the file for writing
    std::ofstream outputFile(filename);

    // Check if the file is opened successfully
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // Iterate over the tensor elements and write them to the file
    for (int64_t i = 0; i < numRows; ++i) {
        for (int64_t j = 0; j < numCols; ++j) {
            // Write x, y, and C to the file
            outputFile << i << " " << j << " " << additionalTensor[i].item<float>() << " ";
            outputFile << tensor.index({i, j}).item<float>() << std::endl;
        }
    }

    // Close the file
    outputFile.close();
}

//- main 
int main(int argc,char * argv[])
{
  
  // check if CUDA is avalilable and train on GPU if yes
  auto cuda_available = torch::cuda::is_available();
  auto device_str = cuda_available ? torch::kCUDA : torch::kCPU;
  //- create device 
  torch::Device device(device_str);
  //- Info out
  std::cout << (cuda_available ? "CUDA available. Training on GPU.\n" : "Training on CPU.\n") << '\n';
  
  //- create common Dictionary for both nets
  //- both nets share the same architecture, only network params update
  Dictionary netDict = Dictionary("../params.txt");
  
  //- create first net primary net, is the one being trained
  auto net1 = PinNet(netDict);
  //- create second net place holder for converged net 
  auto net2 = PinNet(netDict);
  
  //- load nets to device if available
  net1->to(device);
  net2->to(device);
  
  //- create dict for mesh
  Dictionary meshDict = Dictionary("../mesh.txt");
  //- create dict for thermoPhysical class
  Dictionary thermoDict = Dictionary("../thermo.txt");
  //- create thermoPhysical object
  thermoPhysical thermo(thermoDict);
  //- create Mesh
  mesh2D mesh(meshDict,net1,net2,device,thermo);
  // torch::Tensor testLoss = CahnHillard::PDEloss(mesh);
  // std::cout<<testLoss<<"\n";
  
  //- code to save initial condition for the phase field variable 
  mesh.iIC_ = torch::stack
  (
    {
      torch::flatten(mesh.initialGrid_[0]),
      torch::flatten(mesh.initialGrid_[1]),
      torch::flatten(mesh.initialGrid_[2])
    },1
  );

  writeTensorToFile(mesh.iIC_,"total.txt");
  torch::Tensor C = CahnHillard::C_at_InitialTime(mesh);
  writeTensorToFile(C,"intial.txt");


  //- declare optimizer instance to be used in training
  //- learning rate is decreased over the traning process and the optimizer class instance is changed
  torch::optim::Adam adam_optim1(mesh.net_->parameters(), torch::optim::AdamOptions(1e-4));  
  torch::optim::Adam adam_optim2(mesh.net_->parameters(), torch::optim::AdamOptions(1e-5)); 
  torch::optim::Adam adam_optim3(mesh.net_->parameters(), torch::optim::AdamOptions(1e-6));


  //- trainig begins here
  int iter=1;
  float loss;

  //- file to print out loss history
  std::ofstream lossFile("loss.txt");
  std::cout<<"Traning...\n";
  
  //- start profile clock
  auto start_time = std::chrono::high_resolution_clock::now();
  // mesh.update(0);
  //- epoch loop
  while(iter<=mesh.net_->K_EPOCH)
  {
    //- define closure for optimizer class to work with
    auto closure = [&](torch::optim::Optimizer &optim)
    { 
      float totalLoss=0.0;
      for(int i=0;i<mesh.net_->NITER_;i++)
      {
        //- clear the gradients wrt to network parameters and input features
        optim.zero_grad();
        //- generate solution fields from forward pass, accumulate gradients
        mesh.update(i);
        // std::cout<<mesh.pdeIndices_.sizes()<<"\n";
        // std::cout<<mesh.iPDE_.sizes()<<"\n";
        //- get total loss for the optimizer (PDE IC BC)
        auto loss = CahnHillard::loss(mesh);
        // auto loss = Heat::loss(mesh);
        //- back propogate
        loss.backward();
        //- update network parameters
        optim.step();
        //- get the total loss across all iterations within the epoch
        totalLoss +=loss.item<float>();
      }
      //- return the average loss
      return totalLoss/mesh.net_->NITER_;
    };
    std::cout<<iter<<"\n";
    //- change learning rate 
    if (iter <= 1000)
    {
      loss = closure(adam_optim1);
    } 
    else if(iter <= 2000) 
    { 
      loss = closure(adam_optim2);
    }
    else
    {
      loss = closure(adam_optim3);
    }

    //- info out to terminal
    if (iter % 10 == 0) 
    {
      std::cout << "  iter=" << iter << ", loss=" << std::setprecision(7) << loss<<" lr: "<<adam_optim1.defaults().get_lr()<<"\n";
      // lossFile<<iter<<" "<<loss<<"\n";
    }

    //- stop training if target loss achieved
    if (loss < mesh.net_->ABS_TOL) 
    {
      //- save model to file for post processing
      // torch::save(net,"saved_model.pt");
      //- update iter to get correct iter count
      iter += 1;
      //- update netPrev_ field in the mesh class instance
      // loadState(mesh.net_,mesh.netPrev_);
      std::cout<<iter<<std::endl;
      // break;
	  }
    iter += 1;
  }
  torch::Tensor test =torch::slice(mesh.pdeIndices_,0,4000,5000);
  // std::cout<<test<<"\n";
  //- stuff for profiling
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
  
  //- for plotting
  /*
  mesh.iIC_ = torch::stack
  (
    {
      torch::flatten(mesh.mesh_[0]),
      torch::flatten(mesh.mesh_[1]),
      // torch::full_like(torch::flatten(mesh.mesh_[2]),0.1)
      torch::flatten(mesh.mesh_[2])
    },1
  );
  */
  // torch::Tensor C1 = mesh.net_->forward(mesh.iIC_);
  
  writeTensorToFile(mesh.iPDE_,"randomSample.txt");
  // writeTensorToFile(C1,"initial1.txt"); 
  // std::cout<<mesh.thermo_.Mo;
  /*
  // loadState(net2,mesh1.net_);
  torch::Tensor xy = torch::stack({mesh.xyGrid[0].flatten(),mesh.xyGrid[1].flatten()},1);
  torch::Tensor uFinal = mesh.net_->forward(xy);
  writeTensorToFile(xy, "output23.txt");
  writeTensorToFile(uFinal, "output.txt");
  //-
  torch::Tensor yValues = torch::arange(0, 1.005, 0.005,device);

  // Repeat the x value (0.5) for each y value
  torch::Tensor xValues = torch::full_like(yValues, 0.75,device);

  // Create a 2D tensor by stacking x and y values
  torch::Tensor tensor2D = torch::stack({xValues, yValues}, 1);

  // PinNet netPost;
  // torch::load(netPost,"saved_model.pt");
  torch::Tensor test = mesh.net_->forward(tensor2D);
  writeTensorToFile(test, "test.txt");
  writeTensorToFile(tensor2D, "testXY.txt");
  

  */

 
  
  return 0;

} 
