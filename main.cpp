// main file, run code here
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/meshgrid.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <fstream>
#include <iostream>
#include <string>
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
  Dictionary debugDict = Dictionary("../debug.txt");
  bool debug = debugDict.get<bool>("DEBUG");
  if(debug)
    std::cout<<"debug is: "<<debug<<"\n";
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
  /*mesh.iIC_ = torch::stack
  (
    {
      torch::flatten(mesh.initialGrid_[0]),
      torch::flatten(mesh.initialGrid_[1]),
      torch::flatten(mesh.initialGrid_[2])
    },1
  );
  */

  // writeTensorToFile(mesh.iIC_,"total.txt");
  // torch::Tensor C = CahnHillard::C_at_InitialTime(mesh);
  // writeTensorToFile(C,"intial.txt");
    
  if(debug)
  { 
    mesh.update(0);
    mesh.update(1);
    std::cout<<mesh.fieldsPDE_<<std::endl;
  }


  //- TODO intialize optim paras from dictionary to avoid recompilation 
  //- declare optimizer instance to be used in training
  //- learning rate is decreased over the traning process and the optimizer class instance is changed
  torch::optim::Adam adam_optim1(mesh.net_->parameters(), torch::optim::AdamOptions(1e-3));  
  torch::optim::Adam adam_optim2(mesh.net_->parameters(), torch::optim::AdamOptions(1e-4)); 
  torch::optim::Adam adam_optim3(mesh.net_->parameters(), torch::optim::AdamOptions(1e-5));


  // Put info statement here

  //- Time marching loop
  for(int N=0;N<2;N++)
  {
    //- set up epoch loop
    int iter=1;
    float loss;

    //- file to print out loss history
    std::ofstream lossFile("loss.txt");
    std::cout<<"Traning...\n";
    
    //- start profile clock
    auto start_time = std::chrono::high_resolution_clock::now();
    //- epoch loop
    while(iter<=mesh.net_->K_EPOCH)
    {
      //- define closure for optimizer class to work with
      auto closure = [&](torch::optim::Optimizer &optim)
      { 
        float totalLoss=0.0;
        for(int i=0;i<mesh.net_->NITER_;i++)
        {
          //- generate solution fields from forward pass, accumulate gradients
          mesh.update(i);
          //- get total loss for the optimizer (PDE,IC,BC)
          auto loss = CahnHillard::loss(mesh);
          //- back propogate and accumulate gradiets of loss wrt to parameters
          loss.backward();
          //- get the total loss across all iterations within the epoch
          totalLoss +=loss.item<float>();
        }
        //- update network parameters
        optim.step();
        //- clear gradients for next batch iteration
        optim.zero_grad();
        //- return the average loss
        return totalLoss/mesh.net_->NITER_;
      };

      //- print out iteration numbers
      if(debug)
      {
        std::cout<<iter<<"\n";
      }

      //- TODO make it more general by making it a Dict 
      //  and making all the other parameters as dicts as well
      //- learning rate schedule
      if (iter <= 4000)
      {
        loss = closure(adam_optim1);
      } 
      else if(iter <= 8000) 
      { 
        loss = closure(adam_optim2);
      }
      else
      {
        loss = closure(adam_optim3);
      }
      
      // TODO
      //- make a dict for all of this, do not recompile the code every time you change something trivial

      //- info out to terminal
      //- does not work on the cluster for some reason, rely on the loss.txt for info
      if (iter % 10 == 0) 
      {
        std::cout << "  iter=" << iter << ", loss=" << std::setprecision(7) << loss<<" lr: "<<adam_optim1.defaults().get_lr()<<"\n";
        // lossFile<<iter<<" "<<loss<<"\n";
      }
      if(iter % 5000 == 0)
      {
        std::cout<<"saving output..."<<"\n";
        std::string modelName = "pNetSave" + std::to_string(mesh.ubT_);
        torch::Tensor grid = torch::stack
        (
          {
            torch::flatten(mesh.xyGrid[0]),
            torch::flatten(mesh.xyGrid[1]),
            torch::full_like(torch::flatten(mesh.xyGrid[1]),mesh.ubT_) //time values 
          },1
        );
    
        //- transfer grid to device
        grid.to(mesh.device_);  

        //- get predicted output, and from that get phaseField 
        torch::Tensor C1 = mesh.net_->forward(grid);
        std::string gridName = "gridSave" + std::to_string(mesh.ubT_);
        std::string fieldsName = "fieldsSave" + std::to_string(mesh.ubT_);
        //- write out input data for python to plot
        writeTensorToFile(grid,gridName);
        writeTensorToFile(C1,fieldsName); 
   
      }
      //- stop training if target loss achieved
      if (loss < mesh.net_->ABS_TOL) 
      {
        std::string modelName = "pNet" + std::to_string(mesh.ubT_) + ".pt"; // ".pt" is the extension of for pyTorch module
        //- save model to file for post processing, indicate saved model is due to convergence
        torch::save(mesh.net_,modelName);
        //- update iter to get correct iter count
        iter += 1;
        //- update mesh for nex Time step in the time marching loop
        break;
      }
      iter += 1;
    }
    //- end profile clock
    auto end_time = std::chrono::high_resolution_clock::now();
    
    //- get runTime for traning
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
   
    //- info out runTime
    std::cout << "Epoch execution time: " << duration.count() << " microseconds" << std::endl;
    
    //- Grid  for plotting final timeStep
    torch::Tensor grid = torch::stack
    (
      {
        torch::flatten(mesh.xyGrid[0]),
        torch::flatten(mesh.xyGrid[1]),
        torch::full_like(torch::flatten(mesh.xyGrid[1]),0.5) //time values 
        //torch::flatten(mesh.mesh_[2])
      },1
    );
    
    //- get predicted output, and from that get phaseField 
    torch::Tensor C1 = mesh.net_->forward(grid);
    
    //- file Names for the
    std::string gridName = "grid" + std::to_string(mesh.ubT_);
    std::string fieldsName = "fields" + std::to_string(mesh.ubT_);

    //- write out input data for python to plot
    writeTensorToFile(grid,gridName);
    writeTensorToFile(C1,fieldsName); 
    mesh.updateMesh();
  }
  return 0;
} 
