#include "nn_main.h"
#include "utils.h"
#include <ATen/TensorIndexing.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/all.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/flatten.h>
#include <ATen/ops/full_like.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros_like.h>
#include <c10/core/DispatchKeySet.h>
#include <cassert>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/linear.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>
#include <vector>
#include <cmath>
#include <torch/script.h>
//- loads in python like indaexing of tensors
using namespace torch::indexing; 

//-------------------PINN definitions----------------------------------------//

//- function to create layers present in the net
void PinNetImpl::create_layers()
{
  //- register input layer 
  input = register_module
  (
    "fc_input",
    torch::nn::Linear(INPUT_DIM,HIDDEN_LAYER_DIM)
  );
  //- register and  hidden layers 
  for(int i=0;i<N_HIDDEN_LAYERS;i++)
  {
    //- hiden layer name
    std::string layer_name = "fc_hidden" + std::to_string(i);
    //- register each hidden layer
    torch::nn::Linear linear_layer = register_module
    (
      layer_name,
      torch::nn::Linear(HIDDEN_LAYER_DIM,HIDDEN_LAYER_DIM)
    );
    //- intialize network parameters
    torch::nn::init::xavier_normal_(linear_layer->weight);
    //- populate sequential with layers
    hidden_layers->push_back(linear_layer);
    //- register activation functions 
    hidden_layers->push_back
    (
      register_module
      (
        "fc_silu_hidden" + std::to_string(i), 
        torch::nn::SiLU() // swish function X * RELU(X)
      )
    );
  }
  //- register output layer
  output = register_module
  (
    "fc_output",
    torch::nn::Linear(HIDDEN_LAYER_DIM,OUTPUT_DIM)
  );
}

//- constructor for PinNet module implementation
PinNetImpl::PinNetImpl
(
  const Dictionary &netDict // reference to Dictionary object
)
: 
  dict(netDict), //pass in Dictionary  
  INPUT_DIM(dict.get<int>("inputDim")), // no. of input features  
  HIDDEN_LAYER_DIM(dict.get<int>("hiddenLayerDim")), // no. of neurons in HL
  N_HIDDEN_LAYERS(dict.get<int>("nHiddenLayer")), // no. of hidden layers
  OUTPUT_DIM(dict.get<int>("outputDim")) //- no. of output features
{
  //- set parameters from Dictionary lookup
  N_EQN = dict.get<int>("NEQN");
  N_BC = dict.get<int>("NBC");
  N_IC = dict.get<int>("NIC");
  //- flag for transient or steady state mode
  transient_ = dict.get<int>("transient");
  //- get target loss from dict
  ABS_TOL = dict.get<float>("ABSTOL");
  K_EPOCH = dict.get<int>("KEPOCH");
  //- batch size for pde loss input
  BATCHSIZE=dict.get<int>("BATCHSIZE");
  //- number of iterations in one epoch 
  NITER_ = N_EQN/BATCHSIZE;
  //- create and intialize the layers in the net
  create_layers();
}

//- forward propagation 
torch::Tensor PinNetImpl::forward
(
 const torch::Tensor& X
)
{
  torch::Tensor I = torch::silu(input(X));
  I = hidden_layers->forward(I);
  I = output(I);
  return I;
}
//------------------end PINN definitions-------------------------------------//


//-----------------derivative definitions------------------------------------//

//- first order derivative
torch::Tensor d_d1
(
  const torch::Tensor &I,
  const torch::Tensor &X,
  int spatialIndex
)
{
  torch::Tensor derivative = torch::autograd::grad 
  (
    {I},
    {X},
    {torch::ones_like(I)},
    true,
    true,
    true
  )[0].requires_grad_(true);
  return derivative.index({Slice(),spatialIndex});
}

torch::Tensor d_d1
(
  const torch::Tensor &I,
  const torch::Tensor &X
)
{
  torch::Tensor derivative = torch::autograd::grad 
  (
    {I},
    {X},
    {torch::ones_like(I)},
    true,
    true,
    true
  )[0].requires_grad_(true);
  return derivative;
}

//- higher order derivative
torch::Tensor d_dn
(
  const torch::Tensor &I, 
  const torch::Tensor &X, 
  int order, // order of derivative
  int spatialIndex
)
{
  torch::Tensor derivative =  d_d1(I,X,spatialIndex);
  for(int i=0;i<order-1;i++)
  {
    derivative = d_d1(derivative,X,spatialIndex);
  }
  return derivative;
}

//- function overload when X is 1D tensor
torch::Tensor d_dn
(
  const torch::Tensor &I, 
  const torch::Tensor &X, 
  int order // order of derivative
)
{
  torch::Tensor derivative =  d_d1(I,X);
  for(int i=0;i<order-1;i++)
  {
    derivative = d_d1(derivative,X);
  }
  return derivative;
}

//----------------------------end derivative definitions---------------------//


//----------------------------CahnHillard function definitions---------------//


//- TODO make all derivatives members of mesh class and calculate them onlyn once
//  per iteration, might reduce runTime 

//- thermoPhysical properties for mixture
torch::Tensor CahnHillard::thermoProp
(
  float propLiquid, //thermoPhysical prop of liquid  phase 
  float propGas, // thermoPhysical prop of gas phase
  const torch::Tensor &I
)
{
  //- get auxillary phase field var to correct for bounds 
  const torch::Tensor C = CahnHillard::Cbar(I.index({Slice(),3}));
  torch::Tensor mixtureProp = 
    0.5*(1+C)*propLiquid + 0.5*(1-C)*propGas;
  return mixtureProp;
}

//- continuity loss 
torch::Tensor CahnHillard::L_Mass2D
(
  const mesh2D &mesh 
)
{
  const torch::Tensor &du_dx = mesh.internalMesh_.gradU_.index({Slice(),0});
  const torch::Tensor &dv_dy = mesh.internalMesh_.gradV_.index({Slice(),0});
  torch::Tensor loss = du_dx + dv_dy;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- returns the phi term needed 
torch::Tensor CahnHillard::phi
(
  const mesh2D &mesh
)
{
  float &e = mesh.thermo_.epsilon;
  const torch::Tensor &C = mesh.internalMesh_.output_.index({Slice(),3});
  const torch::Tensor &Cxx = mesh.internalMesh_.d2C_dxx_;
  const torch::Tensor &Cyy = mesh.internalMesh_.d2C_dyy_;
  return C*(C*C-1) - e*e*(Cxx + Cyy); 
}

//- returns CahnHillard Loss
torch::Tensor CahnHillard::CahnHillard2D
(
  const mesh2D &mesh
)
{
  const float &e = mesh.thermo_.epsilon;
  const float &Mo = mesh.thermo_.Mo;
  //- derivatives 
  const torch::Tensor &u = mesh.internalMesh_.output_.index({Slice(),0});
  const torch::Tensor &v = mesh.internalMesh_.output_.index({Slice(),0});
  torch::Tensor dC_dt = mesh.internalMesh_.gradC_.index({Slice(),2});
  torch::Tensor dC_dx = mesh.internalMesh_.gradC_.index({Slice(),0});
  torch::Tensor dC_dy = mesh.internalMesh_.gradC_.index({Slice(),1});
  torch::Tensor phi = CahnHillard::phi(mesh);
  torch::Tensor dphi_dxx = d_dn(phi,mesh.internalMesh_.input_,2,0);
  torch::Tensor dphi_dyy = d_dn(phi,mesh.internalMesh_.input_,2,1);
  //- loss term
  torch::Tensor loss = dC_dt + u*dC_dx + v*dC_dy - 
    Mo*(dphi_dxx + dphi_dyy);
  return torch::mse_loss(loss,torch::zeros_like(loss));
}

//- returns the surface tension tensor needed in mom equation
torch::Tensor CahnHillard::surfaceTension
(
  const mesh2D &mesh,
  int dim
)
{
  const float &sigma = mesh.thermo_.sigma0;
  const float &e_inv = 1.0/mesh.thermo_.epsilon;
  const torch::Tensor &C = mesh.internalMesh_.output_.index({Slice(),3});
  torch::Tensor surf = e_inv*sigma*mesh.thermo_.C*CahnHillard::phi(mesh)
    *d_d1(C,mesh.internalMesh_.input_,dim);
  return surf;
} 

//- momentum loss for x direction in 2D 
torch::Tensor CahnHillard::L_MomX2d
(
  const mesh2D &mesh
)
{ 
  float &rhoL = mesh.thermo_.rhoL;
  float &muL = mesh.thermo_.muL;
  float rhoG = mesh.thermo_.rhoG;
  float muG = mesh.thermo_.muG;
  const torch::Tensor &u = mesh.internalMesh_.output_.index({Slice(),0});
  const torch::Tensor &v = mesh.internalMesh_.output_.index({Slice(),1});
  const torch::Tensor &p = mesh.internalMesh_.output_.index({Slice(),2});
  const torch::Tensor &C = mesh.internalMesh_.output_.index({Slice(),3});
  //- get density of mixture TODO correct this function to take in just mesh
  torch::Tensor rhoM = CahnHillard::thermoProp(rhoL, rhoG, mesh.internalMesh_.output_);
  //- get viscosity of mixture
  torch::Tensor muM = CahnHillard::thermoProp(muL, muG, mesh.internalMesh_.output_);
  const torch::Tensor &du_dt = mesh.internalMesh_.gradU_.index({Slice(),2});
  const torch::Tensor &du_dx = mesh.internalMesh_.gradU_.index({Slice(),0});
  const torch::Tensor &du_dy = mesh.internalMesh_.gradU_.index({Slice(),1});
  const torch::Tensor &dv_dx = mesh.internalMesh_.gradV_.index({Slice(),0});
  const torch::Tensor &dC_dx = mesh.internalMesh_.gradC_.index({Slice(),0});
  const torch::Tensor &dC_dy = mesh.internalMesh_.gradC_.index({Slice(),1});
  const torch::Tensor &dp_dx = mesh.internalMesh_.gradP_.index({Slice(),0});
  //- derivative order first spatial variable later
  const torch::Tensor &du_dxx = mesh.internalMesh_.d2u_dxx_;
  const torch::Tensor &du_dyy = mesh.internalMesh_.d2u_dyy_;
  //- get x component of the surface tension force
  torch::Tensor fx = CahnHillard::surfaceTension(mesh,0);
  torch::Tensor loss1 = rhoM*(du_dt + u*du_dx + v*du_dy) + dp_dx;
  torch::Tensor loss2 = -0.5*(muL - muG)*dC_dy*(du_dy + dv_dx) - (muL -muG)*dC_dx*du_dx;
  torch::Tensor loss3 = -muM*(du_dxx + du_dyy) - fx;
  //- division by rhoL for normalization?
  torch::Tensor loss = (loss1 + loss2 + loss3)/rhoL;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- momentum loss for y direction in 2D
torch::Tensor CahnHillard::L_MomY2d
(
  const mesh2D &mesh
)
{
 
  float &rhoL = mesh.thermo_.rhoL;
  float &muL = mesh.thermo_.muL;
  float rhoG = mesh.thermo_.rhoG;
  float muG = mesh.thermo_.muG;
  const torch::Tensor &u = mesh.internalMesh_.output_.index({Slice(),0});
  const torch::Tensor &v = mesh.internalMesh_.output_.index({Slice(),1});
  const torch::Tensor &p = mesh.internalMesh_.output_.index({Slice(),2});
  const torch::Tensor &C = mesh.internalMesh_.output_.index({Slice(),3});
  //- get density of mixture TODO correct this function to take in just mesh
  torch::Tensor rhoM = CahnHillard::thermoProp(rhoL, rhoG, mesh.internalMesh_.output_);
  //- get viscosity of mixture
  torch::Tensor muM = CahnHillard::thermoProp(muL, muG, mesh.internalMesh_.output_);
  const torch::Tensor &dv_dt = mesh.internalMesh_.gradV_.index({Slice(),2});
  const torch::Tensor &du_dx = mesh.internalMesh_.gradU_.index({Slice(),0});
  const torch::Tensor &dv_dy = mesh.internalMesh_.gradV_.index({Slice(),1});
  const torch::Tensor &dv_dx = mesh.internalMesh_.gradV_.index({Slice(),0});
  const torch::Tensor &dC_dx = mesh.internalMesh_.gradC_.index({Slice(),0});
  const torch::Tensor &dC_dy = mesh.internalMesh_.gradC_.index({Slice(),1});
  const torch::Tensor &dp_dy = mesh.internalMesh_.gradP_.index({Slice(),1});
  //- derivative order first spatial variable later
  const torch::Tensor &du_dxx = mesh.internalMesh_.d2u_dxx_;
  const torch::Tensor &du_dyy = mesh.internalMesh_.d2u_dyy_;  //- derivative order first spatial variable later
  torch::Tensor dv_dxx = mesh.internalMesh_.d2v_dxx_;
  torch::Tensor dv_dyy = mesh.internalMesh_.d2v_dyy_;
  //- get x component of the surface tension force
  torch::Tensor fy = CahnHillard::surfaceTension(mesh,1);
  torch::Tensor gy = torch::full_like(fy,0.98);
  torch::Tensor loss1 = rhoM*(dv_dt + u*dv_dx + v*dv_dy) + dp_dy;
  torch::Tensor loss2 = -0.5*(muL - muG)*dC_dx*(du_dx + dv_dy) - (muL -muG)*dC_dy*dv_dy;
  torch::Tensor loss3 = -muM*(dv_dxx + dv_dyy) - fy - rhoM*gy;
  torch::Tensor loss = (loss1 + loss2 + loss3)/rhoL;
  return torch::mse_loss(loss, torch::zeros_like(loss));
}

//- get total PDE loss
torch::Tensor CahnHillard::PDEloss(mesh2D &mesh)
{
  //- loss from mass conservation
  torch::Tensor LM = CahnHillard::L_Mass2D(mesh);
  torch::Tensor LMX = CahnHillard::L_MomX2d(mesh);
  torch::Tensor LMY = CahnHillard::L_MomY2d(mesh);
  torch::Tensor LC = CahnHillard::CahnHillard2D(mesh);
  //- return total pde loss
  return LM + LC + LMX + LMY;
}

//- TODO make the function more general by adding in another int for u or v
torch::Tensor CahnHillard::slipWall(torch::Tensor &I, torch::Tensor &X,int dim)
{
  const torch::Tensor &u = I.index({Slice(),0});  
  const torch::Tensor &v = I.index({Slice(),1});
  torch::Tensor dv_dx = d_d1(v,X,dim);
  return torch::mse_loss(dv_dx,torch::zeros_like(dv_dx))
    + torch::mse_loss(u,torch::zeros_like(u));
}

torch::Tensor CahnHillard::noSlipWall(torch::Tensor &I, torch::Tensor &X)
{
  const torch::Tensor &u = I.index({Slice(),0});
  const torch::Tensor &v = I.index({Slice(),1});
  return torch::mse_loss(u,torch::zeros_like(u))  + 
    torch::mse_loss(v,torch::zeros_like(v));
  
}
//- get boundary loss
torch::Tensor CahnHillard::BCloss(mesh2D &mesh)
{
  
  //- get phase field vars at all the boundaries
  torch::Tensor Cleft = mesh.leftWall_.output_.index({Slice(),3});
  torch::Tensor Cright = mesh.rightWall_.output_.index({Slice(),3});
  torch::Tensor Ctop = mesh.topWall_.output_.index({Slice(),3});
  torch::Tensor Cbottom = mesh.bottomWall_.output_.index({Slice(),3});
  
  //- total boundary loss for u, v and C
  torch::Tensor lossLeft = CahnHillard::slipWall(mesh.leftWall_.output_, mesh.leftWall_.input_,0); 
       //+ CahnHillard::zeroGrad(Cleft, mesh.iLeftWall_, 0);
  torch::Tensor lossRight = CahnHillard::slipWall(mesh.rightWall_.output_,mesh.rightWall_.input_, 0);
       //+ CahnHillard::zeroGrad(Cright, mesh.iRightWall_, 0);
  torch::Tensor lossTop = CahnHillard::noSlipWall(mesh.topWall_.output_, mesh.topWall_.input_);
       //+ CahnHillard::zeroGrad(Ctop, mesh.iTopWall_, 1);
  torch::Tensor lossBottom = CahnHillard::noSlipWall(mesh.bottomWall_.output_, mesh.bottomWall_.input_);
       //+ CahnHillard::zeroGrad(Cbottom, mesh.iBottomWall_, 1);
  return lossLeft + lossRight + lossTop + lossBottom;
}

//- get the intial loss for the 
torch::Tensor CahnHillard::ICloss(mesh2D &mesh)
{
  //- x vel prediction in current iteration
  const torch::Tensor &u = mesh.initialMesh_.output_.index({Slice(),0});
  //- y vel prediction in current iteration
  const torch::Tensor &v = mesh.initialMesh_.output_.index({Slice(),1});
  //- phaseField variable prediction in current iteration
  const torch::Tensor &C = mesh.initialMesh_.output_.index({Slice(),3});
  //- get all the intial losses
  torch::Tensor uLoss = torch::mse_loss(u,CahnHillard::u_at_InitialTime(mesh));
  torch::Tensor vLoss = torch::mse_loss(v,CahnHillard::v_at_InitialTime(mesh));
  torch::Tensor CLoss = torch::mse_loss(C,CahnHillard::C_at_InitialTime(mesh));
  //- return total loss
  return uLoss +vLoss +CLoss;
}

//- total loss function for the optimizer
torch::Tensor CahnHillard::loss(mesh2D &mesh)
{
  // torch::Tensor pdeloss = CahnHillard::PDEloss(mesh);
  torch::Tensor bcLoss = CahnHillard::BCloss(mesh);
  torch::Tensor pdeLoss = CahnHillard::PDEloss(mesh);
  torch::Tensor icLoss = CahnHillard::ICloss(mesh);
  return bcLoss + pdeLoss + icLoss;
}


//- TODO make radius a variable 
torch::Tensor CahnHillard::C_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ == 0)
  {
    const float &xc = mesh.xc;
    const float &yc = mesh.yc;
    const float &e = mesh.thermo_.epsilon;
    //- x 
    const torch::Tensor &x = mesh.initialMesh_.input_.index({Slice(),0});
    //- y
    const torch::Tensor &y = mesh.initialMesh_.input_.index({Slice(),1});
    //- intial condition
    torch::Tensor Ci =torch::tanh((torch::sqrt(torch::pow(x - xc, 2) + torch::pow(y - yc, 2)) - 0.15)/ (1.41421356237 * e));
    
    return Ci;
  }
  else  
  {
    //- use previous converged neural net as intial conditions
    torch::Tensor Ci = mesh.netPrev_->forward(mesh.initialMesh_.input_).index({Slice(),3});
    return Ci;
  }
}
//- intial velocity fields for u and v
torch::Tensor CahnHillard::u_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ ==0)
  {
    return torch::zeros_like(mesh.initialMesh_.input_.index({Slice(),0}));
    
  }
  else
  {
    return mesh.netPrev_->forward(mesh.initialMesh_.input_).index({Slice(),0});
  }
}
//-v at intial time
torch::Tensor CahnHillard::v_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ ==0)
  {
    return torch::zeros_like(mesh.initialMesh_.input_.index({Slice(),0}));
  }
  else
  {
    return mesh.netPrev_->forward(mesh.initialMesh_.input_).index({Slice(),1});
  }
}


//- auxiliary variable to bound thermophysical properties 
torch::Tensor CahnHillard::Cbar(const torch::Tensor &C)
{
  //- get the absolute value of the phasefield tensor
  torch::Tensor absC = torch::abs(C);
  if(torch::all(absC <=1).item<float>())
  {
    return C;
  }
  else {
    return torch::sign(C);
  }
}

//- zero Grad function for phaseField boundary condtion
torch::Tensor CahnHillard::zeroGrad(torch::Tensor &I, torch::Tensor &X, int dim)
{ 
  torch::Tensor grad = d_d1(I,X,dim);
  return torch::mse_loss(grad, torch::zeros_like(grad));
}
//---------------------end CahnHillard function definitions------------------//

//---------------------------mesh2d function definitions---------------------//

//- construct computational domain for the PINN instance
mesh2D::mesh2D
(
  Dictionary &meshDict, //mesh parameters
  PinNet &net,
  PinNet &netPrev,
  torch::Device &device, // device info
  thermoPhysical &thermo
):
  net_(net), // pass in current neural net
  netPrev_(netPrev), // pass in other neural net
  dict(meshDict),
  device_(device), // pass in device info
  thermo_(thermo), // pass in thermo class instance
  lbX_(dict.get<float>("lbX")), // read in mesh props from dict
  ubX_(dict.get<float>("ubX")),
  lbY_(dict.get<float>("lbY")),
  ubY_(dict.get<float>("ubY")),
  lbT_(dict.get<float>("lbT")),
  ubT_(dict.get<float>("ubT")),
  deltaX_(dict.get<float>("dx")),
  deltaY_(dict.get<float>("dy")),
  deltaT_(dict.get<float>("dt")),
  xc(dict.get<float>("xc")),
  yc(dict.get<float>("yc"))

{
  TimeStep_ = dict.get<float>("stepSize");
  Nx_ = (ubX_ - lbX_)/deltaX_ + 1;
  Ny_ = (ubY_ - lbY_)/deltaY_ + 1;
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;
  Ntotal_ = Nx_*Ny_*Nt_;
  //- populate the individual 1D grids
  xGrid = torch::linspace(lbX_, ubX_, Nx_,device_);
  yGrid = torch::linspace(lbY_, ubY_, Ny_,device_);
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  internalMesh_.grid_ = torch::meshgrid({xGrid,yGrid,tGrid});
  //- create boundary grids
  createBC();
}

//- operator overload () to acess main computational domain
torch::Tensor  mesh2D::operator()(int i, int j, int k)  
{
  return torch::stack
  (
    {
      internalMesh_.grid_[0].index({i, j, k}), 
      internalMesh_.grid_[1].index({i, j, k}), 
      internalMesh_.grid_[2].index({i, j, k})
    }
  ); 
}

//- create boundary grids
void mesh2D::createBC()
{
  //temp tensors
  torch::Tensor xLeft = torch::tensor(lbX_,device_);
  torch::Tensor xRight = torch::tensor(ubX_,device_);
  torch::Tensor yBottom = torch::tensor(lbY_, device_);
  torch::Tensor yTop = torch::tensor(ubY_, device_);
  torch::Tensor tInitial = torch::tensor(lbT_,device_);
  leftWall_.grid_ = torch::meshgrid({xLeft,yGrid,tGrid});
  rightWall_.grid_ = torch::meshgrid({xRight,yGrid,tGrid});
  topWall_.grid_= torch::meshgrid({xGrid,yTop,tGrid});
  bottomWall_.grid_ = torch::meshgrid({xGrid,yBottom,tGrid});
  initialMesh_.grid_ = torch::meshgrid({xGrid,yGrid,tInitial});
}

//- general method to create samples
//- used to create boundary as well as intial state samples
void mesh2D::createSamples
(
  std::vector<torch::Tensor> &grid, 
  torch::Tensor &samples,
  int nSamples
) 
{
  std::vector<torch::Tensor> vectorStack;
  int ntotal = grid[0].numel();
  torch::Tensor indices = torch::randperm
  (ntotal,device_).slice(0,0,nSamples);
  for(int i=0;i<grid.size();i++)
  {
    vectorStack.push_back
    (
      torch::flatten
      (
        grid[i]
      ).index_select(0,indices)
    );
  }
  samples = torch::stack(vectorStack,1);
  samples.set_requires_grad(true);
}

//- create the total samples required for neural net
//- these samples are the input features to the neural net forward passes
void mesh2D::createTotalSamples
(
  int iter // current iter when looping through the batches
) 
{
  //- generate random indices to generate random samples from grids
  if(iter == 0)
  { 
    //- create Indices in the first iteration itself
    createIndices();
  }
  
  torch::Tensor batchIndices = internalMesh_.indices_.slice
  (
    0,
    iter*net_->BATCHSIZE,
    (iter + 1)*net_->BATCHSIZE,
    1 // step size when slicing
  );
  createSamples(internalMesh_.grid_,internalMesh_.input_,batchIndices);
  if(iter ==0)
  {
    //- update samples 
    createSamples(leftWall_.grid_,leftWall_.input_,net_->N_BC);
    createSamples(rightWall_.grid_, rightWall_.input_,net_->N_BC);
    createSamples(topWall_.grid_,topWall_.input_,net_->N_BC);
    createSamples(bottomWall_.grid_,bottomWall_.input_,net_->N_BC);
    createSamples(initialMesh_.grid_,initialMesh_.input_,net_->N_IC);
  }
}

//- forward pass of current batch in batch iteration loop
//- update output features for each batch iteration,
//- pass in the iteration 
// (extremely SHITTY method, but cannot think of a 
//  better one as of now)
void mesh2D::update(int iter)
{ 
  createTotalSamples(iter);
  internalMesh_.output_ = net_->forward(internalMesh_.input_);
  initialMesh_.output_ = net_->forward(initialMesh_.input_);
  leftWall_.output_ = net_->forward(leftWall_.input_);
  rightWall_.output_ = net_->forward(rightWall_.input_);
  bottomWall_.output_ = net_->forward(bottomWall_.input_);
  topWall_.output_ = net_->forward(topWall_.input_);
  computeGradPDE(internalMesh_);
}

//- creates indices tensor for internal mesh, can add indices for other boundary as well
void mesh2D::createIndices()
{
  internalMesh_.indices_ = 
  torch::randperm
  (
    internalMesh_.grid_[0].numel(),
    device_
  ).slice(0,0,net_->N_EQN,1);
}

//- createSamples over load to create samples for pde loss as it will buffer
//- passed in batches instead of one go, the other samples being way smaller
//- in size remain unchanged
void mesh2D::createSamples
(
 std::vector<torch::Tensor> &grid,
 torch::Tensor &samples,
 torch::Tensor &indices
)
{
  //- vectors to stack
  std::vector<torch::Tensor> vectorStack;
  //- push vectors to vectors stack
  for(int i=0;i<grid.size();i++)
  {
    vectorStack.push_back
    (
      torch::flatten
      (
        grid[i]
      ).index_select(0,indices)
    );
  }
  //- pass stack to get samples
  samples = torch::stack(vectorStack,1);
  //- set gradient =true
  samples.set_requires_grad(true);
}

void mesh2D::updateMesh()
{
  //- update the lower level of time grid
  lbT_ = lbT_ + TimeStep_;
  //- get new number of time steps in the current time domain
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;
  //- update tGrid
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  //- update main mesh
  internalMesh_.grid_ = torch::meshgrid({xGrid,yGrid,tGrid});
  //- update the boundary grids
  createBC();
  //- transfer over parameters of current converged net to 
  //- previous net reference to use as intial condition for 
  //- intial losses
  loadState(net_, netPrev_);
}

void mesh2D::computeGradPDE(feature &feat)
{
  const torch::Tensor &u = internalMesh_.output_.index({Slice(),0});
  const torch::Tensor &v = internalMesh_.output_.index({Slice(),1});
  const torch::Tensor &p = internalMesh_.output_.index({Slice(),2});
  const torch::Tensor &C = internalMesh_.output_.index({Slice(),3});
  feat.gradU_ = d_d1(u,internalMesh_.input_);
  feat.gradV_ = d_d1(v,internalMesh_.input_);
  feat.gradP_ = d_d1(p,internalMesh_.input_);
  feat.gradC_ = d_d1(C,internalMesh_.input_);
  feat.d2C_dxx_ = d_dn(C,internalMesh_.input_,2,0);
  feat.d2C_dyy_ = d_dn(C,internalMesh_.input_,2,1);  
  feat.d2u_dxx_ = d_dn(u,internalMesh_.input_,2,0);
  feat.d2u_dyy_ = d_dn(u,internalMesh_.input_,2,1);
  feat.d2v_dxx_ = d_dn(v,internalMesh_.input_,2,0);
  feat.d2v_dyy_ = d_dn(v,internalMesh_.input_,2,1);
}


//-------------------------end mesh2D definitions----------------------------//

//---thermophysical class definition
thermoPhysical::thermoPhysical(Dictionary &dict)
{
  Mo = dict.get<float>("Mo");
  epsilon = dict.get<float>("epsilon");
  sigma0 = dict.get<float>("sigma0");
  muL = dict.get<float>("muL");
  muG = dict.get<float>("muG");
  rhoL = dict.get<float>("rhoL");
  rhoG = dict.get<float>("rhoG");
  C = 1.06066017178;
}

void loadState(PinNet& net1, PinNet &net2)
{
  torch::autograd::GradMode::set_enabled(false);
  auto new_params = net2->named_parameters();
  auto params = net1->named_parameters(true);
  auto buffer = net1->named_buffers(true);
  for(auto &val : new_params)
  {
    auto name = val.key();
    auto *t = params.find(name);
    if(t!=nullptr)
    {
      t->copy_(val.value());
    }
    else
    {
      t= buffer.find(name);
      if (t !=nullptr)
      {
        t->copy_(val.value());
      }
    }
  }
  torch::autograd::GradMode::set_enabled(true);
} 












