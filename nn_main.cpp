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
    //- why ????
    /*
    hidden_layers->push_back
    (
      register_module
      (
        "batch norm",
        torch::nn::BatchNorm1d()
      )
    );
    */
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
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor dv_dy = d_d1(v,mesh.iPDE_,1);
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
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  torch::Tensor Cxx = d_dn(C,mesh.iPDE_,2,0);
  torch::Tensor Cyy = d_dn(C,mesh.iPDE_,2,1);
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
  //- u vel
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  //- v vel
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  //- phase field var
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- derivatives 
  torch::Tensor dC_dt = d_d1(C,mesh.iPDE_,2);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor phi = CahnHillard::phi(mesh);
  torch::Tensor dphi_dxx = d_dn(phi,mesh.iPDE_,2,0);
  torch::Tensor dphi_dyy = d_dn(phi,mesh.iPDE_,2,1);
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
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  torch::Tensor surf = e_inv*sigma*mesh.thermo_.C*CahnHillard::phi(mesh)
    *d_d1(C,mesh.iPDE_,dim);
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
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  const torch::Tensor &p = mesh.fieldsPDE_.index({Slice(),2});
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- get density of mixture TODO correct this function to take in just mesh
  torch::Tensor rhoM = CahnHillard::thermoProp(rhoL, rhoG, mesh.fieldsPDE_);
  //- get viscosity of mixture
  torch::Tensor muM = CahnHillard::thermoProp(muL, muG, mesh.fieldsPDE_);
  torch::Tensor du_dt = d_d1(u,mesh.iPDE_,2);
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor du_dy = d_d1(u,mesh.iPDE_,1);
  torch::Tensor dv_dx = d_d1(v,mesh.iPDE_,0);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor dp_dx = d_d1(p,mesh.iPDE_,0);
  //- derivative order first spatial variable later
  torch::Tensor du_dxx = d_dn(u,mesh.iPDE_,2,0);
  torch::Tensor du_dyy = d_dn(u,mesh.iPDE_,2,1);
  //- get x component of the surface tension force
  torch::Tensor fx = CahnHillard::surfaceTension(mesh,0);
  torch::Tensor loss1 = rhoM*(du_dt + u*du_dx + v*du_dy) + dp_dx;
  torch::Tensor loss2 = -0.5*(muL - muG)*dC_dy*(du_dy + dv_dx) - (muL -muG)*dC_dx*du_dx;
  torch::Tensor loss3 = -muM*(du_dxx + du_dyy) - fx;
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
  const torch::Tensor &u = mesh.fieldsPDE_.index({Slice(),0});
  const torch::Tensor &v = mesh.fieldsPDE_.index({Slice(),1});
  const torch::Tensor &p = mesh.fieldsPDE_.index({Slice(),2});
  const torch::Tensor &C = mesh.fieldsPDE_.index({Slice(),3});
  //- get density of mixture TODO correct this function to take in just mesh
  torch::Tensor rhoM = CahnHillard::thermoProp(rhoL, rhoG, mesh.fieldsPDE_);
  //- get viscosity of mixture
  torch::Tensor muM = CahnHillard::thermoProp(muL, muG, mesh.fieldsPDE_);
  torch::Tensor dv_dt = d_d1(v,mesh.iPDE_,2);
  torch::Tensor dv_dx = d_d1(v,mesh.iPDE_,0);
  torch::Tensor dv_dy = d_d1(v,mesh.iPDE_,1);
  torch::Tensor du_dx = d_d1(u,mesh.iPDE_,0);
  torch::Tensor dC_dx = d_d1(C,mesh.iPDE_,0);
  torch::Tensor dC_dy = d_d1(C,mesh.iPDE_,1);
  torch::Tensor dp_dy = d_d1(p,mesh.iPDE_,1);
  //- derivative order first spatial variable later
  torch::Tensor dv_dxx = d_dn(v,mesh.iPDE_,2,0);
  torch::Tensor dv_dyy = d_dn(v,mesh.iPDE_,2,1);
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
  torch::Tensor Cleft = mesh.fieldsLeft_.index({Slice(),3});
  torch::Tensor Cright = mesh.fieldsRight_.index({Slice(),3});
  torch::Tensor Ctop = mesh.fieldsTop_.index({Slice(),3});
  torch::Tensor Cbottom = mesh.fieldsBottom_.index({Slice(),3});
  
  //- total boundary loss for u, v and C
  torch::Tensor lossLeft = CahnHillard::slipWall(mesh.fieldsLeft_, mesh.iLeftWall_,0); 
       //+ CahnHillard::zeroGrad(Cleft, mesh.iLeftWall_, 0);
  torch::Tensor lossRight = CahnHillard::slipWall(mesh.fieldsRight_,mesh.iRightWall_, 0);
       //+ CahnHillard::zeroGrad(Cright, mesh.iRightWall_, 0);
  torch::Tensor lossTop = CahnHillard::noSlipWall(mesh.fieldsTop_, mesh.iTopWall_);
       //+ CahnHillard::zeroGrad(Ctop, mesh.iTopWall_, 1);
  torch::Tensor lossBottom = CahnHillard::noSlipWall(mesh.fieldsBottom_, mesh.iBottomWall_);
       //+ CahnHillard::zeroGrad(Cbottom, mesh.iBottomWall_, 1);
  return lossLeft + lossRight + lossTop + lossBottom;
}

//- get the intial loss for the 
torch::Tensor CahnHillard::ICloss(mesh2D &mesh)
{
  //- x vel prediction in current iteration
  const torch::Tensor &u = mesh.fieldsIC_.index({Slice(),0});
  //- y vel prediction in current iteration
  const torch::Tensor &v = mesh.fieldsIC_.index({Slice(),1});
  //- phaseField variable prediction in current iteration
  const torch::Tensor &C = mesh.fieldsIC_.index({Slice(),2});
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
    const torch::Tensor &x = mesh.iIC_.index({Slice(),0});
    //- y
    const torch::Tensor &y = mesh.iIC_.index({Slice(),1});
    //- intial condition
    torch::Tensor Ci =torch::tanh((torch::sqrt(torch::pow(x - xc, 2) + torch::pow(y - yc, 2)) - 0.15)/ (1.41421356237 * e));
    
    return Ci;
  }
  else 
  {
    torch::Tensor Ci = mesh.netPrev_->forward(mesh.iIC_).index({Slice(),3});
    return Ci;
  }
}
//- intial velocity fields for u and v
torch::Tensor CahnHillard::u_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ ==0)
  {
    return torch::zeros_like(mesh.iIC_.index({Slice(),0}));
    
  }
  else
  {
    return mesh.netPrev_->forward(mesh.iIC_).index({Slice(),0});
  }
}
//-v
torch::Tensor CahnHillard::v_at_InitialTime(mesh2D &mesh)
{
  if(mesh.lbT_ ==0)
  {
    return torch::zeros_like(mesh.iIC_.index({Slice(),0}));
  }
  else
  {
    return mesh.netPrev_->forward(mesh.iIC_).index({Slice(),1});
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

torch::Tensor Heat::L_Diffusion2D
(
  mesh2D &mesh
)
{
  float PI = 3.14159265358979323846;
  torch::Tensor u_xx = d_dn(mesh.fieldsPDE_,mesh.iPDE_,2,0);
  torch::Tensor u_yy = d_dn(mesh.fieldsPDE_,mesh.iPDE_,2,1);
  torch::Tensor fTerm =
    -2*sin(PI*mesh.iPDE_.index({Slice(),0}))*sin(PI*mesh.iPDE_.index({Slice(),1}))*PI*PI;
  return torch::mse_loss(u_xx+u_yy,fTerm);

}

//- total loss for 2d diffusion equation
torch::Tensor Heat::loss(mesh2D &mesh)
{
  //- create samples
  torch::Tensor pdeloss = Heat::L_Diffusion2D(mesh);
  torch::Tensor l = torch::mse_loss(mesh.fieldsLeft_,torch::zeros_like(mesh.fieldsLeft_));
  torch::Tensor r = torch::mse_loss(mesh.fieldsRight_,torch::zeros_like(mesh.fieldsRight_));
  torch::Tensor t = torch::mse_loss(mesh.fieldsTop_,torch::zeros_like(mesh.fieldsTop_));
  torch::Tensor b = torch::mse_loss(mesh.fieldsBottom_,torch::zeros_like(mesh.fieldsBottom_));
  return l+b+r+t+pdeloss;
  
}
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
    //- get number of ponits from bounds and step size
  Nx_ = (ubX_ - lbX_)/deltaX_ + 1;
  Ny_ = (ubY_ - lbY_)/deltaY_ + 1;
  Nt_ = (ubT_ - lbT_)/deltaT_ + 1;

  //- total number of points in the entire domain (nDOF)
  Ntotal_ = Nx_*Ny_*Nt_;
  //- populate the individual 1D grids
  xGrid = torch::linspace(lbX_, ubX_, Nx_,device_);
  yGrid = torch::linspace(lbY_, ubY_, Ny_,device_);
  tGrid = torch::linspace(lbT_, ubT_, Nt_,device_);
  //- construct entire mesh domain for transient 2D simulations
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
  //- spatial grid for steady state simulations 
  xyGrid = torch::meshgrid({xGrid,yGrid});
  //- tensor to pass for converged neural net
  xy = torch::stack({xyGrid[0].flatten(),xyGrid[1].flatten()},1);
  xy.set_requires_grad(true);
  //- create boundary grids
  createBC();
}

//- operator overload () to acess main computational domain
torch::Tensor  mesh2D::operator()(int i, int j, int k)  
{
  return torch::stack
  (
    {
      mesh_[0].index({i, j, k}), 
      mesh_[1].index({i, j, k}), 
      mesh_[2].index({i, j, k})
    }
  ); 
}

//- create boundary grids
void mesh2D::createBC()
{
  
  torch::Tensor xLeft = torch::tensor(lbX_,device_);
  torch::Tensor xRight = torch::tensor(ubX_,device_);
  torch::Tensor yBottom = torch::tensor(lbY_, device_);
  torch::Tensor yTop = torch::tensor(ubY_, device_);
  torch::Tensor tInitial = torch::tensor(lbT_,device_);
  if(net_->transient_==1)
  {
    leftWall = torch::meshgrid({xLeft,yGrid,tGrid});
    rightWall = torch::meshgrid({xRight,yGrid,tGrid});
    topWall = torch::meshgrid({xGrid,yTop,tGrid});
    bottomWall = torch::meshgrid({xGrid,yBottom,tGrid});
    initialGrid_ = torch::meshgrid({xGrid,yGrid,tInitial});
  }
  else 
  {
    leftWall = torch::meshgrid({xLeft,yGrid});
    rightWall = torch::meshgrid({xRight,yGrid});
    topWall = torch::meshgrid({xGrid,yTop});
    bottomWall = torch::meshgrid({xGrid,yBottom});
  }
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
  //- vectors to stack
  std::vector<torch::Tensor> vectorStack;
  //- total number of points in the grid
  int ntotal = grid[0].numel();
  //- random indices for PDE loss
  torch::Tensor indices = torch::randperm
  (ntotal,device_).slice(0,0,nSamples);
  
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
  if(net_->transient_==0)
  {
    //- create samples for intial condition loss only if simulationn is transient
    torch::Tensor batchIndices = torch::slice
    (
      pdeIndices_,
      0,
      iter*net_->BATCHSIZE,
      (iter + 1)*net_->BATCHSIZE-1 // -1 to not have overlapping samples
    );
    createSamples(xyGrid,iPDE_,batchIndices);
  }
  else
  {
    torch::Tensor batchIndices = pdeIndices_.slice
    (
      0,
      iter*net_->BATCHSIZE,
      (iter + 1)*net_->BATCHSIZE-1 // -1 to not have overlapping samples
    );
    createSamples(mesh_,iPDE_,batchIndices);
  }
  //- create samples only for the first iteration
  if(iter ==0)
  {
    if(net_->transient_ == 1)
    {
    //- update samples for intialGrid
      createSamples(initialGrid_,iIC_,net_->N_IC);
    }
    //- update samples for left wall 
    createSamples(leftWall,iLeftWall_,net_->N_BC);
    //- update samples for right wall 
    createSamples(rightWall, iRightWall_,net_->N_BC);
    //- update samples for top wall 
    createSamples(topWall,iTopWall_,net_->N_BC);
    //- update samples for bottom wall
    createSamples(bottomWall,iBottomWall_,net_->N_BC); 
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
  // std::cout<<"updating solution fields\n";
  //- update all fields
  fieldsPDE_ = net_->forward(iPDE_);
  if(net_->transient_ == 1)
  { 
    fieldsIC_ = net_->forward(iIC_);
  }
  fieldsLeft_ = net_->forward(iLeftWall_);
  fieldsRight_ = net_->forward(iRightWall_);
  fieldsBottom_ = net_->forward(iBottomWall_);
  fieldsTop_ = net_->forward(iTopWall_);
}

//- creates indices tensor for iPDE
void mesh2D::createIndices()
{
  if(net_->transient_==0)
  {
    pdeIndices_ = 
      torch::randperm(xyGrid[0].numel(),device_).slice(0,0,net_->N_EQN);
  }
  else
  {
    pdeIndices_ = 
      torch::randperm(mesh_[0].numel(),device_).slice(0,0,net_->N_EQN);
    
  }
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
  mesh_ = torch::meshgrid({xGrid,yGrid,tGrid});
  //- update the boundary grids
  createBC();
  //- transfer over parameters of current converged net to 
  //- previous net reference to use as intial condition for 
  //- intial losses
  loadState(net_, netPrev_);
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


















