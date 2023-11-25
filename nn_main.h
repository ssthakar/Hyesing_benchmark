#include <ATen/ops/meshgrid.h>
#include <ATen/ops/tensor.h>
#include <c10/core/TensorImpl.h>
#include <memory>
#include <sstream>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/serialize/input-archive.h>
#include <torch/serialize/output-archive.h>
#include <torch/torch.h> // should include almost everything
#include <torch/nn.h>
#include <iostream>
#include <math.h>
#include <string>
#include "utils.h"
                     //----thermophysical properties----//
class thermoPhysical
{
  public:
    //- constructor reads in properties from file
    thermoPhysical(Dictionary &dict);
    
    //- mobility
    float Mo;

    //- interface thickness
    float epsilon;
    
    //- surface tension coeff
    float sigma0;

    //- viscosity of liquid
    float muL;
    //- viscosity of gas
    float muG;

    //- density of liquid
    float rhoL;
    //- density of gas
    float rhoG;

    //- constant value for 3*sqrt(2)/4
    float C;

};
                      //----PINNs class declaration----//


// typedef std::shared_ptr<pinnImpl> mlpPtr;


//- Neural network class
class  PinNetImpl:
    public torch::nn::Module
{
    //- private fields
    private:
        //- create and  register sub-modules with the main nn
        void create_layers();
    //- public fields
    public:
        //- parametrized constructor
        PinNetImpl
        (
         const Dictionary &dict
        ); 
    //- public member functions
        // forward propogation with relu activation
        torch::Tensor forward(const torch::Tensor &X);
    //- public members
        //- Dictionary reference
        const Dictionary& dict;
        // Sequential collection of hidden layers
        torch::nn::Sequential hidden_layers;
        //- input layer
        torch::nn::Linear input = nullptr; 
        //- output layer
        torch::nn::Linear output=nullptr;
        //- number of hidden layers
        //- number of hidden layers
        const int N_HIDDEN_LAYERS;
        //- Dimension of  input layer (no. of input features)
        const int INPUT_DIM;
        //- Dimension of output layer (no. of output features)
        const int OUTPUT_DIM;
        //- Dimension of hidden layers (no. of neurons)
        const int HIDDEN_LAYER_DIM;
        //- grid dimension
        int BATCHSIZE;
        //- maximum number of iterations for optim
        int MAX_STEPS;
        //- tolerance for residual
        float ABS_TOL;
        //- no. of sampling points for PDE loss
        int N_EQN;
        //- no. of sampling points for Boundary condition loss
        int N_BC;
        //- no. of sampling points for intial condition loss
        int N_IC;
        //- max number of epochs during each timestep
        int K_EPOCH;
        //- learning rate
        float L_RATE;
        //- flag for transient
        //- 0 for false else true
        int transient_;
        //- test 
        int test_;
        //- number of iterations in each epoch
        int NITER_;

};
//- create Torch module
TORCH_MODULE(PinNet);

//---------------------------------------------------------------------------// 

              //----Computational domain class declaration-----//

//- class to store in computational domain and solution fields
class mesh2D :
  public torch::nn::Module
  {
  //- public fields
  public:
    //- Dictionary reference
    const Dictionary &dict;
    //- current neural net instance
    PinNet &net_;
    //- previous neural net instance
    PinNet &netPrev_;
    //- lower bounds for x 
    const float lbX_;
    //- upper bounds for xGrid
    const float ubX_;
    //- deltaX
    const float deltaX_;
    //- lower bounds for y 
    const float lbY_;
    //- upper bounds for y 
    const float ubY_;
    //- step size for y
    const float deltaY_;
    //- lower bound for time  
    float lbT_;
    //- upper bound for time
    float ubT_;
    //- step size for t
    const float deltaT_;
    //- center of the bubble at time t=0
    const float xc;
    const float yc;
    //- complete computational domain (x X y X t)
    std::vector<torch::Tensor> mesh_;
    //- x grid (1D tensor)
    torch::Tensor xGrid;
    //- y grid
    torch::Tensor yGrid;
    //- t grid
    torch::Tensor tGrid;
    //- spatial grid
    std::vector<torch::Tensor> xyGrid;
    //-xy spatial grid to use for plotting 
    torch::Tensor xy;
    //- left wall grid
    std::vector<torch::Tensor> leftWall;
    torch::Tensor iLeftWall_;
    torch::Tensor il;
    torch::Tensor leftIndices_;
    //- right wall 
    std::vector<torch::Tensor> rightWall;
    torch::Tensor iRightWall_;
    torch::Tensor ir;
    torch::Tensor rightIndices_;
    //- top wall
    std::vector<torch::Tensor> topWall;
    torch::Tensor iTopWall_;
    torch::Tensor it;
    torch::Tensor topIndices_;
    //- bottom walla
    std::vector<torch::Tensor> bottomWall;
    torch::Tensor iBottomWall_;
    torch::Tensor ib;
    torch::Tensor bottomIndices_;
    //- reference to device
    torch::Device &device_;
    //- reference to thermoPhyiscal class
    thermoPhysical &thermo_;
    //- tensor to store in solution fields
    torch::Tensor fieldsPDE_;
    torch::Tensor fieldsIC_;
    torch::Tensor fieldsLeft_;
    torch::Tensor fieldsRight_;
    torch::Tensor fieldsTop_;
    torch::Tensor fieldsBottom_;
    //- sampling points for PDE loss
    torch::Tensor iPDE_;
    torch::Tensor pdeIndices_;
    //-sampling points for IC loss
    std::vector<torch::Tensor> initialGrid_;
    torch::Tensor iIC_;
    torch::Tensor icIndices_;
    //- number of points in x direction 
    int Nx_;
    //- number of points in y direction
    int Ny_;
    //- number of intervals in time 
    int Nt_;
    //- number of points in the spatial domain
    int Nxy_;
    //- total number of points
    int Ntotal_;
    //- time step for adaptive time marching 
    float TimeStep_;
    //- constructor
    mesh2D
    (
      Dictionary &meshDict,
      PinNet &net,
      PinNet &netPrev,
      torch::Device &device,
      thermoPhysical &thermo
    );
    //- operator overload to use index notation for access
    torch::Tensor operator()(int i,int j,int k);
    //- create boundary grids
    void createBC();
    //- after sub-net converges, upadate solution fields
    void update(int iter);
    //- general function to create samples for neural net input
    void createSamples 
    (
      std::vector<torch::Tensor> & grid, // grid to generate samples from
      torch::Tensor &samples, // reference to input feature tensor
      int nSamples // total number of samples to extract from grid
    );
    void createSamples
    (
      std::vector<torch::Tensor> &grid,
      torch::Tensor &samples,
      torch::Tensor &indices
    );
    void createIndices();
    //-  creates total samples 
    void createTotalSamples
    (
      int iter //index for batch for pde
    );
    //- update time parameters for next time interval
    void updateMesh();
};


//---------------------------------------------------------------------------//

                        //----governing Equations----//

//- begin CahnHillard namespace
namespace CahnHillard
{
//- returns the surface tension force term for the momentum equation
torch::Tensor surfaceTension
(
  const mesh2D &mesh,
  int dim
);
//- returns phi term used in CH and momentum equation 
torch::Tensor phi
(
  const mesh2D &mesh
);
//- continuity loss in 2D
torch::Tensor L_Mass2D
(
  const mesh2D &mesh // pass reference for gradient input
);
//- momentum loss for x direction in 2D 
torch::Tensor L_MomX2d
(
  const mesh2D &mesh
);
//- momentual loss for y direction in 2D
torch::Tensor L_MomY2d
(
  const mesh2D &mesh
);
//- loss from Cahn-Hillard equation for phase-field transport
torch::Tensor CahnHillard2D
(
  const mesh2D &mesh
);
//- get thermoPhysical properties
torch::Tensor thermoProp
(
  float propLiquid,
  float propGas,
  const torch::Tensor &I
);

torch::Tensor PDEloss(mesh2D &mesh);
torch::Tensor ICloss(mesh2D &mesh);

torch::Tensor slipWall(torch::Tensor &I,torch::Tensor &X, int dim);
torch::Tensor noSlipWall(torch::Tensor &I, torch::Tensor &X);

torch::Tensor BCloss(mesh2D &mesh);
//- total loss function for the net
torch::Tensor loss(mesh2D &mesh);

//- constrains C
torch::Tensor Cbar(const torch::Tensor &C);


//- generates phase field vars at t=0
torch::Tensor C_at_InitialTime(mesh2D &mesh);

torch::Tensor u_at_InitialTime(mesh2D &mesh);

torch::Tensor v_at_InitialTime(mesh2D &mesh);
//- zero gradient at horizontal or vertical wall for now,
//- TOOD make general boundary condition for curved walls based on points
torch::Tensor zeroGrad(torch::Tensor &I,torch::Tensor &X, int dim);

} 
//- end CahnHillard namespace 

//- Heat Equation 

namespace Heat
{
torch::Tensor L_Diffusion2D
(
  mesh2D &mesh
);

//- total loss
torch::Tensor loss(mesh2D &mesh);
    
}
//---------------------------------------------------------------------------//

                           //----loss functions----//

//- function to calculate higher order derivatives
torch::Tensor d_dn
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X, // independant vars
  int order, // order of derivative
  int spatialIndex // gradient wrt to which independant var? auto generator = torch::cuda::detail::getRandomCUDA()
);

//- function overload when X is a 1D tensor
torch::Tensor d_dn
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X, // independant vars
  int order // order of derivative
);

//- first partial derivative 
torch::Tensor d_d1
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X, // indpendant vars
  int spatialIndex // gradient wrt to which indpendant var?
);

//- function overload when X is a 1D tensor
torch::Tensor d_d1
(
  const torch::Tensor &I, // dependant scalar var
  const torch::Tensor &X // indpendant vars
);

//---------------------------------------------------------------------------//
//- function to make a local clone a neural net given already constructed 
//- essentially updates parameters using the copy function 
//- other members are left the same, intially the two nets are the same, 
//- once net1 converges, net2 takes the network parameters and acts as an initial 
//- condition while net1 continues to train
//- the process repeats each time a net converges or max iters run out
void loadState(PinNet& net1, PinNet &net2);
