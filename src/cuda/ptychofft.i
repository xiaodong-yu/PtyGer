/*interface*/
%module ptychofft

%{
#define SWIG_FILE_WITH_INIT
#include "ptychofft.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}

class ptychofft
{
  bool is_free = false;

  float2 *f;		// object
	float2 *g;		// data
	float2 *prb;	// probe function
	float *scanx;   // x scan positions
	float *scany;   // y scan positions
	float2 *shiftx; // x shift (-1,1) of scan positions to nearest integer
	float2 *shifty; // y shift (-1,1) of scan positions to nearest integer

	cufftHandle plan2d; // 2D FFT plan
	cufftHandle plan2dshift; // 2D FFT plan for the shift in the frequency domain

	dim3 BS3d; // 3d thread block on GPU

	// 3d thread grids on GPU for different kernels
	dim3 GS3d0;
	dim3 GS3d1;
	dim3 GS3d2;

public:
  %immutable;
  size_t ptheta; // number of projections
  size_t nz;	 // object vertical size
  size_t n;	  // object horizontal size
  size_t nscan;  // number of scan positions for 1 projection
  size_t ndetx;  // detector x size
  size_t ndety;  // detector y size
  size_t nprb;   // probe size in 1 dimension
  %mutable;

	// constructor, memory allocation
	ptychofft(size_t ptheta, size_t nz, size_t n,
			  size_t nscan, size_t ndetx, size_t ndety, size_t nprb);
	// destructor, memory deallocation
	~ptychofft();
	// forward ptychography operator FQ
	void fwd(size_t g_, size_t f_, size_t scan_, size_t prb_);
	// adjoint ptychography operator with respect to object (fgl==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
	void adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg);
  void free();
};
