// Ptychography kernels on GPU

#define PI 3.1415926535

// probe multiplication of the object array part
void __global__ mulprobe(float2 *g, float2 *f, float2 *prb, float *scanx, float *scany,
						 int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
		return;

	// coordinates in the probe array
	int ix = tx % Nprb;
	int iy = tx / Nprb;
	// shift in the object array to the starting point of probe multiplication
	int shift = (Ndety - Nprb) / 2 * Ndetx + (Ndetx - Nprb) / 2;

	float2 prb0 = prb[ix + iy * Nprb + tz * Nprb * Nprb];									  // probe voxel for multiplication
	float2 g0 = g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan]; // object voxel for multiplication
	float c = 1 / sqrtf(Ndetx * Ndety);														  //fft constant

	// multilication in complex variables
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].x = c * prb0.x * g0.x - c * prb0.y * g0.y;
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].y = c * prb0.x * g0.y + c * prb0.y * g0.x;
}

// adjoint probe multiplication of the object array part
void __global__ mulaprobe(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
						  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
		return;

	// coordinates in the probe array
	int ix = tx % Nprb;
	int iy = tx / Nprb;
	// closest integers for scan positions
	int stx = roundf(scanx[ty + tz * Nscan]);
	int sty = roundf(scany[ty + tz * Nscan]);
	// skip scans where the probe position is negative (undefined)
	if (stx < 0 || sty < 0)
		return;
	// shift in the object array to the starting point of probe multiplication
	int shift = (Ndety - Nprb) / 2 * Ndetx + (Ndetx - Nprb) / 2;

	float2 g0 = g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan]; // data voxel for multiplication
	float2 prb0 = prb[ix + iy * Nprb + tz * Nprb * Nprb];									  // probe voxel for multiplication
	float c = 1 / sqrtf(Ndetx * Ndety);														  //fft constant
	// multilication in complex variables with simultaneous writing to the object array
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].x = c * prb0.x * g0.x + c * prb0.y * g0.y;
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].y = c * prb0.x * g0.y - c * prb0.y * g0.x;
}

// adjoint object part multiplication of the probe array
void __global__ mulaobj(float2 *prb, float2 *g, float2 *f, float *scanx, float *scany,
						int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
		return;

	// coordinates in the probe array
	int ix = tx % Nprb;
	int iy = tx / Nprb;
	// closest integers for scan positions
	int stx = roundf(scanx[ty + tz * Nscan]);
	int sty = roundf(scany[ty + tz * Nscan]);
	// skip scans where the probe position is negative (undefined)
	if (stx < 0 || sty < 0)
		return;
	// shift in the object array to the starting point of probe multiplication
	int shift = (Ndety - Nprb) / 2 * Ndetx + (Ndetx - Nprb) / 2;

	float2 g0 = g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan]; // data voxel for multiplication
	float2 f0 = f[(stx + ix) + (sty + iy) * N + tz * Nz * N];								  // object voxel for multiplication
	float c = 1 / sqrtf(Ndetx * Ndety);														  //fft constant
	// multilication in complex variables with simultaneous writing to the object array
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].x = c * f0.x * g0.x + c * f0.y * g0.y;
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].y = c * f0.x * g0.y - c * f0.y * g0.x;
}

// take part of the object array
void __global__ takepart(float2 *g, float2 *f, float2 *prb, float *scanx, float *scany,
						 int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
		return;

	// coordinates in the probe array
	int ix = tx % Nprb;
	int iy = tx / Nprb;
	// closest integers for scan positions
	int stx = roundf(scanx[ty + tz * Nscan]);
	int sty = roundf(scany[ty + tz * Nscan]);
	// skip scans where the probe position is negative (undefined)
	if (stx < 0 || sty < 0)
		return;
	// shift in the object array to the starting point of probe multiplication
	int shift = (Ndety - Nprb) / 2 * Ndetx + (Ndetx - Nprb) / 2;

	// take the part
	float2 f0 = f[(stx + ix) + (sty + iy) * N + tz * Nz * N];

	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].x = f0.x;
	g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan].y = f0.y;
}

// simultaneous writing to the object array
void __global__ setpartobj(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
						   int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
		return;

	// coordinates in the probe array
	int ix = tx % Nprb;
	int iy = tx / Nprb;
	// closest integers for scan positions
	int stx = roundf(scanx[ty + tz * Nscan]);
	int sty = roundf(scany[ty + tz * Nscan]);
	// skip scans where the probe position is negative (undefined)
	if (stx < 0 || sty < 0)
		return;
	// shift in the object array to the starting point of probe multiplication
	int shift = (Ndety - Nprb) / 2 * Ndetx + (Ndetx - Nprb) / 2;

	float2 g0 = g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan]; // data voxel for multiplication
	// multilication in complex variables with simultaneous writing to the object array
	atomicAdd(&f[(stx + ix) + (sty + iy) * N + tz * Nz * N].x, g0.x);
	atomicAdd(&f[(stx + ix) + (sty + iy) * N + tz * Nz * N].y, g0.y);
}

// simultaneous writing to the probe array
void __global__ setpartprobe(float2 *prb, float2 *g, float2 *f, float *scanx, float *scany,
							 int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)

{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
		return;

	// coordinates in the probe array
	int ix = tx % Nprb;
	int iy = tx / Nprb;
	// closest integers for scan positions
	int stx = roundf(scanx[ty + tz * Nscan]);
	int sty = roundf(scany[ty + tz * Nscan]);
	// skip scans where the probe position is negative (undefined)
	if (stx < 0 || sty < 0)
		return;
	// shift in the object array to the starting point of probe multiplication
	int shift = (Ndety - Nprb) / 2 * Ndetx + (Ndetx - Nprb) / 2;

	float2 g0 = g[shift + ix + iy * Ndetx + ty * Ndetx * Ndety + tz * Ndetx * Ndety * Nscan]; // data voxel for multiplication
	// multilication in complex variables with simultaneous writing to the object array
	atomicAdd(&prb[ix + iy * Nprb + tz * Nprb * Nprb].x, g0.x);
	atomicAdd(&prb[ix + iy * Nprb + tz * Nprb * Nprb].y, g0.y);
}

// compute exp(1j dir* dx),exp(1j dir* dy) where dx,dy are in (-1,1) and correspond to shifts to nearest integer,, dir==1 - fowrward, dir==-1 backward
void __global__ takeshifts(float2 *shiftx, float2 *shifty, float *scanx, float *scany, int dir, int Ntheta, int Nscan)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx >= Nscan || ty >= Ntheta)
		return;
	int ind = tx + ty * Nscan;
	// compute exp(1j dx),exp(1j dy) in terms of cos and sin
	shiftx[ind].x = cosf(2 * PI * dir * (scanx[ind] - roundf(scanx[ind])));
	shiftx[ind].y = sinf(2 * PI * dir * (scanx[ind] - roundf(scanx[ind])));
	shifty[ind].x = cosf(2 * PI * dir * (scany[ind] - roundf(scany[ind])));
	shifty[ind].y = sinf(2 * PI * dir * (scany[ind] - roundf(scany[ind])));
}

// perform shifts in the frequency domain by multiplication with exp(1j dx),exp(j dy)
void __global__ shifts(float2 *f, float2 *shiftx, float2 *shifty, int Ntheta, int Nscan, int NdetxNdety, int NprbNprb)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx >= NdetxNdety || ty >= Nscan || tz >= Ntheta)
		return;
	int ind = tx + ty * NdetxNdety + tz * NdetxNdety * Nscan;
	int inds = ty + tz * Nscan;
	// multiplication f by complex numbers (shiftx,shifty)
	float2 f0 = f[ind];
	float2 shiftx0 = shiftx[inds];
	float2 shifty0 = shifty[inds];
	float c = 1 / (float)(NprbNprb); //fft constant for shifts
	f[ind].x = f0.x * shiftx0.x - f0.y * shiftx0.y;
	f[ind].y = f0.y * shiftx0.x + f0.x * shiftx0.y;
	f0 = f[ind];
	f[ind].x = c * (f0.x * shifty0.x - f0.y * shifty0.y);
	f[ind].y = c * (f0.y * shifty0.x + f0.x * shifty0.y);
}
