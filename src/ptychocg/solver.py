"""A module for ptychography solvers.

This module implements ptychographic solvers which all inherit from a
ptychography base class. The base class implements the forward and adjoint
ptychography operators and manages GPU memory.

Solvers in this module are Python context managers which means they should be
instantiated using a with-block. e.g.

```python
# load data and such
data = cp.load(...)
# instantiate the solver with memory allocation related parameters
with CustomPtychoSolver(...) as solver:
    # call the solver with solver specific parameters
    result = solver.run(data, ...)
# solver memory is automatically freed at with-block exit
```

Context managers are capable of gracefully handling interruptions (CTRL+C).

"""

import warnings
import numpy as np
import cupy as cp
import sys
import signal
from time import perf_counter
from ptychocg.ptychofft import ptychofft


class PtychoCuFFT(ptychofft):
    """Base class for ptychography solvers using the cuFFT library.

    This class is a context manager which provides the basic operators required
    to implement a ptychography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.

    Attribtues
    ----------
    nscan : int
        The number of scan positions at each angular view.
    nprb : int
        The pixel width and height of the probe illumination.
    ndetx, ndety : int
        The pixel width and height of the detector.
    ntheta : int
        The number of angular partitions of the data.
    n, nz : int
        The pixel width and height of the reconstructed grid.
    ptheta : int
        The number of angular partitions to process together
        simultaneously.
    """

    def __init__(self, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta):
        """Please see help(PtychoCuFFT) for more info."""
        super().__init__(ptheta, nz, n, nscan, ndetx, ndety, nprb)
        self.ntheta = ntheta  # number of projections
        self.fpsi = cp.zeros([self.ptheta, self.nscan, self.ndety, self.ndetx],
                       dtype='complex64')
        self.fdpsi = cp.zeros([self.ptheta, self.nscan, self.ndety, self.ndetx],
                       dtype='complex64')
        self.gradpsi0 = cp.zeros([self.ntheta, self.nz, self.n], dtype='complex64')

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_ptycho(self, res, psi, scan, prb):
        """Ptychography transform (FQ)."""
        assert psi.dtype == cp.complex64, f"{psi.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert prb.dtype == cp.complex64, f"{prb.dtype}"
        #res = cp.zeros([self.ptheta, self.nscan, self.ndety, self.ndetx],
        #               dtype='complex64')
        self.fwd(res.data.ptr, psi.data.ptr, scan.data.ptr, prb.data.ptr)
        return res

    def fwd_ptycho_batch(self, psi, scan, prb):
        """Batch of Ptychography transform (FQ)."""
        assert psi.dtype == cp.complex64, f"{psi.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert prb.dtype == cp.complex64, f"{prb.dtype}"
        data = cp.zeros([self.ntheta, self.nscan, self.ndety, self.ndetx],
                        dtype='float32')
        res = cp.zeros([self.ptheta, self.nscan, self.ndety, self.ndetx],
                       dtype='complex64')
        # angle partitions in ptychography
        for k in range(0, self.ntheta // self.ptheta):
            res.fill(0+0j)
            ids = np.arange(k * self.ptheta, (k + 1) * self.ptheta)
            # compute part on GPU
            data[ids] = cp.abs(self.fwd_ptycho(res, psi[ids], scan[:, ids],
                                           prb[ids]))**2
            #data[ids] = data0.get()  # copy to CPU
        del res
        return data

    def adj_ptycho(self, data, scan, prb):
        """Adjoint ptychography transform (Q*F*)."""
        assert data.dtype == cp.complex64, f"{data.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert prb.dtype == cp.complex64, f"{prb.dtype}"
        res = cp.zeros([self.ptheta, self.nz, self.n], dtype='complex64')
        flg = 0  # compute adjoint operator with respect to object
        self.adj(res.data.ptr, data.data.ptr, scan.data.ptr, prb.data.ptr, flg)
        return res

    def adj_ptycho_prb(self, data, scan, psi):
        """Adjoint ptychography probe transform (O*F*), object is fixed."""
        assert data.dtype == cp.complex64, f"{data.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert psi.dtype == cp.complex64, f"{psi.dtype}"
        res = cp.zeros([self.ptheta, self.nprb, self.nprb], dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        self.adj(psi.data.ptr, data.data.ptr, scan.data.ptr, res.data.ptr, flg)
        return res

    def grad(self, data, psi, scan, prb, **kwargs):
        """Placehold for a child's solving function."""
        raise NotImplementedError("Cannot run a base class.")

    def grad_batch(self, data, psi, scan, prb, first, dpsi, gradpsi, testpsi,  gammapsi, **kwargs):
        """Run by dividing the work into batches."""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim

        #psi = psi.copy()
        #prb = prb.copy()

        # angle partitions in ptychography
        for k in range(0, self.ntheta // self.ptheta):
            ids = np.arange(k * self.ptheta, (k + 1) * self.ptheta)
            #datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            result = self.grad(
                data[ids],
                psi[ids],
                scan[:, ids],
                prb[ids, :, :],
                first,
                dpsi[ids],
                gradpsi[ids],
                testpsi[ids],
                **kwargs,
            )
            psi[ids], prb[ids], dpsi[ids], gradpsi[ids], testpsi[ids], gammapsi = result['psi'], result['prb'], result['dpsi'], result['gradpsi'], result['testpsi'], result['gammapsi']
        return {
            'psi': psi,
            'prb': prb,
            'dpsi': dpsi,
            'gradpsi': gradpsi,
            'testpsi': testpsi,
            'gammapsi': gammapsi,
        }

    def dir(self, data, psi, scan, prb, **kwargs):
        """Placehold for a child's solving function."""
        raise NotImplementedError("Cannot run a base class.")

    def dir_batch(self, data, psi, scan, prb, first, dpsi, gradpsi, testpsi,  gammapsi, **kwargs):
        """Run by dividing the work into batches."""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim

        #psi = psi.copy()
        #prb = prb.copy()

        # angle partitions in ptychography
        for k in range(0, self.ntheta // self.ptheta):
            ids = np.arange(k * self.ptheta, (k + 1) * self.ptheta)
            #datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            result = self.dir(
                data[ids],
                psi[ids],
                scan[:, ids],
                prb[ids, :, :],
                first,
                dpsi[ids],
                gradpsi[ids],
                testpsi[ids],
                **kwargs,
            )
            psi[ids], prb[ids], dpsi[ids], gradpsi[ids], testpsi[ids], f = result['psi'], result['prb'], result['dpsi'], result['gradpsi'], result['testpsi'], result['f']
        return {
            'psi': psi,
            'prb': prb,
            'dpsi': dpsi,
            'gradpsi': gradpsi,
            'testpsi': testpsi,
            'f': f,
        }

    def minf(self, data, psi, scan, prb, **kwargs):
        """Placehold for a child's solving function."""
        raise NotImplementedError("Cannot run a base class.")

    def minf_batch(self, data, psi, scan, prb, first, dpsi, gradpsi, testpsi,  gammapsi, **kwargs):
        """Run by dividing the work into batches."""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim

        #psi = psi.copy()
        #prb = prb.copy()

        # angle partitions in ptychography
        for k in range(0, self.ntheta // self.ptheta):
            ids = np.arange(k * self.ptheta, (k + 1) * self.ptheta)
            #datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            f = self.minf(
                data[ids],
                psi[ids],
                scan[:, ids],
                prb[ids, :, :],
                first,
                dpsi[ids],
                gradpsi[ids],
                gammapsi,
                **kwargs,
            )
        return f

    def update(self, data, psi, scan, prb, **kwargs):
        """Placehold for a child's solving function."""
        raise NotImplementedError("Cannot run a base class.")

    def update_batch(self, data, psi, scan, prb, first, dpsi, gradpsi, testpsi,  gammapsi, **kwargs):
        """Run by dividing the work into batches."""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim

        #psi = psi.copy()
        #prb = prb.copy()

        # angle partitions in ptychography
        for k in range(0, self.ntheta // self.ptheta):
            ids = np.arange(k * self.ptheta, (k + 1) * self.ptheta)
            #datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            result = self.update(
                data[ids],
                psi[ids],
                scan[:, ids],
                prb[ids, :, :],
                first,
                dpsi[ids],
                gradpsi[ids],
                gammapsi,
                **kwargs,
            )
            psi[ids], prb[ids], dpsi[ids], gradpsi[ids], gammapsi = result['psi'], result['prb'], result['dpsi'], result['gradpsi'], result['gammapsi']
        return {
            'psi': psi,
            'prb': prb,
            'dpsi': dpsi,
            'gradpsi': gradpsi,
            'gammapsi': gammapsi,
        }

class CGPtychoSolver(PtychoCuFFT):
    """Solve the ptychography problem using congujate gradient."""

    #@staticmethod
    #def line_search(f, x, d, step_length=1, step_shrink=0.5):
    #    """Return a new step_length using a backtracking line search.

    #    https://en.wikipedia.org/wiki/Backtracking_line_search

    #    Parameters
    #    ----------
    #    f : function(x)
    #        The function being optimized.
    #    x : vector
    #        The current position.
    #    d : vector
    #        The search direction.

    #    """
    #    assert step_shrink > 0 and step_shrink < 1
    #    m = 0  # Some tuning parameter for termination
    #    fx = f(x)  # Save the result of f(x) instead of computing it many times
    #    # Decrease the step length while the step increases the cost function
    #    accu = 0
    #    while f(x + step_length * d) > fx + step_shrink * m:
    #        accu += 1
    #        if step_length < 1e-32:
    #            warnings.warn("Line search failed for conjugate gradient.")
    #            return 0
    #        step_length *= step_shrink
    #    return step_length

    def grad(
            self,
            data,
            psi,
            scan,
            prb,
            first,
            dpsi,
            gradpsi,
            testpsi,
            piter,
            model='gaussian',
            recover_prb=False,
    ):
        """Conjugate gradients for ptychography.

        Parameters
        ----------
        model : str gaussian or poisson
            The noise model to use for the gradient.
        piter : int
            The number of gradient steps to take.
        recover_prb : bool
            Whether to recover the probe or assume the given probe is correct.

        """
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim


        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min"
              )  # csv column headers
        gammaprb=0
        for i in range(piter):
            # 1) object retrieval subproblem with fixed probe
            # forward operator
            self.fpsi.fill(0+0j)
            self.fpsi = self.fwd_ptycho(self.fpsi, psi, scan, prb)
            gradpsi0 = gradpsi
           # take gradient
            if model == 'gaussian':
                gradpsi = self.adj_ptycho(
                    self.fpsi - cp.sqrt(data) * cp.exp(1j * cp.angle(self.fpsi)),
                    scan,
                    prb,
                ) / (cp.max(cp.abs(prb))**2)
            elif model == 'poisson':
                gradpsi = self.adj_ptycho(
                    self.fpsi - data * self.fpsi / (cp.abs(self.fpsi)**2 + 1e-32),
                    scan,
                    prb,
                ) / (cp.max(cp.abs(prb))**2)
            gammapsi = 0.25

            if (recover_prb):
                # 2) probe retrieval subproblem with fixed object
                # forward operator
                fprb = self.fwd_ptycho(psi, scan, prb)
                # take gradient
                if model == 'gaussian':
                    gradprb = self.adj_ptycho_prb(
                        fprb - cp.sqrt(data) * cp.exp(1j * cp.angle(fprb)),
                        scan,
                        psi,
                    ) / cp.max(cp.abs(psi))**2 / self.nscan
                elif model == 'poisson':
                    gradprb = self.adj_ptycho_prb(
                        fprb - data * fprb / (cp.abs(fprb)**2 + 1e-32),
                        scan,
                        psi,
                    ) / cp.max(cp.abs(psi))**2 / self.nscan
                # Dai-Yuan direction
                if (i == 0):
                    dprb = -gradprb
                else:
                    dprb = -gradprb + (
                        cp.linalg.norm(gradprb)**2 /
                        (cp.sum(cp.conj(dprb) * (gradprb - gradprb0))) * dprb)
                gradprb0 = gradprb
                # line search
                fdprb = self.fwd_ptycho(psi, scan, dprb)
                gammaprb = self.line_search(minf, fprb, fdprb)
                # update prb
                prb = prb + gammaprb * dprb

        #    # check convergence
        #    if (np.mod(i, 8) == 0):
        #        fpsi = self.fwd_ptycho(psi, scan, prb)
        #        print("%4d, %.3e, %.3e, %.7e" %
        #              (i, gammapsi, gammaprb, minf(fpsi)))

        return {
            'psi': psi,
            'prb': prb,
            'dpsi': dpsi,
            'gradpsi': gradpsi,
            'testpsi': testpsi,
            'gammapsi': gammapsi,
        }

    def dir(
            self,
            data,
            psi,
            scan,
            prb,
            first,
            dpsi,
            gradpsi,
            testpsi,
            piter,
            model='gaussian',
            recover_prb=False,
    ):
        # Dai-Yuan direction
        #dpsi = -gradpsi
        if first == True:
            dpsi = -gradpsi
            testpsi = dpsi
        else:
            dpsi = -gradpsi + (
                cp.linalg.norm(gradpsi)**2 /
                (cp.sum(cp.conj(dpsi) * (gradpsi - self.gradpsi0))) * dpsi)
        self.gradpsi0 = gradpsi.copy()
        #gammapsi = 0.25
        # line search
        if model == 'gaussian':
            f = cp.linalg.norm(cp.abs(self.fpsi) - cp.sqrt(data))**2
        elif model == 'poisson':
            f = cp.sum(
                cp.abs(self.fpsi)**2 - 2 * data * cp.log(cp.abs(self.fpsi) + 1e-32))
        self.fdpsi.fill(0+0j)
        self.fdpsi = self.fwd_ptycho(self.fdpsi, dpsi, scan, prb)

        return {
            'psi': psi,
            'prb': prb,
            'dpsi': dpsi,
            'gradpsi': gradpsi,
            'testpsi': testpsi,
            'f': f,
        }

    def minf(
            self,
            data,
            psi,
            scan,
            prb,
            first,
            dpsi,
            gradpsi,
            gammapsi,
            piter,
            model='gaussian',
            recover_prb=False,
    ):
        #self.fpsi = self.fpsi + gammapsi * self.fdpsi
        if model == 'gaussian':
            f = cp.linalg.norm(cp.abs(self.fpsi + gammapsi * self.fdpsi) - cp.sqrt(data))**2
        elif model == 'poisson':
            f = cp.sum(
                cp.abs(self.fpsi)**2 - 2 * data * cp.log(cp.abs(self.fpsi) + 1e-32))
        return f

    def update(
            self,
            data,
            psi,
            scan,
            prb,
            first,
            dpsi,
            gradpsi,
            gammapsi,
            piter,
            model='gaussian',
            recover_prb=False,
    ):
        # minimization functional

        #t1_start = perf_counter()
        #gammapsi = self.line_search(minf, self.fpsi, self.fdpsi)
        #t1_stop = perf_counter()
        #print("Elapsed time during the whole program in seconds:",
        #                                                t1_stop-t1_start)
        # update psi
        deltapsi = gammapsi * dpsi
        psi = psi + gammapsi * dpsi
        #psi_gpu = cp.asnumpy(psi[0])

        return {
            'psi': psi,
            'prb': prb,
            'dpsi': dpsi,
            'gradpsi': gradpsi,
            'gammapsi': gammapsi,
        }
