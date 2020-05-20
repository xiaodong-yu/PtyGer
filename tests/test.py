import h5py
import numpy as np
import cupy as cp
import dxchange
from scipy import ndimage
import sys
import warnings
import ptychocg as pt
import concurrent.futures as cf
from functools import partial
from numpy import linalg as la
from skimage.measure import block_reduce
from skimage.metrics import structural_similarity as ssim
from cupy import linalg
from time import perf_counter
import time

def read_data(idd,step):

    name = '/home/beams/XYU/chip_pillar_interlace_2/projection_%06d.h5' % (idd)
    fid = h5py.File(name, 'r')
    # read scan positions
    tmp = fid['/positions'][::step]*70976797.5332996
    scan = np.zeros([2, 1, tmp.shape[0]], dtype='float32', order='C')
    nscan = tmp.shape[0]
    scan[0, :, :] = tmp[:, 1]+533
    scan[1, :, :] = tmp[:, 0]+428

    ids = np.where((scan[0,0]>=0)*(scan[0,0]<1024)*(scan[1,0]>=0)*(scan[1,0]<768))[0]
    scan = np.array(scan[:,:,ids], order='C') # important!
    nscan = scan.shape[2]
    # import matplotlib.pyplot as plt
    # plt.plot(scan[1,0], scan[0,0],'.', color='blue')
    # plt.savefig('scan.png',dpi=500)
    data = np.zeros([1, nscan, 256, 256], dtype='float32')
    tmp = np.fft.fftshift(fid['data'][::step, :, :], axes=(1, 2)).astype('float32')
    data[0] = tmp[ids]
    # read probe initial guess
    prb = np.ones([1, 256, 256], dtype='complex64')
    prbamp = dxchange.read_tiff('/home/beams/XYU/chip_pillar_interlace_2/prbamp.tiff')
    prbangle = dxchange.read_tiff('/home/beams/XYU/chip_pillar_interlace_2/prbangle.tiff')
    prb[0] = prbamp*np.exp(1j*prbangle)
    # initial guess for psi (can be read)
    psi = np.zeros([1, 768+256, 1024+256], dtype='complex64', order='C')+1
    return psi, prb, scan, data

def read_synthetic(ds_name, ds_prb, ds_scan, step):

    # read scan positions
    tmp = np.load('data/'+ds_name+'/scan'+ds_scan+'.npy')
    #tmp = np.load('/home/beams/XYU/tike/tests/coins/scan512.npy')
    scan = tmp[:, ::step]
    print(scan.shape)
    print(scan[:,:,0].shape)
    #for e in np.nditer(tmp[0, :, 0], flags = ['external_loop'], order='C'):

    scan = np.moveaxis(scan, [0, 1, 2], [-2, -1, -3])
    nscan = scan.shape[2]
    # import matplotlib.pyplot as plt
    # plt.plot(scan[1,0], scan[0,0],'.', color='blue')
    # plt.savefig('scan.png',dpi=500)
    # read probe initial guess
    prb = np.load('data/'+ds_name+'/probe'+ds_prb+'.npy')
    #prb = np.load('/home/beams/XYU/tike/tests/coins/probe512.npy')
    print(prb.dtype)
    data = np.load('data/'+ds_name+'/'+ds_prb+'data'+ds_scan+'.npy')
    #data = np.zeros([1, scan.shape[2], prb.shape[1], prb.shape[2]], dtype='float32')
    #data = np.load('/home/beams/XYU/tike/tests/coins/data512.npy')
    data = data[:, ::step]
    print(data.dtype)
    # initial guess for psi (can be read)
    psi_ref = np.load('data/'+ds_name+'/'+ds_name+'.npy')
    #dxchange.write_tiff(np.angle(psi_ref),
    #                    'rec_'+ds_name+'/refangle_e', overwrite=True)
    #dxchange.write_tiff(np.abs(psi_ref),  'rec_'+ds_name+'/refamp_e', overwrite=True)
    #exit()
    #psi_ref = np.load('/home/beams/XYU/tike/tests/coins/coins.npy')
    psi = np.zeros([1, psi_ref.shape[1], psi_ref.shape[2]], dtype='complex64', order='c')+1
    return psi_ref, psi, prb, scan, data

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        gpu_count = 2
    else:
        gpu_count = np.int(sys.argv[1])
        assert gpu_count%2 == 0, "# of GPU should be an even number"
    if (gpu_count == 0):
        mgpu = False
    else:
        mgpu = True
    gpu_offset = np.int(sys.argv[2])
    idd = np.int(sys.argv[3])
    step = np.int(sys.argv[4])
    pit = np.int(sys.argv[5]) # ptychography iterations
    fname = sys.argv[6]
    open(fname+'_iter.txt', 'w').close()
    open(fname+'_norm.txt', 'w').close()
    open(fname+'_time.txt', 'w').close()
    file_iter = open(fname+'_iter.txt', 'a')
    file_norm = open(fname+'_norm.txt', 'a')
    file_time = open(fname+'_time.txt', 'a')
    print(fname)
    syn = str(sys.argv[7])
    #cp.cuda.Device(igpu).use()  # gpu id to use
    if (syn == '-s'):
        synth = True
        if (len(sys.argv) < 11):
            sys.exit('Must specify the input data name.')
        ds_name = str(sys.argv[8])
        ds_prb = str(sys.argv[9])
        ds_scan = str(sys.argv[10])
        psi_ref,psi_cpu,prb_cpu,scan_cpu,data_cpu = read_synthetic(ds_name,ds_prb,ds_scan,step)
        open(fname+'_ssim.txt', 'w').close()
        file_ssim = open(fname+'_ssim.txt', 'a')
    elif (syn == '-r'):
        synth = False
        psi_cpu,prb_cpu,scan_cpu,data_cpu = read_data(idd,step)
    else:
        sys.exit('Must specify the input data type.')

    print(scan_cpu.shape)
    print(data_cpu.shape)
    print(prb_cpu.shape)

    [ntheta, nz,n] = psi_cpu.shape
    print("ntheta:", ntheta, nz, n)
    [ntheta,nscan,ndety,ndetx] = data_cpu.shape
    print("nscan:", ntheta, nscan, ndety, ndetx)
    print("scan:", scan_cpu.shape)
    nscan = scan_cpu.shape[2]
    nprb = prb_cpu.shape[2]
    print("nprb:", nprb)
    ptheta = 1 # number of angles to process simultaneosuly
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    recover_prb = False # recover probe or not

    t1 = perf_counter()
    # set cupy to use unified memory
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    #gpu_count = 4

    pinned_memory_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

    if mgpu == False:
        cp.cuda.Device(gpu_offset).use()  # gpu id to use
        # Class gpu solver
        if synth == True:
            psi_e_cpu = np.zeros([1, psi_ref.shape[1], psi_ref.shape[2]], dtype='complex64', order='C')+1
            psi_e = cp.zeros([1, psi_ref.shape[1], psi_ref.shape[2]], dtype='complex64', order='C')+1
        else:
            psi_e_cpu = np.zeros([1, 768+256, 1024+256], dtype='complex64', order='C')+1
            psi_e = cp.zeros([1, 768+256, 1024+256], dtype='complex64', order='C')+1
        scan_e = cp.zeros([2, ntheta, nscan], dtype='float32')
        data_e = cp.zeros([1, nscan, ndety, ndetx], dtype='float32')

        #ith pt.CGPtychoSolver(nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta) as slv:
        slv = pt.CGPtychoSolver(nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta)
        # Compute data
        #data = slv.fwd_ptycho_batch(psi0, scan, prb0)
        #dxchange.write_tiff(data, 'data', overwrite=True)
        scan_e = cp.array(scan_cpu)

        # Initial guess
        #psi = cp.ones([ntheta, nz, n], dtype='complex64')
        psi_e = cp.array(psi_cpu)
        dpsi0_e = cp.zeros([ntheta, nz, n], dtype='complex64')
        dpsi_e = cp.zeros([ntheta, nz, n], dtype='complex64')
        dpsi_et = cp.zeros([ntheta, nz, n], dtype='complex64')
        test_psi = cp.zeros([ntheta, nz, n], dtype='complex64')
        gradpsi0_e = cp.zeros([ntheta, nz, n], dtype='complex64')
        gradpsi_e = cp.zeros([ntheta, nz, n], dtype='complex64')
        #deltapsi_e = cp.zeros([ntheta, nz, n], dtype='complex64')
        gammapsi_e = 0.0
        fx_e = 0.0
        fdx_e = 0.0
        step_length = 1
        step_shrink = 0.5
        #if (recover_prb):
        #    # Choose an adequate probe approximation
        #    prb = prb0.copy().swapaxes(1, 2)
        #else:
        #    prb = prb0.copy()
        prb_e = cp.array(prb_cpu)
        if synth == True:
            #slv_gen = pt.CGPtychoSolver(nscan//10, nprb, ndetx, ndety, ntheta, nz, n, ptheta)
            #for i in range(10):
            #    data_e[:, nscan//10*i:nscan//10*(i+1)] = slv_gen.fwd_ptycho_batch(cp.array(psi_ref), scan_e[:, :, nscan//10*i:nscan//10*(i+1)], prb_e)
            #del slv_gen
            #np.save('/home/beams/XYU/tike/tests/'+ds_name+'/'+ds_prb+'data'+ds_scan, cp.asnumpy(data_e))
            #exit()
            data_e = cp.array(data_cpu)
        else:
            data_e = cp.array(data_cpu)

        print('test:', data_e.shape)
        time = 0.0
        for i in range(pit):
            t2 = perf_counter()
            if i == 0:
                first = True
            else:
                first = False
            result_e = slv.grad_batch(
                data_e, psi_e, scan_e, prb_e, first, dpsi_e, gradpsi_e, test_psi,  gammapsi_e, piter=1, model=model, recover_prb=recover_prb)
            psi_e, prb_e, dpsi_e, gradpsi_e, test_psi, gammapsi_e = result_e['psi'], result_e['prb'], result_e['dpsi'], result_e['gradpsi'], result_e['testpsi'], result_e['gammapsi']
            result_e = slv.dir_batch(
                data_e, psi_e, scan_e, prb_e, first, dpsi_e, gradpsi_e, test_psi,  gammapsi_e, piter=1, model=model, recover_prb=recover_prb)
            psi_e, prb_e, dpsi_e, gradpsi_e, test_psi, fx_e = result_e['psi'], result_e['prb'], result_e['dpsi'], result_e['gradpsi'], result_e['testpsi'], result_e['f']

            # line search
            step_length = 1
            step_shrink = 0.5
            m = 0
            gammapsi_e = step_length
            fdx_e = slv.minf_batch(
                data_e, psi_e, scan_e, prb_e, first, dpsi_e, gradpsi_e, test_psi,  gammapsi_e, piter=1, model=model, recover_prb=recover_prb)
            while fdx_e > fx_e + step_shrink * m:
                if step_length < 1e-32:
                    warnings.warn("Line search failed for conjugate gradient.")
                    step_length = 0
                    break
                step_length *= step_shrink
                gammapsi_e = step_length
                fdx_e = slv.minf_batch(
                    data_e, psi_e, scan_e, prb_e, first, dpsi_e, gradpsi_e, test_psi,  gammapsi_e, piter=1, model=model, recover_prb=recover_prb)

            gammapsi_e = step_length
            print("sbgamma:", gammapsi_e)
            result_e = slv.update_batch(
                data_e, psi_e, scan_e, prb_e, first, dpsi_e, gradpsi_e, test_psi,  gammapsi_e, piter=1, model=model, recover_prb=recover_prb)
            psi_e, prb_e, dpsi_e, gradpsi_e, gammapsi_e = result_e['psi'], result_e['prb'], result_e['dpsi'], result_e['gradpsi'], result_e['gammapsi']
            print(i, file=file_iter)
            t3 = perf_counter()
            time += t3-t2
            print(time, file=file_time)
            if synth == True:
                print(la.norm(np.angle(cp.asnumpy(psi_e[0]))-np.angle(psi_e_cpu[0])), file=file_norm)
                psi_e_cpu = cp.asnumpy(psi_e)
                if ds_name == 'siemens':
                    print(ssim(np.angle(psi_e_cpu[0])-np.mean(np.angle(psi_e_cpu[0][100:200, 100:200])), np.angle(psi_ref[0])), file=file_ssim)
                elif ds_name == 'coins':
                    print(ssim(np.angle(psi_e_cpu[0])-np.mean(np.angle(psi_e_cpu[0][630:730, 350:450])-np.angle(psi_ref[0][630:730, 350:450])), np.angle(psi_ref[0])), file=file_ssim)
            else:
                print(la.norm(np.angle(cp.asnumpy(psi_e[0]))-np.angle(psi_e_cpu[0])), file=file_norm)
                psi_e_cpu = cp.asnumpy(psi_e)
            #if first == True:
            #    dpsi_e = -gradpsi_e.copy()
            #else:
            #    dpsi_e = -gradpsi_e + (
            #        cp.linalg.norm(gradpsi_e)**2 /
            #        (cp.sum(cp.conj(dpsi0_e) * (gradpsi_e - gradpsi0_e))) * dpsi0_e)
            #dpsi0_e = dpsi_e.copy()
            #gradpsi0_e = gradpsi_e.copy()
            ##print("tesett:",cp.where((dpsi_et - test_psi)!=0+0j)[0].size)
            #psi_e += gammapsi_e * dpsi_e
            #if i > 32:
            #    name = str(gpu_count)+'gradpsi_rec'+str(idd)+str(model)+str(i)
            #    psi_e_cpu = cp.asnumpy(psi_e)
            #    dxchange.write_tiff(np.angle(psi_e_cpu),
            #                        'rec_pillar/'+name+'/psiangle_e', overwrite=True)
            #    dxchange.write_tiff(np.abs(psi_e_cpu),  'rec_pillar/'+name+'/psiamp_e', overwrite=True)
        del slv
        del scan_e
        del data_e
        #print("Elapsed time during the GPU running in seconds:",
        #                                                t3-t2, file=outfile)
        name = str(gpu_count)+'psi_rec'+str(idd)+str(model)+str(pit)
        psi_e_cpu = cp.asnumpy(psi_e)
        if synth == True:
            dxchange.write_tiff(np.angle(psi_e_cpu),
                                'rec_siemens/'+name+'/psiangle_e', overwrite=True)
            dxchange.write_tiff(np.abs(psi_e_cpu),  'rec_siemens/'+name+'/psiamp_e', overwrite=True)
            dxchange.write_tiff(np.angle(psi_ref),
                                'rec_siemens/'+name+'/refangle_e', overwrite=True)
            dxchange.write_tiff(np.abs(psi_ref),  'rec_siemens/'+name+'/refamp_e', overwrite=True)
        else:
            dxchange.write_tiff(np.angle(psi_e_cpu),
                                'rec_pillar/'+name+'/psiangle_e', overwrite=True)
            dxchange.write_tiff(np.abs(psi_e_cpu),  'rec_pillar/'+name+'/psiamp_e', overwrite=True)
        print("test_psi:", psi_e_cpu.shape, psi_e_cpu[:,100:102,100:102])
        print("test_dpsi:", cp.asnumpy(dpsi_e)[:,10:20,10:20])
        print("test_gradpsi0:", cp.asnumpy(gradpsi0_e)[:,10:20,10:20])
        del dpsi_e
        del gradpsi0_e

    else:

        def _pin_memory(array):
            mem = cp.cuda.alloc_pinned_memory(array.nbytes)
            ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
            ret[...] = array
            return ret

        #scan_cpu = np.ones([2, ntheta, nscan], dtype='float32')
        #scan_cpu[:, 0] = np.load('model/coords.npy')[:, :nscan].astype('float32')
        xmax = np.amax(scan_cpu[0])
        ymax = np.amax(scan_cpu[1])
        print("x=%f, y=%f" % (xmax, ymax))
        boarder = np.array([[0, 0], [0, ymax/2], [xmax/2, 0], [xmax/2, ymax/2]])
        test1 = np.ones([2, 1], dtype='float32')
        sizes = np.zeros([gpu_count], dtype='int32')
        for e in range(nscan):
            xtileid = scan_cpu[0, 0, e] // (xmax/(gpu_count//2)) - int(scan_cpu[0, 0, e] != 0 and scan_cpu[0, 0, e] % (xmax/(gpu_count//2)) == 0)
            ytileid = scan_cpu[1, 0, e] // (ymax/2) - int(scan_cpu[1, 0, e] != 0 and scan_cpu[1, 0, e] % (ymax/2) == 0)
            sizes[int(xtileid)*2+int(ytileid)] += 1
            #for j in range(gpu_count):
            #    if j%2 == 0 and e[1] <= ymax/2:
            #        if e[0] <= xmax/(gpu_count//2):
            #            sizes[0] += 1
            #        else:
            #        if xmax/(gpu_count//2)*(j//2) < e[0] <= xmax/(gpu_count//2)*(j//2+1) and ymax/2*(j%2) < e[1] <= ymax/2*(j%2+1):
            #            sizes[j] += 1
            #            break

            #    else if j%2 == 1 and e[1] > ymax/2:

            #if e[0] <= xmax/(gpu_count//2) and e[1] <= ymax/2:
            #    sizes[0] += 1
            #else:
            #        if xmax/(gpu_count//2)*(j//2) < e[0] <= xmax/(gpu_count//2)*(j//2+1) and ymax/2*(j%2) < e[1] <= ymax/2*(j%2+1):
            #            sizes[j] += 1
            #            break
            test1 = e

        print(sizes)
        x_rint = np.rint(xmax/2)
        y_rint = np.rint(ymax/2)
        x_rint = x_rint.astype(int)
        y_rint = y_rint.astype(int)
        print("xr=%f, yr=%f" % (x_rint, y_rint))

        counters = np.zeros([gpu_count], dtype='int32')
        times = np.zeros([gpu_count], dtype='float32')
        copy_done = np.zeros([gpu_count, 2], dtype='int32')
        copy_area1 = np.ones([gpu_count, (y_rint + nprb), nprb], dtype='complex64')
        copy_area2 = np.ones([gpu_count, nprb, (x_rint + nprb)], dtype='complex64')
        copy_dpsi1 = np.ones([gpu_count, (y_rint + nprb), nprb], dtype='complex64')
        copy_dpsi2 = np.ones([gpu_count, nprb, (x_rint + nprb)], dtype='complex64')
        copy_gradpsi1 = np.ones([gpu_count, (y_rint + nprb), nprb], dtype='complex64')
        copy_gradpsi2 = np.ones([gpu_count, nprb, (x_rint + nprb)], dtype='complex64')
        copy_delta1 = np.ones([gpu_count, y_rint, nprb], dtype='complex64')
        copy_delta2 = np.ones([gpu_count, nprb, x_rint], dtype='complex64')
        copy_delta3 = np.ones([gpu_count//2, nprb, nprb], dtype='complex64')
        copy_done_pinned = _pin_memory(copy_done)
        copy_area1_pinned = _pin_memory(copy_area1)
        copy_area2_pinned = _pin_memory(copy_area2)
        copy_dpsi1_pinned = _pin_memory(copy_dpsi1)
        copy_dpsi2_pinned = _pin_memory(copy_dpsi2)
        copy_gradpsi1_pinned = _pin_memory(copy_gradpsi1)
        copy_gradpsi2_pinned = _pin_memory(copy_gradpsi2)
        copy_delta1_pinned = _pin_memory(copy_delta1)
        copy_delta2_pinned = _pin_memory(copy_delta2)
        copy_delta3_pinned = _pin_memory(copy_delta3)

        events = [None] * gpu_count * 2
        streams = [None] * gpu_count * 2

        for i in range(gpu_count):
            with cp.cuda.Device(i):
                events[i] = cp.cuda.Event(True, True, True)
                events[i+gpu_count] = cp.cuda.Event(True, True, True)
                streams[i] = cp.cuda.Stream()
                streams[i+gpu_count] = cp.cuda.Stream()


        def data_comm(psi, gpu_id):
            print("start:", psi[0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb].shape)
            print("start2:", psi[0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb].shape)
            #stream = cp.cuda.Stream()
            #stream.use()
            #with cp.cuda.Stream() as stream_out:
            #with streams[gpu_id]:
            copy_area1_pinned[gpu_id] = cp.asnumpy(psi[0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb])
            #copy_area1_pinned[gpu_id] = cp.asnumpy(cp.zeros([(y_rint + nprb), nprb], dtype='complex64'))
            streams[gpu_id].synchronize()
            events[gpu_id].record()
            copy_area2_pinned[gpu_id] = cp.asnumpy(psi[0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb])
            streams[gpu_id].synchronize()
            events[gpu_id+gpu_count].record()
            #cp.cuda.runtime.deviceSynchronize()
            #gpu_done = cp.ones([1, 2], dtype='int32')
            #copy_done_pinned[gpu_id] = cp.asnumpy(gpu_done)
            cp.cuda.runtime.deviceSynchronize()
            streams[gpu_id].record()
            if gpu_id == 2:
                print("area12:", copy_area1_pinned[gpu_id])

            with streams[gpu_id+gpu_count]:
            #with cp.cuda.Stream() as stream_in:
                #while copy_done_pinned[(gpu_id+2)%4][0] == 0 or copy_done_pinned[(5-gpu_id)%4][1] == 0:
                #    #print ("gpu0:", gpu_id,  copy_done[gpu_id+2][0], copy_done[gpu_id+1][1])
                #    pass
                #cp.cuda.runtime.deviceSynchronize()
                events[(gpu_id+2)%4].synchronize()
                print("id_in:", gpu_id, copy_done_pinned)
                if gpu_id == 0:
                    print("area1:", copy_area1_pinned[(gpu_id+2)%4])
                psi[0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb] += cp.asarray(copy_area1_pinned[(gpu_id+2)%4])
                psi[0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb] /= 2
                print("x0=", gpu_id, (psi!=(1+0j)).nonzero()[0].size)
                #print("x0=", psi)
                events[((5-gpu_id)%4)+gpu_count].synchronize()
                psi[0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb] += cp.asarray(copy_area2_pinned[(5-gpu_id)%4])
                psi[0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb] /= 2
                #copy_done_pinned[(gpu_id+2)%4][0] = 0
                #copy_done_pinned[(5-gpu_id)%4][1] = 0
                streams[gpu_id+gpu_count].synchronize()
            #stream_out.synchronize()

            cp.cuda.Device().synchronize()
            print("id_out:", gpu_id, copy_done_pinned)

        if synth == True:
            psi = cp.zeros([1, psi_ref.shape[1], psi_ref.shape[2]], dtype='complex64', order='C')+1
            psi_multi_cpu = np.zeros([gpu_count, 1, psi_ref.shape[1], psi_ref.shape[2]], dtype='complex64', order='C')+1
        else:
            psi = cp.zeros([1, 768+256, 1024+256], dtype='complex64', order='C')+1
            psi_multi_cpu = np.zeros([gpu_count, 1, 768+256, 1024+256], dtype='complex64', order='C')+1

        #copy_grad = np.zeros([gpu_count, (y_rint + nprb), (x_rint + nprb)], dtype='complex64')
        copy_grad = np.zeros([gpu_count, ntheta, nz, n], dtype='complex64')
        grad = np.zeros([ntheta, nz, n], dtype='complex64')
        copy_grad_pinned = _pin_memory(copy_grad)
        grad_pinned = _pin_memory(grad)

        scan = [None] * gpu_count
        data = [None] * gpu_count
        psi = [None] * gpu_count
        dpsi = [None] * gpu_count
        gammapsi = [None] * gpu_count
        gradpsi0 = [None] * gpu_count
        gradpsi = [None] * gpu_count
        test_psi = [None] * gpu_count
        prb = [None] * gpu_count
        slvs = [None] * gpu_count

        fx = np.zeros([gpu_count], dtype='float32')
        fdx = np.zeros([gpu_count], dtype='float32')
        step_length = 1
        step_shrink = 0.5
        m = 0

        for i in range(gpu_count):
            with cp.cuda.Device(i):
                # read scan positions
                size = sizes.item(i)
                scan[i] = cp.ones([2, ntheta, size], dtype='float32')
                data[i]  = cp.zeros([1, size, nprb, nprb], dtype='float32')
                #psi = cp.ones([ntheta, nz, n], dtype='complex64')
                psi[i]  = cp.asarray(psi_cpu)
                dpsi[i]  = cp.zeros([ntheta, nz, n], dtype='complex64')
                #deltapsi[i]  = cp.zeros([ntheta, nz, n], dtype='complex64')
                gradpsi0[i] = cp.zeros([ntheta, nz, n], dtype='complex64')
                gradpsi[i] = cp.zeros([ntheta, nz, n], dtype='complex64')
                test_psi[i] = cp.zeros([ntheta, nz, n], dtype='complex64')
                gammapsi[i] = step_length
                #if (recover_prb):
                #    # Choose an adequate probe approximation
                #    prb = prb0.copy().swapaxes(1, 2)
                #else:
                #    prb = prb0.copy()
                prb[i] = cp.asarray(prb_cpu)

        def multiGPU_init(gpu_id):
            with cp.cuda.Device(gpu_id):
                for e in range(nscan):
                    xgpuid = scan_cpu[0, 0, e] // (xmax/(gpu_count//2)) - int(scan_cpu[0, 0, e] != 0 and scan_cpu[0, 0, e] % (xmax/(gpu_count//2)) == 0)
                    ygpuid = scan_cpu[1, 0, e] // (ymax/2) - int(scan_cpu[1, 0, e] != 0 and scan_cpu[1, 0, e] % (ymax/2) == 0)
                    if (xgpuid*2+ygpuid) == gpu_id:
                        scan[gpu_id][:, 0, counters[gpu_id]] = cp.asarray(scan_cpu[:, 0, e])
                        data[gpu_id][:, counters[gpu_id]] = cp.asarray(data_cpu[:, e])
                        cp.cuda.Device().synchronize()
                        counters[gpu_id] += 1
                    #if gpu_id == 0:
                    #    if e[0] <= xmax/(gpu_count//2) and e[1] <= ymax/2:
                    #        scan[gpu_id][:, 0, counters[gpu_id]] = cp.asarray(e)
                    #        data[gpu_id][:, counters[gpu_id]] = cp.asarray(data_cpu[:, c])
                    #        cp.cuda.Device().synchronize()
                    #        counters[gpu_id] += 1
                    #else:
                    #    if xmax/(gpu_count//2)*(gpu_id//2) < e[0] <= xmax/(gpu_count//2)*(gpu_id//2+1) and ymax/2*(gpu_id%2) < e[1] <= ymax/2*(gpu_id%2+1):
                    #        scan[gpu_id][:, 0, counters[gpu_id]] = cp.asarray(e)
                    #        data[gpu_id][:, counters[gpu_id]] = cp.asarray(data_cpu[:, c])
                    #        cp.cuda.Device().synchronize()
                    #        counters[gpu_id] += 1



                    #if gpu_id == 1 and e[0] <= xmax/2 and e[1] > ymax/2:
                    #    scan[gpu_id][:, 0, counters[gpu_id]] = cp.asarray(e)
                    #    data[gpu_id][:, counters[gpu_id]] = cp.asarray(data_cpu[:, c])
                    #    cp.cuda.Device().synchronize()
                    #    counters[gpu_id] += 1
                    #if gpu_id == 2 and e[0] > xmax/2 and e[1] <= ymax/2:
                    #    scan[gpu_id][:, 0, counters[gpu_id]] = cp.asarray(e)
                    #    data[gpu_id][:, counters[gpu_id]] = cp.asarray(data_cpu[:, c])
                    #    cp.cuda.Device().synchronize()
                    #    counters[gpu_id] += 1
                    #if gpu_id == 3 and e[0] > xmax/2 and e[1] > ymax/2:
                    #    scan[gpu_id][:, 0, counters[gpu_id]] = cp.asarray(e)
                    #    data[gpu_id][:, counters[gpu_id]] = cp.asarray(data_cpu[:, c])
                    #    cp.cuda.Device().synchronize()
                    #    counters[gpu_id] += 1
                cp.cuda.Device().synchronize()
                slvs[gpu_id] = pt.CGPtychoSolver(sizes.item(gpu_id), nprb, ndetx, ndety, ntheta, nz, n, ptheta)
                #import matplotlib.pyplot as plt
                #if gpu_id == 1:
                #    x, y = cp.asnumpy(scan)[0], cp.asnumpy(scan)[1]
                #    plt.scatter(x,y, s=1)
                #    plt.savefig('test2.png', dpi=600)

        def multiGPU_grad(gpu_id):
            with cp.cuda.Device(gpu_id):
                # Class gpu solver
                #with stream0:
                time = cp.zeros([1], dtype='float32')

                start_event = cp.cuda.Event()
                stop_event = cp.cuda.Event()

                with streams[gpu_id]:
                    start_event.record()
                    result = slvs[gpu_id].grad_batch(
                        data[gpu_id], psi[gpu_id], scan[gpu_id], prb[gpu_id], first, dpsi[gpu_id], gradpsi[gpu_id], test_psi[gpu_id], gammapsi[gpu_id], piter=1, model=model, recover_prb=recover_prb)
                    psi[gpu_id], prb[gpu_id], dpsi[gpu_id], gradpsi[gpu_id], test_psi[gpu_id], gammapsi[gpu_id] = result['psi'], result['prb'], result['dpsi'], result['gradpsi'], result['testpsi'], result['gammapsi']
                    #if gpu_id == 0:
                    #    psi_cpu = cp.asnumpy(psi[0])
                    #    print("x0=", (psi_cpu!=(1+0j)).nonzero()[0].size)
                    #    print("y0=", (psi_cpu!=(1+0j)).nonzero()[1].size)
                    #print("shape1:", psi[0, y_rint:y_rint*2+nprb, x_rint:x_rint+nprb].shape)
                    #print("shape2:", psi[0, y_rint:y_rint+nprb, :x_rint+nprb].shape)
                    #print("shape3:", copy_area1[gpu_id].shape)
                    #print("shape4:", copy_area2[gpu_id].shape)
                    #copy_out(psi, gpu_id, xmax, ymax)
                    #copy_in(psi, gpu_id, xmax, ymax)
                    #data_comm(psi, gpu_id)
                    stop_event.record()
                    stop_event.synchronize()
                    time += cp.cuda.get_elapsed_time(start_event, stop_event)/1000
                    cp.cuda.Device().synchronize()

                    #copy_dpsi1_pinned[gpu_id] = cp.asnumpy(dpsi[gpu_id][0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb])
                    #copy_dpsi2_pinned[gpu_id] = cp.asnumpy(dpsi[gpu_id][0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb])
                    #copy_grad_pinned[gpu_id] = cp.asnumpy(gradpsi[gpu_id][0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb])
                    copy_grad_pinned[gpu_id] = cp.asnumpy(gradpsi[gpu_id])
                    #cp.cuda.runtime.deviceSynchronize()
                    #gpu_done = cp.ones([1, 2], dtype='int32')
                    #copy_done_pinned[gpu_id] = cp.asnumpy(gpu_done)
                    cp.cuda.Device().synchronize()
                    #if gpu_id == 0:
                    #    psi_cpu = cp.asnumpy(psi[0])
                    #    print("x:", psi_cpu[np.where(psi_cpu!=(1+0j))])
                    #    print("x=", (psi_cpu!=(1+0j)).nonzero()[0].size)
                    #    print("y=", (psi_cpu!=(1+0j)).nonzero()[1].size)
                #times[gpu_id] = cp.asnumpy(time)

                    #if gpu_id == 0:
                    #    print("x=", (cp.asnumpy(psi[0])!=(1+0j)).nonzero()[0].size)
                    #    print("x:", (cp.asnumpy(psi[0])!=(1+0j)).nonzero()[0].tolist())
                    #    print("y=", (cp.asnumpy(psi[0])!=(1+0j)).nonzero()[1].size)
                    #    print("y:", (cp.asnumpy(psi[0])!=(1+0j)).nonzero()[1].tolist())
                    #psi, prb = result['psi'], result['prb']
                #stream.synchronize()
                return result
                    #cp.cuda.runtime.streamDestroy(0)

        def multiGPU_dir(gpu_id):
            with cp.cuda.Device(gpu_id):

                gradpsi[gpu_id] = cp.asarray(grad_pinned)
                cp.cuda.Device().synchronize()

                with streams[gpu_id]:
                    result = slvs[gpu_id].dir_batch(
                        data[gpu_id], psi[gpu_id], scan[gpu_id], prb[gpu_id], first, dpsi[gpu_id], gradpsi[gpu_id], test_psi[gpu_id], gammapsi[gpu_id], piter=1, model=model, recover_prb=recover_prb)
                    psi[gpu_id], prb[gpu_id], dpsi[gpu_id], gradpsi[gpu_id], test_psi[gpu_id], fx[gpu_id] = result['psi'], result['prb'], result['dpsi'], result['gradpsi'], result['testpsi'], result['f']
                    print("sbfx:", gpu_id, fx[gpu_id])
                    cp.cuda.Device().synchronize()

                return result
                    #cp.cuda.runtime.streamDestroy(0)

        def multiGPU_minf(gpu_id):
            with cp.cuda.Device(gpu_id):
                # Class gpu solver

                with streams[gpu_id]:
                    fdx[gpu_id] = slvs[gpu_id].minf_batch(
                        data[gpu_id], psi[gpu_id], scan[gpu_id], prb[gpu_id], first, dpsi[gpu_id], gradpsi[gpu_id], test_psi[gpu_id], gammapsi[gpu_id], piter=1, model=model, recover_prb=recover_prb)
                    cp.cuda.Device().synchronize()

                return fdx[gpu_id]

        def multiGPU_update(gpu_id):
            with cp.cuda.Device(gpu_id):
                #deltapsi[gpu_id][0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb] = cp.asarray(copy_delta1_pinned[gpu_id%2])
                #deltapsi[gpu_id][0, y_rint*(gpu_id%2):y_rint*(gpu_id%2+1)+nprb, x_rint:x_rint+nprb] /= 2
                #deltapsi[gpu_id][0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb] = cp.asarray(copy_delta2_pinned[gpu_id//2])
                #deltapsi[gpu_id][0, y_rint:y_rint+nprb, x_rint*(gpu_id//2):x_rint*(gpu_id//2+1)+nprb] /= 2
                #dpsi[gpu_id][0, (y_rint+nprb)*(gpu_id%2):(y_rint+nprb)*(gpu_id%2)+y_rint, x_rint:x_rint+nprb] = cp.asarray(copy_delta1_pinned[gpu_id%2])
                #dpsi[gpu_id][0, y_rint:y_rint+nprb, (x_rint+nprb)*(gpu_id//2):(x_rint+nprb)*(gpu_id//2)+x_rint] = cp.asarray(copy_delta2_pinned[gpu_id//2])
                #dpsi[gpu_id][0, y_rint:y_rint+nprb, x_rint:x_rint+nprb] = cp.asarray(copy_delta3_pinned[0])

                #gradpsi[gpu_id][0, (y_rint+nprb)*(gpu_id%2):(y_rint+nprb)*(gpu_id%2)+y_rint, x_rint:x_rint+nprb] = cp.asarray(copy_delta1_pinned[gpu_id%2])
                #gradpsi[gpu_id][0, y_rint:y_rint+nprb, (x_rint+nprb)*(gpu_id//2):(x_rint+nprb)*(gpu_id//2)+x_rint] = cp.asarray(copy_delta2_pinned[gpu_id//2])
                #gradpsi[gpu_id][0, y_rint:y_rint+nprb, x_rint:x_rint+nprb] = cp.asarray(copy_delta3_pinned[0])

                #gammapsi[gpu_id] = 0
                result = slvs[gpu_id].update_batch(
                    data[gpu_id], psi[gpu_id], scan[gpu_id], prb[gpu_id], first, dpsi[gpu_id], gradpsi[gpu_id], test_psi[gpu_id], gammapsi[gpu_id], piter=1, model=model, recover_prb=recover_prb)
                psi[gpu_id], prb[gpu_id], dpsi[gpu_id], gradpsi[gpu_id], gammapsi[gpu_id] = result['psi'], result['prb'], result['dpsi'], result['gradpsi'], result['gammapsi']
                #if first == True:
                #    dpsi[gpu_id] = -gradpsi[gpu_id].copy()
                #else:
                #    dpsi[gpu_id] = -gradpsi[gpu_id] + (
                #        cp.linalg.norm(gradpsi[gpu_id])**2 /
                #        (cp.sum(cp.conj(dpsi[gpu_id]) * (gradpsi[gpu_id] - gradpsi0[gpu_id]))) * dpsi[gpu_id])
                #gradpsi0[gpu_id] = gradpsi[gpu_id].copy()
                #print("gammapsi:", gpu_id, gammapsi[gpu_id])
                #print("teset:", cp.where(psi[gpu_id]!=1+0j)[0].size)
                #psi[gpu_id] += gammapsi[gpu_id]*dpsi[gpu_id]
                cp.cuda.Device().synchronize()
                psi_multi_cpu[gpu_id] = cp.asnumpy(psi[gpu_id])
                print("cjsb:", gpu_id, gammapsi[gpu_id])
                #print("test_psi_multi:", gpu_id,  psi_multi_cpu.shape, psi_multi_cpu[gpu_id,:,100:102,100:102])
                #print("test_dpsi_multi:", gpu_id, cp.asnumpy(dpsi[gpu_id])[:,10:20,10:20])
                #print("test_gradpsi0_multi:", gpu_id, cp.asnumpy(gradpsi0[gpu_id])[:,10:20,10:20])

        def multiGPU_destruct(gpu_id):
            with cp.cuda.Device(gpu_id):
                del slvs[gpu_id]

        gpu_list = range(gpu_count)

        if synth == True:
            psi_m_cpu = np.zeros([1, psi_ref.shape[1], psi_ref.shape[2]], dtype='complex64', order='C')+1
        else:
            psi_m_cpu = np.zeros([1, 768+256, 1024+256], dtype='complex64', order='C')+1

        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            results = executor.map(multiGPU_init, gpu_list)
            #results = list(results)
        first= True
        time = 0.0
        for i in range(pit):
            t2 = perf_counter()
            if i == 0:
                first = True
            else:
                first = False
            fx.fill(0.0)
            print("sbfx:", fx[0])
            with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
                results = executor.map(multiGPU_grad, gpu_list)
                results = list(results)

            grad_pinned.fill(0+0j)
            for j in range(gpu_count):
                grad_pinned += copy_grad_pinned[j]

            with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
                results = executor.map(multiGPU_dir, gpu_list)
                results = list(results)
            fx_g = np.cumsum(fx)[gpu_count-1]

            #line search
            fdx.fill(0.0)
            step_length = 1
            step_shrink = 0.5
            for j in range(gpu_count):
                gammapsi[j] = step_length
            with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
                results = executor.map(multiGPU_minf, gpu_list)
            fdx_g = np.cumsum(fdx)[gpu_count-1]
            while fdx_g > fx_g + step_shrink * m:
                print("sbls:", fdx_g, fx_g + step_shrink * m)
                if step_length < 1e-32:
                    warnings.warn("Line search failed for conjugate gradient.")
                    step_length = 0
                    break
                step_length *= step_shrink
                for j in range(gpu_count):
                    gammapsi[j] = step_length
                fdx.fill(0.0)
                with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
                    results = executor.map(multiGPU_minf, gpu_list)
                fdx_g = np.cumsum(fdx)[gpu_count-1]

            print("sbgamma:", gammapsi[0])
            print("cjsbi:", i)
            with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
                results = executor.map(multiGPU_update, gpu_list)
                #results = list(results)

            t3 = perf_counter()
            time += t3-t2
            print(time, file=file_time)
            print(i, file=file_iter)
            if synth == True:
                print(la.norm(np.angle(psi_multi_cpu[0])-np.angle(psi_m_cpu)), file=file_norm)
                psi_m_cpu = psi_multi_cpu[0].copy()
                if ds_name == 'siemens':
                    print(ssim(np.angle(psi_m_cpu[0])-np.mean(np.angle(psi_m_cpu[0][100:200, 100:200])), np.angle(psi_ref[0])), file=file_ssim)
                elif ds_name == 'coins':
                    print(ssim(np.angle(psi_m_cpu[0])-np.mean(np.angle(psi_m_cpu[0][630:730, 350:450])-np.angle(psi_ref[0][630:730, 350:450])), np.angle(psi_ref[0])), file=file_ssim)
            else:
                print(la.norm(np.angle(psi_multi_cpu[0])-np.angle(psi_m_cpu)), file=file_norm)
                psi_m_cpu = psi_multi_cpu[0].copy()
            #if i > 32:
            #    psi_cpu = np.zeros([1, 768+256, 1024+256], dtype='complex64', order='C')+1
            #    for j in range(gpu_count):
            #        psi_cpu[0, y_rint*(j%2):y_rint*(j%2+1)+nprb, x_rint*(j//2):x_rint*(j//2+1)+nprb] = psi_multi_cpu[j][0, y_rint*(j%2):y_rint*(j%2+1)+nprb, x_rint*(j//2):x_rint*(j//2+1)+nprb]
            #    # Save result
            #    name = 'gradpsi_rec'+str(idd)+str(model)+str(i)
            #    dxchange.write_tiff(np.angle(psi_multi_cpu),
            #                        'rec_pillar/'+name+'/psiangle_4_m', overwrite=True)
            #    dxchange.write_tiff(np.abs(psi_multi_cpu),  'rec_pillar/'+name+'/psiamp_4_m', overwrite=True)
            #    dxchange.write_tiff(np.angle(psi_cpu),
            #                        'rec_pillar/'+name+'/psiangle_4', overwrite=True)
            #    dxchange.write_tiff(np.abs(psi_cpu),  'rec_pillar/'+name+'/psiamp_4', overwrite=True)
            dxchange.write_tiff(np.angle(psi_m_cpu),
                                'rec_siemens/128iters/psiangle'+str(i), overwrite=True)
            dxchange.write_tiff(np.abs(psi_m_cpu),  'rec_siemens/128iters/psiamp'+str(i), overwrite=True)

        with cf.ThreadPoolExecutor(max_workers=gpu_count) as executor:
            results = executor.map(multiGPU_destruct, gpu_list)
            #results = list(results)

        # Save result
        name = str(gpu_count)+'psi_rec'+str(idd)+str(model)+str(pit)
        if synth == True:
            dxchange.write_tiff(np.angle(psi_multi_cpu),
                                'rec_siemens/'+name+'/psiangle_4_m', overwrite=True)
            dxchange.write_tiff(np.abs(psi_multi_cpu),  'rec_siemens/'+name+'/psiamp_4_m', overwrite=True)
            dxchange.write_tiff(np.angle(psi_m_cpu),
                                'rec_siemens/'+name+'/psiangle_4', overwrite=True)
            dxchange.write_tiff(np.abs(psi_m_cpu),  'rec_siemens/'+name+'/psiamp_4', overwrite=True)
        else:
            dxchange.write_tiff(np.angle(psi_multi_cpu),
                                'rec_pillar/'+name+'/psiangle_4_m', overwrite=True)
            dxchange.write_tiff(np.abs(psi_multi_cpu),  'rec_pillar/'+name+'/psiamp_4_m', overwrite=True)
            dxchange.write_tiff(np.angle(psi_m_cpu),
                                'rec_pillar/'+name+'/psiangle_4', overwrite=True)
            dxchange.write_tiff(np.abs(psi_m_cpu),  'rec_pillar/'+name+'/psiamp_4', overwrite=True)

        t3 = perf_counter()
        print("Elapsed time during the GPU running in seconds:",
                                                        t3-t2)
        #dxchange.write_tiff(np.angle(psi_cpu),
        #                    'rec_pillar/'+name+'/psiangle', overwrite=True)
        #dxchange.write_tiff(np.abs(psi_cpu),  'rec_pillar/'+name+'/psiamp', overwrite=True)
        #name = str(model)+str(piter)
        #dxchange.write_tiff(cp.angle(psi).get(),
        #                    'rec/psiang'+name, overwrite=True)
        #dxchange.write_tiff(cp.abs(psi).get(),  'rec/prbamp'+name, overwrite=True)

        ## recovered
        #dxchange.write_tiff(cp.angle(prb).get(),
        #                    'rec/prbangle'+name, overwrite=True)
        #dxchange.write_tiff(cp.abs(prb).get(),  'rec/prbamp'+name, overwrite=True)
        ## init
        #dxchange.write_tiff(cp.angle(prb0).get(),
        #                    'rec/prb0angle'+name, overwrite=True)
        #dxchange.write_tiff(cp.abs(prb0).get(),
        #                    'rec/prb0amp'+name, overwrite=True)

        psi_c = np.ones([ntheta, nz, n], dtype='complex64')
        t4 = perf_counter()
        print("Elapsed time during the whole program in seconds:",
                                                        t4-t1)
        print("See result.png and tiff files in rec/ folder")
    file_iter.close()
    file_norm.close()
    file_time.close()
    if synth == True:
        file_ssim.close()
