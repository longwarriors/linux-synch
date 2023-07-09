import cuquantum
import cupy as cp

# https://developer.nvidia.com/blog/scaling-quantum-circuit-simulation-with-cutensornet/
# Compute D_{m,x,n,y} = A_{m,h,k,n} B_{u,k,h} C_{x,u,y}
# Create an array of extents (shapes) for each tensor
extentA = (96, 64, 64, 96)
extentB = (96, 64, 64)
extentC = (64, 96, 64)
extentD = (96, 64, 96, 64)

# Generate input tensor data directly on GPU
A_d = cp.random.random(extentA, dtype=cp.float32)
B_d = cp.random.random(extentB, dtype=cp.float32)
C_d = cp.random.random(extentC, dtype=cp.float32)

# Set the pathfinder options
options = cuquantum.OptimizerOptions()
options.slicing.disable_slicing = 1  # disable slicing
options.samples = 100  # number of hyper-optimizer samples

# Run the contraction on a CUDA stream
stream = cp.cuda.Stream()
D_d, info = cuquantum.contract(
    "mhkn,ukh,xuy->mxny",
    A_d,
    B_d,
    C_d,
    optimize=options,
    stream=stream,
    return_info=True,
)
stream.synchronize()

print(D_d)
