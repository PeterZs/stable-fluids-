# Stable Fluids

## Pipeline Overview

In the paper, the velocity update is written as

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{P}\left( -(\mathbf{u} \cdot \nabla)\mathbf{u} + \nu \nabla^2 \mathbf{u} + \mathbf{f} \right),
$$

where $\mathbf{P}$ is the projection onto divergence-free fields.

The paper then splits one step into

$$
\mathbf{w}_0 \rightarrow \mathbf{w}_1 \rightarrow \mathbf{w}_2 \rightarrow \mathbf{w}_3 \rightarrow \mathbf{w}_4,
$$

with

1. add force,
2. advect,
3. diffuse,
4. project.

This repository uses the same split, except that the force step is done outside `stable_fluids_step_cuda`. The CUDA step therefore performs

$$
\mathbf{w}_1 \rightarrow \mathbf{w}_2 \rightarrow \mathbf{w}_3 \rightarrow \mathbf{w}_4,
$$

followed by the same advection-diffusion split for the scalar density field.

Field layout:

- `density`: `(nx, ny, nz)`
- `velocity_x`: `(nx + 1, ny, nz)`
- `velocity_y`: `(nx, ny + 1, nz)`
- `velocity_z`: `(nx, ny, nz + 1)`

## Method Details

### 1. Advection

#### Theory

The paper resolves the non-linear term

$$
-(\mathbf{u} \cdot \nabla)\mathbf{u}
$$

with the method of characteristics. For a quantity $a(\mathbf{x}, t)$,

$$
\frac{\partial a}{\partial t} = - \mathbf{u} \cdot \nabla a
$$

is solved by tracing backward:

$$
a(\mathbf{x}, t + \Delta t) = a(\mathbf{p}(\mathbf{x}, -\Delta t), t).
$$

In this implementation, all samples are taken in grid-index coordinates. If $h$ is the cell size, the traced point is

$$
\mathbf{x}_d = \mathbf{x} - \frac{\Delta t}{h}\mathbf{u}(\mathbf{x}).
$$

Velocity is stored on a MAC grid, so the full vector field is reconstructed by

$$
\mathbf{u}(x,y,z) =
\begin{bmatrix}
u(x, y - 0.5, z - 0.5) \\
v(x - 0.5, y, z - 0.5) \\
w(x - 0.5, y - 0.5, z)
\end{bmatrix}.
$$

#### CUDA Implementation

```cpp
__device__ float sample_grid(
    const float* const field, const float x, const float y, const float z, const int nx, const int ny, const int nz) {
    const float px = clampf(x, 0.0f, static_cast<float>(nx - 1));
    const float py = clampf(y, 0.0f, static_cast<float>(ny - 1));
    const float pz = clampf(z, 0.0f, static_cast<float>(nz - 1));
    const int i0 = static_cast<int>(floorf(px));
    const int j0 = static_cast<int>(floorf(py));
    const int k0 = static_cast<int>(floorf(pz));
    const int i1 = min(i0 + 1, nx - 1);
    const int j1 = min(j0 + 1, ny - 1);
    const int k1 = min(k0 + 1, nz - 1);
    const float tx = px - static_cast<float>(i0);
    const float ty = py - static_cast<float>(j0);
    const float tz = pz - static_cast<float>(k0);
    const float c000 = fetch_clamped(field, i0, j0, k0, nx, ny, nz);
    const float c100 = fetch_clamped(field, i1, j0, k0, nx, ny, nz);
    const float c010 = fetch_clamped(field, i0, j1, k0, nx, ny, nz);
    const float c110 = fetch_clamped(field, i1, j1, k0, nx, ny, nz);
    const float c001 = fetch_clamped(field, i0, j0, k1, nx, ny, nz);
    const float c101 = fetch_clamped(field, i1, j0, k1, nx, ny, nz);
    const float c011 = fetch_clamped(field, i0, j1, k1, nx, ny, nz);
    const float c111 = fetch_clamped(field, i1, j1, k1, nx, ny, nz);
    const float c00 = c000 + (c100 - c000) * tx;
    const float c10 = c010 + (c110 - c010) * tx;
    const float c01 = c001 + (c101 - c001) * tx;
    const float c11 = c011 + (c111 - c011) * tx;
    const float c0 = c00 + (c10 - c00) * ty;
    const float c1 = c01 + (c11 - c01) * ty;
    return c0 + (c1 - c0) * tz;
}

__device__ float3 sample_velocity(
    const float* const u, const float* const v, const float* const w, const float3 p, const int nx, const int ny, const int nz) {
    return make_float3(
        sample_u(u, p.x, p.y - 0.5f, p.z - 0.5f, nx, ny, nz),
        sample_v(v, p.x - 0.5f, p.y, p.z - 0.5f, nx, ny, nz),
        sample_w(w, p.x - 0.5f, p.y - 0.5f, p.z, nx, ny, nz));
}

__global__ void advect_u_kernel(float* const dst, const float* const src, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float dt_over_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j >= ny || k >= nz) return;
    const float3 p = make_float3(static_cast<float>(i), static_cast<float>(j) + 0.5f, static_cast<float>(k) + 0.5f);
    const float3 vel = sample_velocity(u, v, w, p, nx, ny, nz);
    const float3 back = clamp_domain(make_float3(p.x - dt_over_h * vel.x, p.y - dt_over_h * vel.y, p.z - dt_over_h * vel.z), nx, ny, nz);
    dst[index_3d(i, j, k, nx + 1, ny)] = sample_u(src, back.x, back.y - 0.5f, back.z - 0.5f, nx, ny, nz);
}

__global__ void advect_v_kernel(float* const dst, const float* const src, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float dt_over_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j > ny || k >= nz) return;
    const float3 p = make_float3(static_cast<float>(i) + 0.5f, static_cast<float>(j), static_cast<float>(k) + 0.5f);
    const float3 vel = sample_velocity(u, v, w, p, nx, ny, nz);
    const float3 back = clamp_domain(make_float3(p.x - dt_over_h * vel.x, p.y - dt_over_h * vel.y, p.z - dt_over_h * vel.z), nx, ny, nz);
    dst[index_3d(i, j, k, nx, ny + 1)] = sample_v(src, back.x - 0.5f, back.y, back.z - 0.5f, nx, ny, nz);
}

__global__ void advect_w_kernel(float* const dst, const float* const src, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float dt_over_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k > nz) return;
    const float3 p = make_float3(static_cast<float>(i) + 0.5f, static_cast<float>(j) + 0.5f, static_cast<float>(k));
    const float3 vel = sample_velocity(u, v, w, p, nx, ny, nz);
    const float3 back = clamp_domain(make_float3(p.x - dt_over_h * vel.x, p.y - dt_over_h * vel.y, p.z - dt_over_h * vel.z), nx, ny, nz);
    dst[index_3d(i, j, k, nx, ny)] = sample_w(src, back.x - 0.5f, back.y - 0.5f, back.z, nx, ny, nz);
}
```

The step routine first copies the current velocity field into the previous-velocity buffers, then advects into the temporary-velocity buffers. In the paper's notation, this is the transition

$$
\mathbf{w}_1 \rightarrow \mathbf{w}_2.
$$

### 2. Diffusion

#### Theory

The paper writes the diffusion step as

$$
\frac{\partial \mathbf{w}_2}{\partial t} = \nu \nabla^2 \mathbf{w}_2.
$$

Backward Euler gives

$$
(I - \nu \Delta t \nabla^2)\mathbf{w}_3 = \mathbf{w}_2.
$$

On a regular grid, one red-black Gauss-Seidel update is

$$
q_{i,j,k}^{m+1} =
\frac{
q_{i,j,k}^{*}
+ a \left(
q_{i-1,j,k}^{m} + q_{i+1,j,k}^{m}
+ q_{i,j-1,k}^{m} + q_{i,j+1,k}^{m}
+ q_{i,j,k-1}^{m} + q_{i,j,k+1}^{m}
\right)
}{
1 + 6a
},
$$

where

$$
a = \frac{\nu \Delta t}{h^2}.
$$

#### CUDA Implementation

```cpp
__global__ void diffuse_grid_kernel(float* const dst, const float* const src, const int nx, const int ny, const int nz, const float alpha, const float denom, const int parity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    if (((i + j + k) & 1) != parity) return;
    const float center = src[index_3d(i, j, k, nx, ny)];
    const float sum = fetch_clamped(dst, i - 1, j, k, nx, ny, nz) + fetch_clamped(dst, i + 1, j, k, nx, ny, nz) +
                      fetch_clamped(dst, i, j - 1, k, nx, ny, nz) + fetch_clamped(dst, i, j + 1, k, nx, ny, nz) +
                      fetch_clamped(dst, i, j, k - 1, nx, ny, nz) + fetch_clamped(dst, i, j, k + 1, nx, ny, nz);
    dst[index_3d(i, j, k, nx, ny)] = (center + alpha * sum) / denom;
}
```

The CUDA step:

1. copies `temporary_velocity_*` into `velocity_*`,
2. sets `alpha = dt * viscosity / (cell_size * cell_size)`,
3. sets `denom = 1 + 6 * alpha`,
4. applies parity `0` then parity `1`,
5. reapplies velocity boundary conditions after each sweep.

This is the paper's transition

$$
\mathbf{w}_2 \rightarrow \mathbf{w}_3.
$$

### 3. Projection

#### Theory

The paper defines the projection by

$$
\mathbf{w} = \mathbf{u} + \nabla q,
$$

where $\mathbf{u}$ has zero divergence. Applying divergence gives the Poisson equation

$$
\nabla^2 q = \nabla \cdot \mathbf{w}.
$$

The projected field is then

$$
\mathbf{u} = \mathbf{P}\mathbf{w} = \mathbf{w} - \nabla q.
$$

For the MAC layout used here,

$$
d_{i,j,k} =
\frac{
u_{i+1,j,k} - u_{i,j,k}
+ v_{i,j+1,k} - v_{i,j,k}
+ w_{i,j,k+1} - w_{i,j,k}
}{h}.
$$

The pressure update is

$$
q_{i,j,k}^{m+1} =
\frac{
q_{i-1,j,k} + q_{i+1,j,k}
+ q_{i,j-1,k} + q_{i,j+1,k}
+ q_{i,j,k-1} + q_{i,j,k+1}
- h^2 d_{i,j,k}
}{6}.
$$

Then subtract the gradient componentwise.

#### CUDA Implementation

```cpp
__global__ void compute_divergence_kernel(float* const divergence, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    divergence[index_3d(i, j, k, nx, ny)] =
        ((u[index_3d(i + 1, j, k, nx + 1, ny)] - u[index_3d(i, j, k, nx + 1, ny)]) +
         (v[index_3d(i, j + 1, k, nx, ny + 1)] - v[index_3d(i, j, k, nx, ny + 1)]) +
         (w[index_3d(i, j, k + 1, nx, ny)] - w[index_3d(i, j, k, nx, ny)])) * inv_h;
}

__global__ void pressure_rbgs_kernel(float* const pressure, const float* const divergence, const int nx, const int ny, const int nz, const float h, const int parity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    if (((i + j + k) & 1) != parity) return;
    const int im1 = clampi(i - 1, 0, nx - 1);
    const int ip1 = clampi(i + 1, 0, nx - 1);
    const int jm1 = clampi(j - 1, 0, ny - 1);
    const int jp1 = clampi(j + 1, 0, ny - 1);
    const int km1 = clampi(k - 1, 0, nz - 1);
    const int kp1 = clampi(k + 1, 0, nz - 1);
    const float sum = pressure[index_3d(im1, j, k, nx, ny)] + pressure[index_3d(ip1, j, k, nx, ny)] +
                      pressure[index_3d(i, jm1, k, nx, ny)] + pressure[index_3d(i, jp1, k, nx, ny)] +
                      pressure[index_3d(i, j, km1, nx, ny)] + pressure[index_3d(i, j, kp1, nx, ny)];
    pressure[index_3d(i, j, k, nx, ny)] =
        (sum - divergence[index_3d(i, j, k, nx, ny)] * h * h) / 6.0f;
}

__global__ void subtract_gradient_u_kernel(float* const u, const float* const pressure, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i == 0 || i >= nx || j >= ny || k >= nz) return;
    u[index_3d(i, j, k, nx + 1, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i - 1, j, k, nx, ny)]) * inv_h;
}

__global__ void subtract_gradient_v_kernel(float* const v, const float* const pressure, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j == 0 || j >= ny || k >= nz) return;
    v[index_3d(i, j, k, nx, ny + 1)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j - 1, k, nx, ny)]) * inv_h;
}

__global__ void subtract_gradient_w_kernel(float* const w, const float* const pressure, const int nx, const int ny, const int nz, const float inv_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k == 0 || k >= nz) return;
    w[index_3d(i, j, k, nx, ny)] -= (pressure[index_3d(i, j, k, nx, ny)] - pressure[index_3d(i, j, k - 1, nx, ny)]) * inv_h;
}
```

The CUDA step:

1. zeros `temporary_pressure`,
2. computes `temporary_divergence`,
3. runs `pressure_iterations` red-black sweeps,
4. subtracts the pressure gradient from `velocity_x`, `velocity_y`, `velocity_z`,
5. reapplies the boundary kernels.

This is the paper's transition

$$
\mathbf{w}_3 \rightarrow \mathbf{w}_4.
$$

### 4. Density Advection

#### Theory

For a scalar quantity $a$, the paper writes

$$
\frac{\partial a}{\partial t} = - \mathbf{u} \cdot \nabla a + \kappa_a \nabla^2 a - \alpha_a a + S_a.
$$

This implementation uses only the advection-diffusion part for density. The advection substep is

$$
\frac{\partial \rho}{\partial t} = - \mathbf{u} \cdot \nabla \rho,
$$

so

$$
\rho^{*}(\mathbf{x}, t + \Delta t) = \rho(\mathbf{p}(\mathbf{x}, -\Delta t), t).
$$

#### CUDA Implementation

```cpp
__global__ void advect_scalar_kernel(float* const dst, const float* const src, const float* const u, const float* const v, const float* const w, const int nx, const int ny, const int nz, const float dt_over_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    const float3 p = make_float3(static_cast<float>(i) + 0.5f, static_cast<float>(j) + 0.5f, static_cast<float>(k) + 0.5f);
    const float3 vel = sample_velocity(u, v, w, p, nx, ny, nz);
    const float3 back = clamp_domain(make_float3(p.x - dt_over_h * vel.x, p.y - dt_over_h * vel.y, p.z - dt_over_h * vel.z), nx, ny, nz);
    dst[index_3d(i, j, k, nx, ny)] = fmaxf(0.0f, sample_scalar(src, back.x - 0.5f, back.y - 0.5f, back.z - 0.5f, nx, ny, nz));
}
```

The CUDA step copies `density` into `temporary_previous_density`, then advects into `temporary_density`.

### 5. Density Diffusion

#### Theory

The diffusion part of the scalar equation is

$$
\frac{\partial \rho}{\partial t} = \kappa \nabla^2 \rho.
$$

Backward Euler gives

$$
(I - \kappa \Delta t \nabla^2)\rho^{n+1} = \rho^{*}.
$$

The discrete relaxation formula is the same as for velocity, with

$$
a = \frac{\kappa \Delta t}{h^2}.
$$

#### CUDA Implementation

The same `diffuse_grid_kernel` is reused for density.

The CUDA step:

1. copies `temporary_density` into `density`,
2. sets `alpha = dt * diffusion / (cell_size * cell_size)`,
3. sets `denom = 1 + 6 * alpha`,
4. runs `diffuse_iterations` red-black sweeps on `density`, using `temporary_density` as the fixed right-hand side.

## Reproduction Checklist

To reproduce this CUDA implementation faithfully, keep the paper's split and the code's buffer flow:

1. external force is applied outside `stable_fluids_step_cuda`,
2. `stable_fluids_step_cuda` performs advection, diffusion, and projection on velocity,
3. then advection and diffusion on density,
4. all advection uses backward characteristic tracing,
5. all diffusion and projection solves use red-black Gauss-Seidel,
6. all velocity operators use the MAC staggered layout,
7. wall conditions set the normal velocity component to zero.
