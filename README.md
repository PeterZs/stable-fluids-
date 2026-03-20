# Stable Fluids

## Pipeline Overview

One step of this implementation is

$$
\mathbf{u}^n \rightarrow \text{advect} \rightarrow \text{diffuse} \rightarrow \text{project} \rightarrow \mathbf{u}^{n+1},
$$

then

$$
\rho^n \rightarrow \text{advect by } \mathbf{u}^{n+1} \rightarrow \text{diffuse} \rightarrow \rho^{n+1}.
$$

Fields:

- `density`: `(nx, ny, nz)`
- `velocity_x`: `(nx + 1, ny, nz)`
- `velocity_y`: `(nx, ny + 1, nz)`
- `velocity_z`: `(nx, ny, nz + 1)`

The velocity layout is MAC staggered. Density is cell-centered.

## Method Details

### 1. Sampling

#### Theory

Stable Fluids uses semi-Lagrangian advection. For a quantity $q$,

$$
\frac{\partial q}{\partial t} + \mathbf{u} \cdot \nabla q = 0
$$

is updated by backward tracing:

$$
q^{n+1}(\mathbf{x}) = q^n(\mathbf{x} - \Delta t \, \mathbf{u}(\mathbf{x})).
$$

This implementation samples in grid-index coordinates, so the traced point is

$$
\mathbf{x}_d = \mathbf{x} - \frac{\Delta t}{h} \mathbf{u}(\mathbf{x}),
$$

where $h$ is `cell_size`.

Scalar sampling uses trilinear interpolation on the cell-centered grid. Velocity sampling reconstructs a collocated vector from the staggered MAC components:

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
__device__ std::size_t index_3d(const int i, const int j, const int k, const int nx, const int ny) {
    return static_cast<std::size_t>((k * ny + j) * nx + i);
}

__device__ float fetch_clamped(
    const float* const field, const int i, const int j, const int k, const int nx, const int ny, const int nz) {
    return field[index_3d(clampi(i, 0, nx - 1), clampi(j, 0, ny - 1), clampi(k, 0, nz - 1), nx, ny)];
}

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

__device__ float sample_scalar(const float* const field, const float x, const float y, const float z, const int nx, const int ny, const int nz) {
    return sample_grid(field, x, y, z, nx, ny, nz);
}

__device__ float sample_u(const float* const field, const float x, const float y, const float z, const int nx, const int ny, const int nz) {
    return sample_grid(field, x, y, z, nx + 1, ny, nz);
}

__device__ float sample_v(const float* const field, const float x, const float y, const float z, const int nx, const int ny, const int nz) {
    return sample_grid(field, x, y, z, nx, ny + 1, nz);
}

__device__ float sample_w(const float* const field, const float x, const float y, const float z, const int nx, const int ny, const int nz) {
    return sample_grid(field, x, y, z, nx, ny, nz + 1);
}

__device__ float3 sample_velocity(
    const float* const u, const float* const v, const float* const w, const float3 p, const int nx, const int ny, const int nz) {
    return make_float3(
        sample_u(u, p.x, p.y - 0.5f, p.z - 0.5f, nx, ny, nz),
        sample_v(v, p.x - 0.5f, p.y, p.z - 0.5f, nx, ny, nz),
        sample_w(w, p.x - 0.5f, p.y - 0.5f, p.z, nx, ny, nz));
}
```

This code implements:

- x-fastest linear indexing,
- clamped trilinear interpolation,
- MAC-to-collocated velocity reconstruction.

### 2. Boundary Conditions

#### Theory

The wall condition is zero normal velocity on the domain boundary:

- $u = 0$ on the two $x$ walls,
- $v = 0$ on the two $y$ walls,
- $w = 0$ on the two $z$ walls.

#### CUDA Implementation

```cpp
__global__ void set_u_boundary_kernel(float* const u, const int nx, const int ny, const int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx) u[index_3d(i, j, k, nx + 1, ny)] = 0.0f;
}

__global__ void set_v_boundary_kernel(float* const v, const int nx, const int ny, const int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j > ny || k >= nz) return;
    if (j == 0 || j == ny) v[index_3d(i, j, k, nx, ny + 1)] = 0.0f;
}

__global__ void set_w_boundary_kernel(float* const w, const int nx, const int ny, const int nz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k > nz) return;
    if (k == 0 || k == nz) w[index_3d(i, j, k, nx, ny)] = 0.0f;
}
```

Each kernel enforces one normal component.

### 3. Velocity Advection

#### Theory

Velocity advection solves

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = 0
$$

with semi-Lagrangian backtracing.

For the x-face component, the geometric sample point is

$$
\mathbf{x}_u = (i, j + 0.5, k + 0.5).
$$

Then

$$
\mathbf{x}_d = \mathbf{x}_u - \frac{\Delta t}{h} \mathbf{u}(\mathbf{x}_u),
$$

and the new x-face value is

$$
u^*(i,j,k) = u^n(\mathbf{x}_d).
$$

The y-face and z-face updates are the same, but centered at

$$
\mathbf{x}_v = (i + 0.5, j, k + 0.5),
$$

$$
\mathbf{x}_w = (i + 0.5, j + 0.5, k).
$$

#### CUDA Implementation

```cpp
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

The step routine uses these kernels in this order:

1. copy current velocity to the `temporary_previous_velocity_*` fields,
2. advect into `temporary_velocity_*`,
3. apply the three boundary kernels to `temporary_velocity_*`.

### 4. Velocity Diffusion

#### Theory

Velocity diffusion solves

$$
\frac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u}.
$$

Backward Euler gives

$$
(I - \nu \Delta t \nabla^2)\mathbf{u}^{n+1} = \mathbf{u}^*.
$$

With the 7-point stencil, one relaxation update is

$$
q_{i,j,k}^{m+1} =
\frac{
q^*_{i,j,k} + a \left(
q_{i-1,j,k}^m + q_{i+1,j,k}^m +
q_{i,j-1,k}^m + q_{i,j+1,k}^m +
q_{i,j,k-1}^m + q_{i,j,k+1}^m
\right)
}{
1 + 6a
},
$$

where

$$
a = \frac{\nu \Delta t}{h^2}.
$$

The implementation uses red-black Gauss-Seidel.

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

The step routine uses it as follows:

1. if `viscosity <= 0`, copy `temporary_velocity_*` to `velocity_*`,
2. otherwise copy `temporary_velocity_*` to `velocity_*`,
3. set `alpha = dt * viscosity / (cell_size * cell_size)`,
4. set `denom = 1 + 6 * alpha`,
5. run `diffuse_iterations` times:
   - parity 0 on x, y, z velocity,
   - apply x, y, z boundary kernels,
   - parity 1 on x, y, z velocity,
   - apply x, y, z boundary kernels.

Here:

- `src` is the fixed advected velocity in `temporary_velocity_*`,
- `dst` is the evolving iterate in `velocity_*`.

### 5. Projection

#### Theory

Projection solves

$$
\mathbf{w} = \mathbf{u} + \nabla p,
$$

with

$$
\nabla \cdot \mathbf{u} = 0.
$$

So $p$ satisfies

$$
\nabla^2 p = \nabla \cdot \mathbf{w}.
$$

The MAC divergence is

$$
d_{i,j,k} =
\frac{
u_{i+1,j,k} - u_{i,j,k} +
v_{i,j+1,k} - v_{i,j,k} +
w_{i,j,k+1} - w_{i,j,k}
}{h}.
$$

The pressure relaxation step is

$$
p_{i,j,k}^{m+1} =
\frac{
p_{i-1,j,k} + p_{i+1,j,k} +
p_{i,j-1,k} + p_{i,j+1,k} +
p_{i,j,k-1} + p_{i,j,k+1}
- h^2 d_{i,j,k}
}{6}.
$$

Then subtract the pressure gradient:

$$
u_{i,j,k} \leftarrow u_{i,j,k} - \frac{p_{i,j,k} - p_{i-1,j,k}}{h},
$$

$$
v_{i,j,k} \leftarrow v_{i,j,k} - \frac{p_{i,j,k} - p_{i,j-1,k}}{h},
$$

$$
w_{i,j,k} \leftarrow w_{i,j,k} - \frac{p_{i,j,k} - p_{i,j,k-1}}{h}.
$$

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

The step routine applies them in this order:

1. zero `temporary_pressure`,
2. compute `temporary_divergence`,
3. run `pressure_iterations` red-black pressure sweeps,
4. subtract the pressure gradient from `velocity_x`, `velocity_y`, `velocity_z`,
5. apply boundary kernels again.

### 6. Density Advection

#### Theory

Density advection solves

$$
\frac{\partial \rho}{\partial t} + \mathbf{u} \cdot \nabla \rho = 0
$$

with the projected velocity field. At the cell center

$$
\mathbf{x}_\rho = (i + 0.5, j + 0.5, k + 0.5),
$$

the backtraced point is

$$
\mathbf{x}_d = \mathbf{x}_\rho - \frac{\Delta t}{h} \mathbf{u}(\mathbf{x}_\rho),
$$

and the update is

$$
\rho^*(i,j,k) = \rho^n(\mathbf{x}_d).
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

The step routine does:

1. copy `density` to `temporary_previous_density`,
2. advect from `temporary_previous_density` into `temporary_density`.

### 7. Density Diffusion

#### Theory

Density diffusion solves

$$
\frac{\partial \rho}{\partial t} = \kappa \nabla^2 \rho
$$

with backward Euler:

$$
(I - \kappa \Delta t \nabla^2)\rho^{n+1} = \rho^*.
$$

The relaxation formula is the same as velocity diffusion, with

$$
a = \frac{\kappa \Delta t}{h^2}.
$$

#### CUDA Implementation

The same `diffuse_grid_kernel` is reused. The step routine does:

1. if `diffusion <= 0`, copy `temporary_density` to `density`,
2. otherwise copy `temporary_density` to `density`,
3. set `alpha = dt * diffusion / (cell_size * cell_size)`,
4. set `denom = 1 + 6 * alpha`,
5. run `diffuse_iterations` red-black sweeps on `density`, using `temporary_density` as the fixed source.

### 8. Step Order

#### Theory

The implemented operator sequence is

$$
\text{advect velocity}
\rightarrow
\text{diffuse velocity}
\rightarrow
\text{project}
\rightarrow
\text{advect density}
\rightarrow
\text{diffuse density}.
$$

#### CUDA Implementation

`stable_fluids_step_cuda` executes exactly:

1. copy current velocity to previous velocity buffers,
2. advect velocity to temporary velocity buffers,
3. apply velocity boundary kernels,
4. diffuse velocity into the persistent velocity buffers,
5. project the persistent velocity buffers,
6. copy current density to previous density buffer,
7. advect density to temporary density buffer,
8. diffuse density into the persistent density buffer.

## Reproduction Checklist

To reproduce this CUDA implementation, implement:

1. MAC staggered velocity storage,
2. trilinear interpolation on scalar and face grids,
3. collocated velocity reconstruction from MAC components,
4. semi-Lagrangian advection for `u`, `v`, `w`, and `density`,
5. red-black Gauss-Seidel diffusion,
6. MAC divergence, Poisson pressure solve, and pressure-gradient subtraction,
7. zero normal velocity boundary kernels,
8. the exact stage order given above.
