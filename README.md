# [SIGGRAPH 1999] Stable Fluids

## 1. Pipeline Overview

The Stable Fluids method advances an incompressible flow by composing a small number of robust operators. For a velocity field $\mathbf{u}$ and a transported scalar density $\rho$, one time step in this implementation is

$$
\mathbf{u}^n
\xrightarrow{\text{advect}}
\mathbf{u}^{*}
\xrightarrow{\text{diffuse}}
\mathbf{u}^{**}
\xrightarrow{\text{project}}
\mathbf{u}^{n+1},
$$

followed by

$$
\rho^n
\xrightarrow{\text{advect by } \mathbf{u}^{n+1}}
\rho^{*}
\xrightarrow{\text{diffuse}}
\rho^{n+1}.
$$

The two essential ideas are exactly the ones introduced in *Stable Fluids*:

- Semi-Lagrangian advection, which evaluates the old field at a backward-traced footpoint.
- Implicit linear solves for diffusion and projection, which remove the severe stability restrictions of explicit time stepping.

In this repository, source injection is handled outside the ABI step function. The CUDA step therefore assumes that any density or velocity sources have already been written into the current fields before the Stable Fluids update begins.

The implementation uses a staggered MAC layout:

- `density`: $(nx, ny, nz)$ at cell centers,
- `velocity_x`: $(nx + 1, ny, nz)$ on faces normal to the $x$ axis,
- `velocity_y`: $(nx, ny + 1, nz)$ on faces normal to the $y$ axis,
- `velocity_z`: $(nx, ny, nz + 1)$ on faces normal to the $z$ axis.

This choice is not merely an implementation detail. It is the discrete structure that makes the divergence and pressure gradient operators natural and local.

## 2. Method Details

### 2.1 Discrete State and Buffer Roles

At the mathematical level, the method advances two physical quantities:

- the velocity field $\mathbf{u} = (u, v, w)$,
- the passive scalar density $\rho$.

At the discrete level, the CUDA backend uses the following persistent arrays:

- `density`,
- `velocity_x`,
- `velocity_y`,
- `velocity_z`.

In addition, it receives temporary arrays that serve strictly numerical roles:

- one temporary velocity triplet,
- one previous velocity triplet,
- one temporary density field,
- one previous density field,
- one pressure field,
- one divergence field.

These temporary arrays are not extra physics. They exist because the Stable Fluids update repeatedly needs:

- a fixed old state for semi-Lagrangian sampling,
- a write destination for stage outputs,
- a pressure buffer for the Poisson solve,
- a divergence buffer for the right-hand side of the projection.

The linear storage order is x-fastest:

$$
\operatorname{index}(i,j,k;n_x,n_y) = (k\,n_y + j)\,n_x + i.
$$

This same indexing rule is used for the scalar grid and for each staggered component grid, with the only difference being the grid dimensions.

### 2.2 Continuous Advection and Its Discrete Form

The advection equation for a quantity $q$ transported by velocity $\mathbf{u}$ is

$$
\frac{\partial q}{\partial t} + \mathbf{u}\cdot\nabla q = 0.
$$

Stable Fluids replaces a forward Euler update with a backward characteristic trace. If $\mathbf{x}$ is the location at which the new value is desired, the method traces backward to

$$
\mathbf{x}_d = \mathbf{x} - \Delta t\,\mathbf{u}(\mathbf{x}),
$$

and then sets

$$
q^{n+1}(\mathbf{x}) = q^n(\mathbf{x}_d).
$$

In this implementation the fields are sampled in grid-index coordinates, not in physical coordinates. Since the physical cell width is $h = \text{cell\_size}$, the code uses

$$
\mathbf{x}_d = \mathbf{x} - \frac{\Delta t}{h}\,\mathbf{u}(\mathbf{x}).
$$

The factor $\Delta t / h$ is therefore not an arbitrary implementation trick. It is the exact conversion from physical velocity units to index-space displacement.

### 2.3 Continuous Diffusion and Its Discrete Form

For either velocity or density, the diffusion equation is

$$
\frac{\partial q}{\partial t} = \alpha \nabla^2 q,
$$

where $\alpha$ denotes either viscosity $\nu$ or scalar diffusivity $\kappa$.

Backward Euler gives

$$
\frac{q^{n+1} - q^*}{\Delta t} = \alpha \nabla^2 q^{n+1},
$$

which can be rearranged as

$$
\left(I - \alpha \Delta t \nabla^2\right) q^{n+1} = q^*.
$$

Using the standard 7-point Laplacian on a Cartesian grid gives the discrete point update

$$
q_{i,j,k}^{(m+1)} =
\frac{
q^*_{i,j,k}
 + a
\left(
q_{i-1,j,k}^{(m)} + q_{i+1,j,k}^{(m)}
+ q_{i,j-1,k}^{(m)} + q_{i,j+1,k}^{(m)}
+ q_{i,j,k-1}^{(m)} + q_{i,j,k+1}^{(m)}
\right)
}{
1 + 6a
},
$$

with

$$
a = \frac{\alpha \Delta t}{h^2}.
$$

This repository solves that system by red-black Gauss-Seidel relaxation.

### 2.4 Continuous Projection and Its Discrete Form

The defining incompressibility step in Stable Fluids is the Helmholtz-Hodge projection. One writes the intermediate velocity $\mathbf{w}$ as

$$
\mathbf{w} = \mathbf{u} + \nabla p,
$$

where $\mathbf{u}$ is divergence-free:

$$
\nabla \cdot \mathbf{u} = 0.
$$

Taking divergence yields the Poisson equation

$$
\nabla^2 p = \nabla \cdot \mathbf{w}.
$$

Once $p$ is found, the projected velocity is

$$
\mathbf{u} = \mathbf{w} - \nabla p.
$$

With the MAC discretization used here, the discrete divergence at a cell center is

$$
(\nabla \cdot \mathbf{w})_{i,j,k}
=
\frac{
u_{i+1,j,k} - u_{i,j,k}
+ v_{i,j+1,k} - v_{i,j,k}
+ w_{i,j,k+1} - w_{i,j,k}
}{h}.
$$

The discrete pressure-correction formulas are

$$
u_{i,j,k} \leftarrow u_{i,j,k} - \frac{p_{i,j,k} - p_{i-1,j,k}}{h},
$$

$$
v_{i,j,k} \leftarrow v_{i,j,k} - \frac{p_{i,j,k} - p_{i,j-1,k}}{h},
$$

$$
w_{i,j,k} \leftarrow w_{i,j,k} - \frac{p_{i,j,k} - p_{i,j,k-1}}{h}.
$$

The pressure solve itself is again carried out by red-black Gauss-Seidel:

$$
p_{i,j,k}^{(m+1)} =
\frac{
p_{i-1,j,k} + p_{i+1,j,k}
+ p_{i,j-1,k} + p_{i,j+1,k}
+ p_{i,j,k-1} + p_{i,j,k+1}
- h^2 d_{i,j,k}
}{6},
$$

where $d_{i,j,k}$ is the discrete divergence.

### 2.5 Boundary Conditions

The wall condition imposed by this implementation is zero normal velocity on the box boundary. Because the velocity is staggered, this means:

- set x-face velocity to zero on the two $x$ walls,
- set y-face velocity to zero on the two $y$ walls,
- set z-face velocity to zero on the two $z$ walls.

The density field is not given a separate reflective kernel. Instead, scalar interpolation and scalar diffusion both clamp their sample coordinates or neighbor indices to the valid range. This yields a simple box-constrained scalar evolution.

## 3. CUDA Implementation

### 3.1 Sampling Infrastructure

The following device code is the sampling foundation of the entire solver:

```cpp
__device__ int clampi(const int v, const int lo, const int hi) {
    return max(lo, min(hi, v));
}

__device__ float clampf(const float v, const float lo, const float hi) {
    return fmaxf(lo, fminf(hi, v));
}

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

__device__ float3 clamp_domain(const float3 p, const int nx, const int ny, const int nz) {
    return make_float3(
        clampf(p.x, 0.0f, static_cast<float>(nx)),
        clampf(p.y, 0.0f, static_cast<float>(ny)),
        clampf(p.z, 0.0f, static_cast<float>(nz)));
}

__device__ float3 sample_velocity(
    const float* const u, const float* const v, const float* const w, const float3 p, const int nx, const int ny, const int nz) {
    return make_float3(
        sample_u(u, p.x, p.y - 0.5f, p.z - 0.5f, nx, ny, nz),
        sample_v(v, p.x - 0.5f, p.y, p.z - 0.5f, nx, ny, nz),
        sample_w(w, p.x - 0.5f, p.y - 0.5f, p.z, nx, ny, nz));
}
```

This code directly realizes the theory from Sections 2.2 and 2.5.

- `sample_grid` is the full trilinear interpolation formula.
- `sample_scalar`, `sample_u`, `sample_v`, and `sample_w` only differ by grid extents.
- `sample_velocity` performs MAC-to-collocated reconstruction exactly by the half-cell shifts described in the theory.
- `clamp_domain` ensures that a backtraced characteristic stays inside the closed box $[0,nx] \times [0,ny] \times [0,nz]$ in index coordinates.

If one reproduces these functions faithfully, one reproduces the sampling core of the CUDA solver.

### 3.2 Velocity Boundary Kernels

The wall-condition kernels are:

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

Each kernel enforces the corresponding normal-velocity boundary condition and nothing more. This is the discrete realization of the box-wall condition from Section 2.5.

### 3.3 Velocity Advection Kernels

The three velocity-advection kernels are:

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

These three kernels are the exact MAC realization of the semi-Lagrangian velocity update:

1. Choose the geometric location of the face-centered unknown.
2. Reconstruct the full velocity vector there.
3. Trace backward by $\Delta t / h$.
4. Sample the corresponding old face-centered component at the backtraced point.

Nothing else happens in these kernels. They are the textbook semi-Lagrangian operator specialized to the three staggered face locations.

### 3.4 Scalar Advection Kernel

Density advection is:

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

This is exactly the same method as velocity advection, but now the geometric location is the cell center

$$
\left(i+\frac12, j+\frac12, k+\frac12\right),
$$

and the sampled quantity is the cell-centered scalar field. The final `fmaxf(0.0f, ...)` enforces nonnegative density.

### 3.5 Diffusion Kernel

The diffusion kernel is:

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

Its numerical meaning is exactly the formula from Section 2.3:

- `src` is the fixed right-hand side $q^*$,
- `dst` is the mutable iterate $q^{(m)}$,
- `alpha` is $a = \alpha \Delta t / h^2$,
- `denom` is $1 + 6a$,
- `parity` selects red or black cells in the Gauss-Seidel sweep.

The kernel uses `fetch_clamped(dst, ...)` for the six neighbors, which means the current iterate is read with boundary clamping.

### 3.6 Divergence, Pressure, and Gradient Kernels

The projection kernels are:

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

These kernels implement the projection exactly as described in Section 2.4:

- `compute_divergence_kernel` builds the cell-centered right-hand side,
- `pressure_rbgs_kernel` solves the scalar Poisson equation,
- `subtract_gradient_*_kernel` converts the scalar pressure into a face-centered gradient correction.

The formulas in the theory section are therefore not merely related to this code. They are the code.

### 3.7 The Complete CUDA Step Routine

The full CUDA routine is `stable_fluids_step_cuda` in [`backend-cuda.cu`](C:\Users\xayah\Desktop\awesome-smoke-simulation\001-stable-fluids\backend-cuda.cu). Its execution order is:

#### 3.7.1 Descriptor Unpacking

The function reads:

- the dimensions `nx`, `ny`, `nz`,
- the physical parameters `cell_size`, `dt`, `viscosity`, `diffusion`,
- the iteration counts `diffuse_iterations`, `pressure_iterations`,
- all persistent and temporary field pointers,
- the CUDA launch block,
- the CUDA stream.

It then computes:

- `dt_over_h = dt / cell_size`,
- `inv_h = 1 / cell_size`,
- launch grids for scalar, x-face, y-face, and z-face domains.

#### 3.7.2 Velocity Advection Stage

Within the NVTX range `"stable.step.advect_velocity"`, the code performs:

1. three asynchronous copies from persistent velocity into the previous-velocity triplet,
2. three advection kernel launches into the temporary-velocity triplet,
3. three boundary kernel launches on the temporary-velocity triplet.

At the end of this stage:

- `velocity_*_previous` holds $\mathbf{u}^n$,
- `velocity_*_temporary` holds the advected field $\mathbf{u}^{*}$.

#### 3.7.3 Velocity Diffusion Stage

Within `"stable.step.diffuse_velocity"`:

- if viscosity is zero or negative, the temporary velocity triplet is copied directly into the persistent velocity triplet;
- otherwise the temporary triplet is copied into the persistent velocity triplet, then red-black diffusion sweeps are applied to the persistent triplet, using the temporary triplet as the fixed source term.

This means:

- source of the implicit solve: `velocity_*_temporary`,
- evolving iterate: `velocity_*_field`.

#### 3.7.4 Projection Stage

Within `"stable.step.project"`:

1. zero `pressure`,
2. compute `divergence` from the current persistent velocity,
3. run `pressure_iterations` red-black pressure sweeps,
4. subtract the pressure gradient from the persistent velocity triplet,
5. reapply velocity wall conditions.

At the end of this stage the persistent velocity triplet contains the divergence-free velocity $\mathbf{u}^{n+1}$.

#### 3.7.5 Density Advection Stage

Within `"stable.step.advect_density"`:

1. copy `density_field` into `density_previous`,
2. advect density from `density_previous` into `density_temporary` using the projected persistent velocity triplet.

At the end of this stage `density_temporary` holds $\rho^{*}$.

#### 3.7.6 Density Diffusion Stage

Within `"stable.step.diffuse_density"`:

- if diffusion is zero or negative, copy `density_temporary` into `density_field`,
- otherwise copy `density_temporary` into `density_field`, then run red-black diffusion sweeps on `density_field` using `density_temporary` as the fixed right-hand side.

At the end of this stage `density_field` contains $\rho^{n+1}$.

## 4. Reproduction Checklist

To reproduce this CUDA implementation without consulting the source, the reader must implement exactly the following components:

1. MAC storage with shapes `(nx+1,ny,nz)`, `(nx,ny+1,nz)`, `(nx,ny,nz+1)`, `(nx,ny,nz)`,
2. x-fastest linear indexing,
3. clamped trilinear interpolation for scalar and face-centered grids,
4. MAC-to-collocated velocity reconstruction by half-cell shifts,
5. face-centered semi-Lagrangian advection for $u$, $v$, and $w$,
6. cell-centered semi-Lagrangian advection for $\rho$,
7. red-black Gauss-Seidel diffusion using the 7-point stencil,
8. MAC divergence, scalar Poisson solve, and MAC gradient subtraction,
9. zero normal velocity boundary kernels,
10. the exact stage ordering:
   1. advect velocity,
   2. diffuse velocity,
   3. project,
   4. advect density,
   5. diffuse density.

That is the complete Stable Fluids CUDA algorithm implemented in this repository.
